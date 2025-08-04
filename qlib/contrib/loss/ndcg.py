import torch
import numpy as np
from itertools import combinations
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
import math
import warnings

def get_pairs_index(n, device):
    """
    获取所有样本对的索引, 不需要保证有序
    """
     # 创建索引矩阵
    i_indices = torch.arange(n, device=device).unsqueeze(1).expand(n, n)
    j_indices = torch.arange(n, device=device).unsqueeze(0).expand(n, n)
    
    # 创建上三角矩阵掩码（避免重复对）
    mask = i_indices < j_indices
    
    # 获取有效的索引对
    i_valid = i_indices[mask]
    j_valid = j_indices[mask]   
    
    return torch.stack([i_valid, j_valid], dim=1)

def get_pairs_parallel(real_scores):
    """
    并行化获取所有样本对，优化内存使用
    """
    n = len(real_scores)
    device = real_scores.device
    
    # 创建索引矩阵
    i_indices = torch.arange(n, device=device).unsqueeze(1).expand(n, n)
    j_indices = torch.arange(n, device=device).unsqueeze(0).expand(n, n)
    
    # 创建上三角矩阵掩码（避免重复对）
    mask = i_indices < j_indices
    
    # 获取有效的索引对
    i_valid = i_indices[mask]
    j_valid = j_indices[mask]
    
    # 比较真实分数，确定优胜者
    real_i = real_scores[i_valid]
    real_j = real_scores[j_valid]
    
    # 如果i > j，则保持(i,j)；否则交换为(j,i)
    swap_mask = real_i <= real_j
    i_final = torch.where(swap_mask, j_valid, i_valid)
    j_final = torch.where(swap_mask, i_valid, j_valid)
    
    return torch.stack([i_final, j_final], dim=1)

def single_dcg_vectorized(real_scores, positions, k, linear=False):
    """
    向量化计算DCG贡献
    """
    # 只有位置 < k 的才有贡献
    valid_mask = positions < k
    dcg_values = torch.zeros_like(positions, dtype=torch.float32)
    
    if valid_mask.any():
        valid_positions = positions[valid_mask]
        valid_scores = real_scores[valid_mask]
        if linear:
            dcg_values[valid_mask] = valid_scores.float() / torch.log2(valid_positions.float() + 2)
        else:
            dcg_values[valid_mask] = (2 ** valid_scores.float() - 1) / torch.log2(valid_positions.float() + 2)
    
    return dcg_values

def calculate_idcg_optimized(real_scores, k, linear=False):
    """
    优化的IDCG计算
    """
    # 按照真实表现从高到低排序
    sorted_scores, _ = torch.sort(real_scores, descending=True)
    
    # 只取前k个
    top_k_scores = sorted_scores[:k]
    positions = torch.arange(k, device=real_scores.device, dtype=torch.float32)
    
    # 计算IDCG
    if linear:
        idcg = torch.sum(top_k_scores / torch.log2(positions + 2))
    else:
        idcg = torch.sum((2 ** top_k_scores - 1) / torch.log2(positions + 2))
    
    return idcg

def calculate_ndcg_optimized(y_true, y_pred, n_layer, linear=False):
    """
    优化的NDCG计算
    """
    n = len(y_true)
    k = n // n_layer
    
    # 将真实收益率标准化到[0, n_layer-1]区间
    y_min, y_max = y_true.min(), y_true.max()
    if y_max > y_min:
        real_scores = (y_true - y_min) / (y_max - y_min) * (n_layer)
    else:
        real_scores = torch.zeros_like(y_true)
    
    # 获取预测排名
    _, pred_indices = torch.sort(y_pred, descending=True)
    i_rank = torch.empty_like(pred_indices)
    i_rank[pred_indices] = torch.arange(n, device=y_pred.device)
    
    # 计算DCG@k
    top_k_mask = i_rank < k
    if top_k_mask.any():
        top_k_scores = real_scores[top_k_mask]
        top_k_ranks = i_rank[top_k_mask]
        if linear:
            dcg = torch.sum(top_k_scores / torch.log2(top_k_ranks.float() + 2))
        else:
            dcg = torch.sum((2 ** top_k_scores - 1) / torch.log2(top_k_ranks.float() + 2))
    else:
        dcg = torch.tensor(0.0, device=y_pred.device)
    
    # 计算IDCG
    idcg = calculate_idcg_optimized(real_scores, k, linear)
    
    # 避免除零
    if idcg > 0:
        return dcg / idcg
    else:
        return torch.tensor(0.0, device=y_pred.device)

def compute_lambda_gradients(y_true, y_pred, n_layer, sigma=3.03, linear=False):
    """
    Lambda Net: 直接计算并返回用于梯度调节的lambda值
    """
    n = len(y_true)
    k = n // n_layer
    device = y_pred.device
    
    # 初始化lambda梯度
    lambda_gradients = torch.zeros(n, device=device, dtype=torch.float32)
    
    # 标准化真实分数
    y_min, y_max = y_true.min(), y_true.max()
    
    if y_max > y_min:
        real_scores = (y_true - y_min) / (y_max - y_min) * n_layer
        real_scores = real_scores.float()
    else:
        warnings.warn("We get the same prediction for all stocks!!!")
        real_scores = torch.zeros_like(y_true, dtype=torch.float32)
    
    # 获取预测排名
    _, pred_indices = torch.sort(y_pred, descending=True)
    i_rank = torch.empty_like(pred_indices)
    i_rank[pred_indices] = torch.arange(n, device=device)
    
    # 计算IDCG用于归一化
    idcg = calculate_idcg_optimized(real_scores, k, linear)
    
    if idcg <= 0:
        warnings.warn('idcg <= 0, check your data')

        return lambda_gradients
    
    # 获取所有有效对
    pairs = get_pairs_parallel(real_scores)
    
    if len(pairs) == 0:
        warnings.warn('no pairs, check your data')

        return lambda_gradients
    
    # 批量处理所有对
    i_indices = pairs[:, 0]
    j_indices = pairs[:, 1]
    
    # 获取对应的分数和排名
    real_i = real_scores[i_indices]
    real_j = real_scores[j_indices]
    pred_i = y_pred[i_indices]
    pred_j = y_pred[j_indices]
    rank_i = i_rank[i_indices]
    rank_j = i_rank[j_indices]
    
    # 计算S_ij (真实标签差异的符号)
    S_ij = torch.sign(real_i - real_j)
    
    # 计算sigmoid部分 - Lambda Net的核心公式
    pred_diff = pred_i - pred_j
    sigmoid_part = torch.sigmoid(-sigma * pred_diff)
    
    # 计算NDCG增益差异
    # (2 ** valid_scores - 1) / torch.log2(valid_positions.float() + 2)
    dcg_i_at_i = single_dcg_vectorized(real_i, rank_i, k, linear)
    dcg_i_at_j = single_dcg_vectorized(real_i, rank_j, k, linear)
    dcg_j_at_i = single_dcg_vectorized(real_j, rank_i, k, linear)
    dcg_j_at_j = single_dcg_vectorized(real_j, rank_j, k, linear)
    
    # Delta NDCG = |交换位置后的NDCG变化|
    delta_ndcg = torch.abs((dcg_i_at_j + dcg_j_at_i) - (dcg_i_at_i + dcg_j_at_j)) / idcg
    # delta_ndcg_ = ((2 ** real_i - 1) / torch.log2(rank_i.float() + 2) - (2 ** real_j - 1) / torch.log2(rank_j.float() + 2)) / idcg
    # Lambda Net梯度公式: λ_i = - σ * sigmoid(-σ(s_i - s_j)) * |ΔNDCG|
    lambda_contribution =  - sigma * sigmoid_part * delta_ndcg
    
    # 聚合到对应的样本
    lambda_gradients.scatter_add_(0, i_indices, lambda_contribution)
    lambda_gradients.scatter_add_(0, j_indices, -lambda_contribution)
    
    return lambda_gradients

def compute_delta_ndcg(y_pred, y_true, n_layer, sigma=3.03, linear=True):
    """
    计算每个股票的Delta NDCG
    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_
        n_layer (_type_): _description_
        sigma (float, optional): _description_. Defaults to 3.03.
    """
    n = len(y_true)
    k = n // n_layer
    device = y_pred.device
    
    # 初始化lambda梯度
    delta_ndcg_ = torch.zeros((n, n), device=device, dtype=torch.float32)
    
    # 标准化真实分数
    y_min, y_max = y_true.min(), y_true.max()
    if y_max > y_min:
        real_scores = (y_true - y_min) / (y_max - y_min) * n_layer 
    else:
        real_scores = torch.zeros_like(y_true)
    
    # 获取预测排名
    _, pred_indices = torch.sort(y_pred, descending=True)
    i_rank = torch.empty_like(pred_indices)
    i_rank[pred_indices] = torch.arange(n, device=device)
    
    # 计算IDCG用于归一化
    idcg = calculate_idcg_optimized(real_scores, k, linear)
    
    if idcg <= 0:
        warnings.warn('idcg <= 0, check your data')

        return torch.ones(n, device=device, dtype=torch.float32)
    
    # 获取所有有效对
    pairs = get_pairs_index(n, device)
    
    if len(pairs) == 0:
        warnings.warn('no pairs, check your data')

        return torch.ones(n, device=device, dtype=torch.float32)
    
    # 批量处理所有对
    i_indices = pairs[:, 0]
    j_indices = pairs[:, 1]
    
    # 获取对应的分数和排名
    real_i = real_scores[i_indices]
    real_j = real_scores[j_indices]
    pred_i = y_pred[i_indices]
    pred_j = y_pred[j_indices]
    rank_i = i_rank[i_indices]
    rank_j = i_rank[j_indices]

    # 计算NDCG增益差异
    dcg_i_at_i = single_dcg_vectorized(real_i, rank_i, k, linear)
    dcg_i_at_j = single_dcg_vectorized(real_i, rank_j, k, linear)
    dcg_j_at_i = single_dcg_vectorized(real_j, rank_i, k, linear)
    dcg_j_at_j = single_dcg_vectorized(real_j, rank_j, k, linear)
    delta_ndcg = torch.abs((dcg_i_at_j + dcg_j_at_i) - (dcg_i_at_i + dcg_j_at_j)) / idcg
    
    delta_ndcg_[i_indices, j_indices] = delta_ndcg
    delta_ndcg_[j_indices, i_indices] = delta_ndcg
    
    return delta_ndcg_.sum(dim=1)


class LambdaNetOptimizer(Optimizer):
    """
    Lambda Net优化器：直接基于NDCG@k计算和应用梯度
    
    Args:
        params: 模型参数
        lr: 学习率 (default: 1e-3)
        n_layer: 分层数，用于计算k=n//n_layer (default: 5)
        sigma: Lambda Net的缩放参数 (default: 3.03)
        lambda_weight: Lambda梯度的权重 (default: 1.0)
        eps: 数值稳定性参数 (default: 1e-8)
        weight_decay: L2正则化系数 (default: 0)
        momentum: 动量参数 (default: 0)
        dampening: 动量阻尼 (default: 0)
    """
    
    def __init__(self, params, lr=1e-3, n_layer=5, sigma=3.03, lambda_weight=1.0,
                 eps=1e-8, weight_decay=0, momentum=0, dampening=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= momentum:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= dampening:
            raise ValueError(f"Invalid dampening value: {dampening}")
            
        defaults = dict(lr=lr, n_layer=n_layer, sigma=sigma, lambda_weight=lambda_weight,
                       eps=eps, weight_decay=weight_decay, momentum=momentum,
                       dampening=dampening)
        super(LambdaNetOptimizer, self).__init__(params, defaults)
        
        # 存储输出层参数的引用，用于应用Lambda梯度
        self.output_param = None
        self.last_lambda_grads = None
    
    def register_output_param(self, output_param):
        """
        注册输出层参数，用于应用Lambda梯度
        """
        self.output_param = output_param
    
    def compute_lambda_step(self, y_true, y_pred):
        """
        计算Lambda梯度，这是Lambda Net的核心步骤
        """
        group = self.param_groups[0]
        lambda_grads = compute_lambda_gradients(
            y_true, y_pred, 
            group['n_layer'], 
            group['sigma']
        )
        self.last_lambda_grads = lambda_grads
        return lambda_grads
    
    @torch.no_grad()
    def step(self, y_true=None, y_pred=None, closure=None):
        """
        执行一步优化
        
        Args:
            y_true: 真实标签
            y_pred: 预测值
            closure: 重新计算损失的函数（可选）
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # 如果提供了标签和预测值，计算Lambda梯度
        if y_true is not None and y_pred is not None:
            lambda_grads = self.compute_lambda_step(y_true, y_pred)
            
            # 将Lambda梯度应用到输出层
            if y_pred.grad is not None:
                y_pred.grad += self.param_groups[0]['lambda_weight'] * lambda_grads
            else:
                y_pred.grad = self.param_groups[0]['lambda_weight'] * lambda_grads
        
        # 执行参数更新
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                
                # 应用权重衰减
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                # 应用动量
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p)
                        buf.mul_(momentum).add_(grad)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(grad, alpha=1 - dampening)
                    grad = buf
                
                # 更新参数
                p.add_(grad, alpha=-group['lr'])
        
        return loss
    
    def get_lambda_info(self):
        """
        获取最近一次计算的Lambda梯度信息
        """
        if self.last_lambda_grads is not None:
            return {
                'lambda_norm': self.last_lambda_grads.norm().item(),
                'lambda_mean': self.last_lambda_grads.mean().item(),
                'lambda_std': self.last_lambda_grads.std().item(),
                'lambda_max': self.last_lambda_grads.max().item(),
                'lambda_min': self.last_lambda_grads.min().item(),
            }
        return None

class LambdaAdam(Optimizer):
    """
    Lambda Net + Adam优化器的结合
    结合Adam的自适应学习率和Lambda Net的NDCG梯度
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0, n_layer=5, sigma=3.03, lambda_weight=1.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       n_layer=n_layer, sigma=sigma, lambda_weight=lambda_weight)
        super(LambdaAdam, self).__init__(params, defaults)
        
        self.last_lambda_grads = None
    
    def compute_lambda_step(self, y_true, y_pred):
        """计算Lambda梯度"""
        group = self.param_groups[0]
        lambda_grads = compute_lambda_gradients(
            y_true, y_pred, 
            group['n_layer'], 
            group['sigma']
        )
        self.last_lambda_grads = lambda_grads
        return lambda_grads
    
    @torch.no_grad()
    def step(self, y_true=None, y_pred=None, closure=None):
        """执行一步优化"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # 计算并应用Lambda梯度
        if y_true is not None and y_pred is not None:
            lambda_grads = self.compute_lambda_step(y_true, y_pred)
            
            if y_pred.grad is not None:
                y_pred.grad += self.param_groups[0]['lambda_weight'] * lambda_grads
            else:
                y_pred.grad = self.param_groups[0]['lambda_weight'] * lambda_grads
        
        # Adam参数更新
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                
                state = self.state[p]
                
                # 初始化状态
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # 权重衰减
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                
                # 更新偏置第一矩估计
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # 更新偏置第二矩估计
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # 计算步长
                step_size = group['lr'] / bias_correction1
                bias_correction2_sqrt = math.sqrt(bias_correction2)
                
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group['eps'])
                
                # 更新参数
                p.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss
    
    def get_lambda_info(self):
        """获取Lambda梯度信息"""
        if self.last_lambda_grads is not None:
            return {
                'lambda_norm': self.last_lambda_grads.norm().item(),
                'lambda_mean': self.last_lambda_grads.mean().item(),
                'lambda_std': self.last_lambda_grads.std().item(),
            }
        return None

# 使用示例
def training_example():
    """
    Lambda Net优化器的完整使用示例
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    class AssetRankingNet(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim=64):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(hidden_dim, 1)
            )
        
        def forward(self, x):
            return self.layers(x).squeeze(-1)
    
    # 初始化
    input_dim = 20
    n_assets = 100
    n_layer = 5
    
    model = AssetRankingNet(input_dim).to(device)
    
    # 创建Lambda Net优化器
    optimizer = LambdaAdam(
        model.parameters(),
        lr=0.001,
        n_layer=n_layer,
        sigma=3.03,
        lambda_weight=0.5  # Lambda梯度权重
    )
    
    # 模拟数据
    X = torch.randn(n_assets, input_dim, device=device)
    y_true = torch.randn(n_assets, device=device)
    print("开始Lambda Net + Adam训练...")
    print(f"资产数量: {n_assets}, 分层数: {n_layer}, Top-K: {n_assets//n_layer}")
    
    for epoch in range(20):
        optimizer.zero_grad()
        
        # 前向传播      
        y_pred = model(X)
        lambda_grads = compute_lambda_gradients(y_true, y_pred, n_layer, sigma=3.03)
        
        # 计算基础损失（可选，用于监控）
        base_loss = F.mse_loss(y_pred, y_true)
        base_loss.backward()
        
        # Lambda Net优化步骤
        optimizer.step(y_true=y_true, y_pred=y_pred)
        
        # 监控指标
        if epoch % 5 == 0:
            with torch.no_grad():
                ndcg = calculate_ndcg_optimized(y_true, y_pred, n_layer)
                lambda_info = optimizer.get_lambda_info()
                
                print(f"Epoch {epoch:2d}: "
                      f"Loss={base_loss.item():.4f}, "
                      f"NDCG@{n_assets//n_layer}={ndcg.item():.4f}")
                
                if lambda_info:
                    print(f"         Lambda - norm: {lambda_info['lambda_norm']:.4f}, "
                          f"mean: {lambda_info['lambda_mean']:.4f}")
    
    return model, optimizer

def ranknet_cross_entropy_loss(y_pred, y_true, sigma=1.0):
    """
    RankNet交叉熵损失函数
    C = 0.5 * (1 - Sij) * sigma * y_pred_diff - F.logsigmoid(-sigma * y_pred_diff)

    Parameters
    ----------
    y_pred : torch.Tensor
        预测分数，形状为 (batch_size,)
    y_true : torch.Tensor
        真实标签，形状为 (batch_size,)
    sigma : float
        温度参数，控制sigmoid的陡峭程度，默认为1.0

    Returns
    -------
    torch.Tensor
        RankNet交叉熵损失
    """
    device = y_pred.device
    n = len(y_pred)

    if n < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # 创建所有可能的配对
    i_indices = torch.arange(n, device=device).unsqueeze(1).expand(n, n)
    j_indices = torch.arange(n, device=device).unsqueeze(0).expand(n, n)

    # 只考虑i != j的配对
    mask = i_indices != j_indices
    i_valid = i_indices[mask]
    j_valid = j_indices[mask]

    # 获取对应的预测值和真实标签
    pred_i = y_pred[i_valid]
    pred_j = y_pred[j_valid]
    true_i = y_true[i_valid]
    true_j = y_true[j_valid]

    # 计算真实标签的相对关系 S_ij
    # S_ij = 1 if true_i > true_j, -1 if true_i < true_j, 0 if true_i == true_j
    S_ij = torch.sign(true_i - true_j)

    # 只保留有明确排序关系的配对（S_ij != 0）
    valid_pairs_mask = S_ij != 0
    if not valid_pairs_mask.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    S_ij = S_ij[valid_pairs_mask]
    P_ij = (S_ij + 1)/2
    pred_i = pred_i[valid_pairs_mask]
    pred_j = pred_j[valid_pairs_mask]

    # 计算预测分数差异
    pred_diff = pred_i - pred_j

    # RankNet交叉熵损失公式
    # Loss = -S_ij * sigma * (pred_i - pred_j) + log(1 + exp(sigma * (pred_i - pred_j)))
    # loss = - sij * sigma * pred_diff + log(1 + exp(sigma * pred_diff))

        
    # 使用数值稳定的实现
    loss_per_pair = - P_ij * sigma * pred_diff + F.softplus(sigma * pred_diff)

    # 返回平均损失
    return loss_per_pair.mean()

def ranknet_cross_entropy_loss_simple(y_pred, y_true, sigma=1.0):
    """
    简化版RankNet交叉熵损失函数，只考虑相邻排序对

    Parameters
    ----------
    y_pred : torch.Tensor
        预测分数，形状为 (batch_size,)
    y_true : torch.Tensor
        真实标签，形状为 (batch_size,)
    sigma : float
        温度参数，默认为1.0

    Returns
    -------
    torch.Tensor
        简化的RankNet交叉熵损失
    """
    device = y_pred.device
    n = len(y_pred)

    if n < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # 按真实标签排序
    sorted_indices = torch.argsort(y_true, descending=True)
    sorted_pred = y_pred[sorted_indices]

    # 计算相邻对的损失
    losses = []
    for i in range(n - 1):
        # 对于排序后的序列，前面的应该比后面的分数高
        pred_diff = sorted_pred[i] - sorted_pred[i + 1]
        loss = F.softplus(-sigma * pred_diff)  # log(1 + exp(-sigma * pred_diff))
        losses.append(loss)

    if losses:
        return torch.stack(losses).mean()
    else:
        return torch.tensor(0.0, device=device, requires_grad=True)

if __name__ == "__main__":
    model, optimizer = training_example()