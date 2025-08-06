import torch
import torch.nn.functional as F


def quantile_loss(pred, label, tau=0.5):
    loss = torch.max(tau * (label - pred), (tau - 1) * (label - pred))

    return torch.mean(loss)          

def coverage(pred, label, tau):
    coverage = (label <= pred).float().mean()
    
    return coverage - tau   
    

def ranking_loss(pred, label, lambda_reg=0.1):
    """Ranking loss with pointwise regression and pairwise ranking"""
    # Pointwise regression loss
    pointwise_loss = F.mse_loss(pred, label)
    
    # Pairwise ranking loss
    # pred, label: [N]
    pred_diff = pred.unsqueeze(0) - pred.unsqueeze(1)   # [N, N]
    label_diff = label.unsqueeze(0) - label.unsqueeze(1)  # [N, N]
    # 排除对角线元素
    mask = ~torch.eye(pred.size(0), dtype=torch.bool, device=pred.device)  # [N, N]
    # 计算 pairwise loss
    pairwise_loss = F.relu(-pred_diff * label_diff)[mask].mean()
        
    return pointwise_loss + lambda_reg * pairwise_loss


def pairwise_loss(pred, label):
    pred_diff = pred.unsqueeze(0) - pred.unsqueeze(1)   # [N, N]
    label_diff = label.unsqueeze(0) - label.unsqueeze(1)  # [N, N]
    # 排除对角线元素
    mask = ~torch.eye(pred.size(0), dtype=torch.bool, device=pred.device)  # [N, N]
    # 计算 pairwise loss
    return F.relu(-pred_diff * label_diff)[mask].mean()


def mse(pred, label):
    return torch.mean((pred - label) ** 2)

def wmse(pred, label, tau=0.5):
    weight = torch.argsort(
        torch.argsort(label)
    )
    weight = weight / torch.max(weight)
    weight = torch.exp(
        (1 - weight) * torch.log(torch.tensor(0.5)) / tau)

def ic_loss(pred, label):
    return -torch.corrcoef(
        torch.stack((pred, label), dim=0)
    )[0, 1]

def rank_data(x):
    return x.argsort().argsort().float()

def rankic_loss(pred, label):
    pred_rank = rank_data(pred)
    label_rank = rank_data(label)
    return -torch.corrcoef(torch.stack((pred_rank, label_rank), dim=0))[0, 1]

def topk_return(pred, label, k=10):
    
    ...

def topk_ic_loss(pred, label, k=10):
    """
    计算预测值和真实标签在预测前 1/k 比例内的 rank IC（Spearman相关）。

    参数:
        pred: (N,) 预测收益率张量
        label: (N,) 实际收益率张量
        k: 整数，例如 k=5 表示取 top 20%
    """
    N = pred.shape[0]
    topk_num = max(1, int(N / k))  # 至少选一个

    # 取预测值中 topk_num 最大的样本的索引
    _, topk_idx = torch.topk(pred, topk_num)

    # 获取对应的预测值和标签
    pred_topk = pred[topk_idx]
    label_topk = label[topk_idx]

    # 计算  IC
    return -torch.corrcoef(torch.stack((pred_topk, label_topk), dim=0))[0, 1]

def topk_rankic_loss(pred, label, k=10):
    # rank 处理

    N = pred.shape[0]
    topk_num = max(1, int(N / k))  # 至少选一个

    # 取预测值中 topk_num 最大的样本的索引
    _, topk_idx = torch.topk(pred, topk_num)

    # 获取对应的预测值和标签
    pred_topk = rank_data(pred[topk_idx])
    label_topk = rank_data(label[topk_idx])
    
    # 计算  IC
    return -torch.corrcoef(torch.stack((pred_topk, label_topk), dim=0))[0, 1]


def ndcg_loss(pred, label):
    ...

#%% 基于NDCG@k计算λ
# 记录A真实表现优于B的样本对
def get_pairs(real_scores):

    pairs = []
    for i, j in combinations(range(len(real_scores)), 2):
        if real_scores[i] > real_scores[j]: pairs.append((i, j))
        else: pairs.append((j, i))
        
    return pairs

# 计算将第i名的真实标签放在第j名对NDCG@k的贡献
def single_dcg(real_scores, i, j, k):
    
    # 当目标位置j在截断k之后，贡献清零
    if j < k: return (2 ** real_scores[i] - 1) / np.log2(j + 2)
    else: return 0
    
# 计算截断的理想折损累计损益
def calculate_idcg(real_scores, k):
    
    # 按照真实表现从高到低排序获取排名
    i_rank = real_scores.argsort(descending=True).argsort()

    # 标记真实表现位于第一层的位置
    mask_1st_layer = (i_rank < k)

    # 用上述标记截取真实表现及其排序
    r_scores_by_r = real_scores[mask_1st_layer]
    i_rank = i_rank[mask_1st_layer]
    
    # 计算理想状态下的DCG@k
    idcg = torch.nansum((2 ** r_scores_by_r - 1) / (torch.log2(i_rank + 2)))
    
    return idcg

# 计算截断的归一化折损累计损益
def calculate_ndcg(y_true, y_pred, n_layer):

    # 获取输入数据的维度，初始化λ_i，确认多头层的资产数目
    n = len(y_true)
    k = n // n_layer
    
    # 将真实收益率划分为n_layer层；最高分为n_layer - 1，最低分为0
    real_scores = (y_true - y_true.min()) / (y_true.max() - y_true.min()) * (n_layer - 1)
    
    # 按照因子从高到低（连续两次调用argsort实现）获取排名
    i_rank = y_pred.argsort(descending=True).argsort()
    
    # 标记因子位于第一层的位置
    mask_1st_layer = (i_rank < k)
    
    # 用因子排序的标记矩阵截取标签及因子的排序
    r_scores_by_r = real_scores[mask_1st_layer]
    i_rank = i_rank[mask_1st_layer]
    
    # 计算NDCG@k的分子
    dcg = torch.nansum((2 ** r_scores_by_r - 1) / (torch.log2(i_rank + 2)))
    
    # 计算分母IDCG
    idcg = calculate_idcg(real_scores, k)
    
    return dcg / idcg

def compute_lambda(y_true, y_pred, n_layer, sigma=3.03):
    
    # 获取输入数据的维度，初始化λ_i，确认多头层的资产数目
    n = len(y_true)
    λ_i = torch.zeros(n).to(device, dtype=torch.float32)
    k = n // n_layer
    
    # 梯度缩放系数
    
    # 将真实收益率划分为n_layer层；最高分为n_layer - 1，最低分为0
    ...