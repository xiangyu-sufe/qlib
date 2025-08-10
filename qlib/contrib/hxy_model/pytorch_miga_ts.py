from typing import List, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

class AddGateFusion(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()

    def forward(self, price_feat, news_feat, mask):
        # price_feat, news_feat: [N, D]
        # mask: [N, 1], 0/1
        gated_news = mask * news_feat
        return price_feat + gated_news

class LearnableAddGateFusion(nn.Module):
    def __init__(self, feat_dim, hidden_dim=64):
        super().__init__()
        self.proj_news = nn.Linear(feat_dim, feat_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(feat_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim),
            nn.Sigmoid()
        )

    def forward(self, price_feat, news_feat, mask):
        news_feat = self.proj_news(news_feat)
        gate_input = torch.cat([price_feat, mask], dim=-1)
        g = self.gate_net(gate_input) * mask  # mask 保证缺新闻时 g=0
        return price_feat + g * news_feat


class MIGAB1(nn.Module):
    """
        B1: price + news 没有任何交互
        
         可选是否预加载已经训练好的 GRU 价格模型
    """
    def __init__(self, price_dim: int,
                 news_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.0,
                 frozen: bool = False,
                 model_path: Optional[str] = None,
                 padding_method: str = 'zero',
                 min_news: int = 1,):
        super().__init__()
        self.min_news = min_news
        self.hidden_dim = hidden_dim
        self.news_count = []
        self.stock_count = []
        self.gru_price = nn.GRU(
            input_size=price_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.gru_news = nn.GRU(
            input_size=news_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.padding_method = padding_method
        if self.padding_method == 'learn':
            self.padding = nn.Parameter(torch.zeros(1, news_dim))
            nn.init.xavier_uniform_(self.padding)
        if frozen:
            assert model_path is not None, "model_path must be specified when frozen is True"
            self.gru_price.load_state_dict(torch.load(model_path))
        self.ln = nn.LayerNorm(hidden_dim*2)
        self.fc_out = nn.Linear(hidden_dim*2, 1)

    def reset_count(self):
        self.news_count = []
        self.stock_count = []
        
    def update_count(self, news_count, stock_count):
        self.news_count.append(news_count)
        self.stock_count.append(stock_count)
    
    @property
    def news_coverage_ratio(self):
        return np.mean(self.news_count) / np.mean(self.stock_count)
    
    def forward(self, price: torch.Tensor, news: torch.Tensor, mask: torch.Tensor):
        # 最少新闻条数
        # 少于这个新闻数量的直接不进新闻 GRU
        N_price, T_price, D_price = price.shape
        N_news, T_news, D_news = news.shape
        assert N_price == N_news, "price and news must have the same batch size"
        if self.min_news != -1:
            # 开启新闻数量限制
            news_sum = (~mask).sum(dim=1) # N 
            news_sum_mask = news_sum < self.min_news # N
            news = news[~news_sum_mask] # 只选取新闻数量大于 min_news 的
            mask = mask[~news_sum_mask]    
        else:
            news_sum_mask = torch.zeros(N_price, dtype=torch.bool, device=news.device)
        self.update_count((~news_sum_mask).sum().item(), N_price)
        
        if self.padding_method == 'zero':
            news[mask] = 0
        elif self.padding_method == 'learn':
            news[mask] = self.padding
        else:
            raise ValueError(f"Unknown padding method: {self.padding_method}")
        
        price_out, _ = self.gru_price(price) # N , D
        price_out = price_out[:, -1, :]
        news_out = torch.zeros((N_price, self.hidden_dim), device=news.device)
        news, _ = self.gru_news(news) # N, D
        news = news[:, -1, :]
        news_out[~news_sum_mask] = news
        
        price_news = torch.cat([price_out, news_out], dim=1)
        price_news = self.ln(price_news) # layernorm 应该加到哪？
        out = self.fc_out(price_news) # N, 1
        out = out.squeeze()
        # 保持一致
        output = {
            'predictions': out,
            'routing_weights': None,
            'hidden_representations': None,
            'top_k_indices': None,
            'routing_weights_flat': None,
        }
        return output


class PriceNewsCrossAttn(nn.Module):
    """
    价格作为 Query，新闻作为 Key/Value 的 Cross-Attention 模块。
    返回与时间长度一致的融合序列 [N, T, d_model]，供后续 GRU(news) 使用。
    """
    def __init__(self, d_price: int, d_news: int, d_model: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.q_proj = nn.Linear(d_price, d_model)
        self.kv_proj = nn.Linear(d_news, d_model)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.resid_dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, price: torch.Tensor, news: torch.Tensor) -> torch.Tensor:
        # 输入: price [N,T,Dp], news [N,T,Dn]
        # 输出: [N,T,d_model]
        q = self.q_proj(price)
        kv = self.kv_proj(news)
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        fused = self.ln(q + self.resid_dropout(attn_out))
        return fused


class MIGAB2(nn.Module):
    """
    B2: 在 MIGAB1 基础上加入 Price->News Cross-Attention 的消融版本。

    输入:
      - price: [N, T, D_price]
      - news:  [N, T, D_news]
      - mask:  [N, T] True 表示 padding（无新闻）
    """
    def __init__(self, price_dim: int,
                 news_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.0,
                 frozen: bool = False,
                 model_path: Optional[str] = None,
                 padding_method: str = 'zero',
                 min_news: int = 10,
                 n_heads: int = 4):
        super().__init__()
        self.min_news = min_news
        self.hidden_dim = hidden_dim
        self.news_count = []
        self.stock_count = []
        self.padding_method = padding_method

        self.gru_price = nn.GRU(
            input_size=price_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.gru_news = nn.GRU(
            input_size=news_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        if self.padding_method == 'learn':
            self.padding = nn.Parameter(torch.zeros(1, news_dim))
            nn.init.xavier_uniform_(self.padding)

        if frozen:
            assert model_path is not None, "model_path must be specified when frozen is True"
            self.gru_price.load_state_dict(torch.load(model_path))

        # Cross-Attention 输出序列维度与 news_dim 一致，便于直接送入 news GRU
        self.cross_attn = PriceNewsCrossAttn(d_price=price_dim, d_news=news_dim, d_model=news_dim, n_heads=n_heads, dropout=dropout)

        # 双分支拼接 -> 2*hidden_dim
        self.ln = nn.LayerNorm(hidden_dim * 2)
        self.fc_out = nn.Linear(hidden_dim * 2, 1)

    def reset_count(self):
        self.news_count = []
        self.stock_count = []

    def update_count(self, news_count, stock_count):
        self.news_count.append(news_count)
        self.stock_count.append(stock_count)

    @property
    def news_coverage_ratio(self):
        return np.mean(self.news_count) / np.mean(self.stock_count)

    def forward(self, price: torch.Tensor, news: torch.Tensor, mask: torch.Tensor):
        N_price, T_price, D_price = price.shape
        N_news, T_news, D_news = news.shape
        assert N_price == N_news, "price and news must have the same batch size"

        # 与 MIGAB1 一致的最少新闻条数逻辑
        if self.min_news != -1:
            news_sum = (~mask).sum(dim=1)  # N
            news_sum_mask = news_sum < self.min_news  # True 表示新闻不足
            news_valid = news[~news_sum_mask]
        else:
            news_sum_mask = torch.zeros(N_price, dtype=torch.bool, device=news.device)
            news_valid = news
        self.update_count((~news_sum_mask).sum().item(), N_price)

        # padding
        if self.padding_method == 'zero':
            news = news.clone()
            news[mask] = 0
            news_valid = news[~news_sum_mask]
        elif self.padding_method == 'learn':
            news = news.clone()
            news[mask] = self.padding
            news_valid = news[~news_sum_mask]
        else:
            raise ValueError(f"Unknown padding method: {self.padding_method}")

        # price 分支（所有样本）
        price_out, _ = self.gru_price(price)
        price_out = price_out[:, -1, :]  # [N, H]

        # news 分支：先进行 cross-attn，再进入新闻 GRU（仅新闻足够的样本）
        news_out = torch.zeros((N_price, self.hidden_dim), device=news.device)
        if (~news_sum_mask).any():
            price_sub = price[~news_sum_mask]
            news_cross_seq = self.cross_attn(price_sub, news_valid)  # [N_sub, T, news_dim]
            news_sub_out, _ = self.gru_news(news_cross_seq)          # [N_sub, T, H]
            news_sub_out = news_sub_out[:, -1, :]
            news_out[~news_sum_mask] = news_sub_out

        # 双分支拼接并输出
        fused = torch.cat([price_out, news_out], dim=1)  # [N, 2H]
        fused = self.ln(fused)
        out = self.fc_out(fused).squeeze(-1)

        return {
            'predictions': out,
            'routing_weights': None,
            'hidden_representations': None,
            'top_k_indices': None,
            'routing_weights_flat': None,
        }


class MIGAB2MoE(nn.Module):
    """
    B2-MoE: 在 MIGAB2 的基础上，将 cross-attn 后的新闻序列送入 MoE 专家集合。
    Router 使用均匀加权（无学习），但保留 router 接口字段，便于后续替换为真实 Router。

    expert_type: 'gru' 或 'mlp'
      - gru 专家：GRU(news_dim -> hidden_dim)，取最后时刻 hidden
      - mlp 专家：先对序列做 mean-pool -> [N, news_dim]，MLP(news_dim -> hidden_dim)
    """
    def __init__(self, price_dim: int,
                 news_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 1,
                 dropout: float = 0.0,
                 frozen: bool = False,
                 model_path: Optional[str] = None,
                 padding_method: str = 'zero',
                 min_news: int = 10,
                 n_heads: int = 2,
                 num_experts: int = 4,
                 expert_type: str = 'gru'):
        super().__init__()
        assert expert_type in {'gru', 'mlp'}
        self.min_news = min_news
        self.hidden_dim = hidden_dim
        self.news_count = []
        self.stock_count = []
        self.padding_method = padding_method
        self.num_experts = num_experts
        self.expert_type = expert_type

        # price 分支
        self.gru_price = nn.GRU(
            input_size=price_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        # cross-attn 输出与 news_dim 对齐，便于作为专家输入
        self.cross_attn = PriceNewsCrossAttn(d_price=price_dim, d_news=news_dim, d_model=news_dim, n_heads=n_heads, dropout=dropout)

        # 专家集合
        experts = []
        if expert_type == 'gru':
            for _ in range(num_experts):
                experts.append(nn.GRU(input_size=news_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout))
            self.experts = nn.ModuleList(experts)
        else:  # mlp
            for _ in range(num_experts):
                experts.append(nn.Sequential(
                    nn.Linear(news_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ))
            self.experts = nn.ModuleList(experts)

        if self.padding_method == 'learn':
            self.padding = nn.Parameter(torch.zeros(1, news_dim))
            nn.init.xavier_uniform_(self.padding)

        if frozen:
            assert model_path is not None, "model_path must be specified when frozen is True"
            self.gru_price.load_state_dict(torch.load(model_path))

        # 输出层: price_out + news_moe_out -> 2*hidden_dim
        self.ln = nn.LayerNorm(hidden_dim * 2)
        self.fc_out = nn.Linear(hidden_dim * 2, 1)

    def reset_count(self):
        self.news_count = []
        self.stock_count = []

    def update_count(self, news_count, stock_count):
        self.news_count.append(news_count)
        self.stock_count.append(stock_count)

    @property
    def news_coverage_ratio(self):
        return np.mean(self.news_count) / np.mean(self.stock_count)

    def forward(self, price: torch.Tensor, news: torch.Tensor, mask: torch.Tensor):
        N_price, T_price, D_price = price.shape
        N_news, T_news, D_news = news.shape
        assert N_price == N_news, "price and news must have the same batch size"

        # 新闻数量限制
        if self.min_news != -1:
            news_sum = (~mask).sum(dim=1)
            news_sum_mask = news_sum < self.min_news
            news_valid = news[~news_sum_mask]
        else:
            news_sum_mask = torch.zeros(N_price, dtype=torch.bool, device=news.device)
            news_valid = news
        self.update_count((~news_sum_mask).sum().item(), N_price)

        # padding
        if self.padding_method == 'zero':
            news = news.clone()
            news[mask] = 0
            news_valid = news[~news_sum_mask]
        elif self.padding_method == 'learn':
            news = news.clone()
            news[mask] = self.padding
            news_valid = news[~news_sum_mask]
        else:
            raise ValueError(f"Unknown padding method: {self.padding_method}")

        # price 分支
        price_out, _ = self.gru_price(price)
        price_out = price_out[:, -1, :]  # [N, H]

        # MoE on cross-attn(news)
        news_moe = torch.zeros((N_price, self.hidden_dim), device=news.device)
        routing_weights = None
        if (~news_sum_mask).any():
            price_sub = price[~news_sum_mask]
            fused_seq = self.cross_attn(price_sub, news_valid)  # [N_sub, T, D_news]
            expert_outs = []
            if self.expert_type == 'gru':
                for gru in self.experts:
                    out, _ = gru(fused_seq)
                    expert_outs.append(out[:, -1, :])  # [N_sub, H]
            else:  # mlp
                pooled = fused_seq.mean(dim=1)  # [N_sub, D_news]
                for mlp in self.experts:
                    expert_outs.append(mlp(pooled))  # [N_sub, H]
            # 均匀加权
            stack = torch.stack(expert_outs, dim=0)  # [E, N_sub, H]
            moe_sub = stack.mean(dim=0)              # [N_sub, H]
            news_moe[~news_sum_mask] = moe_sub
            # router 接口：提供均匀权重
            routing_weights = torch.full((N_price, self.num_experts), 1.0 / self.num_experts, device=news.device)

        # 拼接并输出
        fused = torch.cat([price_out, news_moe], dim=1)
        fused = self.ln(fused)
        out = self.fc_out(fused).squeeze(-1)

        return {
            'predictions': out,
            'routing_weights': routing_weights,
            'hidden_representations': None,
            'top_k_indices': None,
            'routing_weights_flat': routing_weights,
        }

class MIGAB1VarLen(nn.Module):
    """
    Variable-length news version of MIGAB1.

    Inputs:
      - price: Tensor [N, T, D_price]
      - news:  list of Tensor[L_i, D_news] per sample (variable length)
               or a dense Tensor [N, Tn, D_news] with optional mask
      - mask:  Optional[Tensor[N, Tn]] True for padding positions if dense news is provided

    Behavior:
      - If news is a list, it is padded on-the-fly with zeros or a learnable padding vector.
      - Applies minimum news threshold filtering (min_news) before passing through the news GRU.
    """

    def __init__(
        self,
        price_dim: int,
        news_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        frozen: bool = False,
        model_path: Optional[str] = None,
        padding_method: str = "zero",
        min_news: int = 10,
    ):
        super().__init__()
        self.min_news = min_news
        self.hidden_dim = hidden_dim
        self.news_count: List[int] = []
        self.stock_count: List[int] = []
        self.padding_method = padding_method

        self.gru_price = nn.GRU(
            input_size=price_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.gru_news = nn.GRU(
            input_size=news_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        if self.padding_method == "learn":
            self.padding = nn.Parameter(torch.zeros(1, news_dim))
            nn.init.xavier_uniform_(self.padding)

        if frozen:
            assert model_path is not None, "model_path must be specified when frozen is True"
            self.gru_price.load_state_dict(torch.load(model_path, map_location="cpu"))

        self.ln = nn.LayerNorm(hidden_dim * 2)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.add_gate = AddGateFusion(hidden_dim)

    def reset_count(self):
        self.news_count = []
        self.stock_count = []

    def update_count(self, news_count, stock_count):
        self.news_count.append(news_count)
        self.stock_count.append(stock_count)

    @property
    def news_coverage_ratio(self):
        if len(self.stock_count) == 0:
            return 0.0
        return float(np.mean(self.news_count) / max(1e-12, np.mean(self.stock_count)))

    def _pad_varlen_news(
        self, news: List[torch.Tensor], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pads list of [L_i, D_news] into [N, Lmax, D_news] and returns (news_padded, mask_n).
        mask_n: True for padded (invalid) positions.
        """
        if len(news) == 0:
            return torch.empty(0, 0, 0, device=device), torch.empty(0, 0, dtype=torch.bool, device=device)
        dn = news[0].shape[-1] if news[0].numel() > 0 else 0
        news_padded = pad_sequence(news, batch_first=True, padding_value=0.0).to(device)
        Lmax = news_padded.size(1)
        lengths = torch.tensor([t.size(0) for t in news], device=device)
        mask_n = torch.arange(Lmax, device=device).unsqueeze(0).expand(len(news), Lmax) >= lengths.unsqueeze(1)
        if self.padding_method == "learn" and dn > 0:
            pad_vec = self.padding.to(device)
            news_padded[mask_n] = pad_vec
        return news_padded, mask_n

    def _prepare_news_inputs(
        self,
        news: Union[List[torch.Tensor], torch.Tensor],
        mask: Optional[Union[torch.Tensor, List[torch.Tensor]]],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare news inputs for GRU by aligning and padding.

        Inputs:
          - news: list of [L_i, D], concatenated Tensor [K, D], or dense [N, T, D]
          - mask: BoolTensor [N, T], list of [T], or None

        Returns:
          - news_tensor: [N, T, D]
          - mask_n: BoolTensor [N, T] (same as mask_nt)
        """
        # Case 1: Dense news tensor provided
        if torch.is_tensor(news) and news.dim() == 3:
            news_tensor = news
            if mask is None:
                mask_n = torch.zeros(news_tensor.shape[:2], dtype=torch.bool, device=device)
            else:
                assert torch.is_tensor(mask), "mask must be a Tensor for dense news"
                mask_n = mask.to(torch.bool)
            if self.padding_method == "learn":
                news_tensor[mask_n] = self.padding.to(device)
            else:
                news_tensor[mask_n] = 0.0
            return news_tensor, mask_n

        # Case 2: Variable-length news (list or concat) with mask
        if torch.is_tensor(mask):
            # Vectorized alignment with [N, T] mask
            assert mask.dim() == 2, "mask must be [N, T]"
            N, Tm = mask.shape
            mask_n = mask.to(torch.bool).to(device)
            news_concat = torch.cat([t.to(device) for t in news], dim=0) if isinstance(news, list) else news.to(device)
            Dn = news_concat.size(-1) if news_concat.numel() > 0 else self.gru_news.input_size
            K = news_concat.size(0)
            if self.padding_method == "learn":
                base = self.padding.to(device).expand(N, Tm, Dn).clone()
            else:
                base = torch.zeros(N, Tm, Dn, device=device)
            valid_flat = (~mask_n).view(-1)
            idx_flat = valid_flat.nonzero(as_tuple=False).squeeze(-1)
            assert K == idx_flat.numel(), "K should be equal to idx_flat.numel()"
            if K > 0:
                base_flat = base.view(N * Tm, Dn)
                base_flat[idx_flat] = news_concat
            return base, mask_n
        elif isinstance(mask, list):
            # Fallback for list of masks (less efficient)
            aligned_news, mask_list_full = [], []
            for news_i, day_mask_i in zip(news, mask):
                day_mask_i = day_mask_i.to(device)
                T_i = day_mask_i.numel()
                Dn = news_i.size(-1) if news_i.numel() > 0 else self.gru_news.input_size
                base = torch.zeros(T_i, Dn, device=device)
                if self.padding_method == "learn": base += self.padding.to(device)
                valid_pos = (~day_mask_i).nonzero(as_tuple=False).squeeze(-1)
                take = min(news_i.size(0), valid_pos.numel())
                if take > 0: base[valid_pos[:take]] = news_i.to(device)[:take]
                aligned_news.append(base)
                mask_list_full.append(day_mask_i)
            news_tensor = pad_sequence(aligned_news, batch_first=True, padding_value=0.0)
            mask_n = pad_sequence(mask_list_full, batch_first=True, padding_value=True)
            if self.padding_method == "learn": news_tensor[mask_n] = self.padding.to(device)
            return news_tensor, mask_n
        else:
            # Fallback: pad only by news length (no mask)
            return self._pad_varlen_news(news, device)

    def _compute_news_length_and_filter(
        self, mask_n: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-sample news length and insufficient mask based on self.min_news.

        Inputs:
          - mask_n: BoolTensor [N, T], True means no news.

        Returns:
          - news_len: LongTensor [N]
          - news_insufficient: BoolTensor [N]
        """
        N = mask_n.size(0) if mask_n.numel() > 0 else 0
        if mask_n.numel() == 0:
            news_len = torch.zeros(N, device=device, dtype=torch.long)
        else:
            news_len = (~mask_n).sum(dim=1).to(torch.long)
        if self.min_news != -1:
            news_insufficient = news_len < self.min_news
        else:
            news_insufficient = torch.zeros(news_len.size(0), dtype=torch.bool, device=device)
        return news_len, news_insufficient

    def forward(
        self,
        price: torch.Tensor,
        news: Union[List[torch.Tensor], torch.Tensor],
        mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    ):
        N_price, T_price, _ = price.shape

        news_tensor, mask_n = self._prepare_news_inputs(news, mask, price.device)

        # Count and filter by minimum news (moved to helper)
        _, news_insufficient = self._compute_news_length_and_filter(mask_n, price.device)
        self.update_count((~news_insufficient).sum().item(), N_price)

        # Price branch
        price_out, _ = self.gru_price(price)
        price_out = price_out[:, -1, :]

        # News branch
        news_out = torch.zeros((N_price, self.hidden_dim), device=price.device)
        if (~news_insufficient).any() and news_tensor.numel() > 0:
            news_valid = news_tensor[~news_insufficient]
            news_h, _ = self.gru_news(news_valid)
            news_h = news_h[:, -1, :]
            news_out[~news_insufficient] = news_h

        out = self.add_gate(price_out, news_out, (~news_insufficient).unsqueeze(1))
        # price_news = self.ln(price_news)
        out = self.fc_out(out).squeeze(-1)

        return {
            "predictions": out,
            "routing_weights": None,
            "hidden_representations": None,
            "top_k_indices": None,
            "routing_weights_flat": None,
        }




class NewsPriceCrossAttn(nn.Module):
    """
    Cross-Attention: News as Query, Price as Key/Value.
    Output sequence length follows news length, suitable for feeding into news GRU.
    """
    def __init__(self, d_news: int, d_price: int, d_model: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.q_proj = nn.Linear(d_news, d_model)
        self.kv_proj = nn.Linear(d_price, d_model)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.resid_dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, news: torch.Tensor, price: torch.Tensor) -> torch.Tensor:
        # news: [N, Tn, D_news] as Query
        # price: [N, Tp, D_price] as Key/Value
        q = self.q_proj(news)
        kv = self.kv_proj(price)
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        fused = self.ln(q + self.resid_dropout(attn_out))
        return fused

class MIGAB2VarLenCrossAttn(MIGAB1VarLen):
    """
    在 MIGAB1VarLen 基础上加入 Cross-Attention：以新闻为 Query，价格为 Key/Value。
    Cross-Attention 输出序列长度与新闻一致，可直接送入 news GRU。

    其他逻辑（变长新闻处理、最少新闻条数过滤、price/news 双分支拼接）保持与父类一致。
    """
    def __init__(
        self,
        price_dim: int,
        news_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        frozen: bool = False,
        model_path: Optional[str] = None,
        padding_method: str = "zero",
        min_news: int = 10,
        n_heads: int = 4,
        d_model: Optional[int] = None,
    ):
        super().__init__(
            price_dim=price_dim,
            news_dim=news_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            frozen=frozen,
            model_path=model_path,
            padding_method=padding_method,
            min_news=min_news,
        )
        self.d_model = news_dim if d_model is None else d_model
        # Cross-Attn 输出维度对齐到 d_model，如与 news_dim 不同则添加适配层
        self.gru_news = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )        
        self.cross_attn_np = NewsPriceCrossAttn(d_news=news_dim, d_price=price_dim, d_model=self.d_model, n_heads=n_heads, dropout=dropout)

    def forward(
        self,
        price: torch.Tensor,
        news: Union[List[torch.Tensor], torch.Tensor],
        mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    ):
        N_price, T_price, _ = price.shape

        # 准备变长新闻张量与掩码
        news_tensor, mask_n = self._prepare_news_inputs(news, mask, price.device)

        # 统计与最少新闻过滤
        _, news_insufficient = self._compute_news_length_and_filter(mask_n, price.device)
        self.update_count((~news_insufficient).sum().item(), N_price)

        # 价格分支
        price_out, _ = self.gru_price(price)
        price_out = price_out[:, -1, :]

        # 新闻分支：先做 Cross-Attn(News<-Price)，再进新闻 GRU
        news_out = torch.zeros((N_price, self.hidden_dim), device=price.device)
        if (~news_insufficient).any() and news_tensor.numel() > 0:
            news_valid = news_tensor[~news_insufficient]      # [N_sub, Tn, D_news]
            price_sub = price[~news_insufficient]             # [N_sub, Tp, D_price]
            fused_seq = self.cross_attn_np(news_valid, price_sub)  # [N_sub, Tn, d_model]
            news_h, _ = self.gru_news(fused_seq)                   # [N_sub, Tn, H]
            news_h = news_h[:, -1, :]
            news_out[~news_insufficient] = news_h

        # 融合输出（gate 融合，与父类保持一致）：
        # 使用 (~news_insufficient).unsqueeze(1) 作为有效性的 mask，引导门控在无新闻时更依赖 price_out
        out_hidden = self.add_gate(price_out, news_out, (~news_insufficient).unsqueeze(1))
        out = self.fc_out(out_hidden).squeeze(-1)

        return {
            "predictions": out,
            "routing_weights": None,
            "hidden_representations": None,
            "top_k_indices": None,
            "routing_weights_flat": None,
        }

class MIGAB3VarLenMoE(MIGAB2VarLenCrossAttn):
    """
    B3-VarLen-MoE: 在 MIGAB2VarLenCrossAttn 基础上，将新闻分支替换为简单加权的 MoE 专家集合。

    - 继承变长处理与 Cross-Attn(News<-Price) 逻辑；
    - 使用全局可学习 logits 产生固定权重，对专家输出做加权和；
    - 融合方式保持与父类一致（AddGateFusion + fc_out(hidden_dim->1)）。
    """
    def __init__(self,
                 price_dim: int,
                 news_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.0,
                 frozen: bool = False,
                 model_path: Optional[str] = None,
                 padding_method: str = "zero",
                 min_news: int = 10,
                 n_heads: int = 4,
                 d_model: Optional[int] = None,
                 num_experts: int = 4,
                 expert_type: str = 'gru'):
        super().__init__(price_dim=price_dim,
                         news_dim=news_dim,
                         hidden_dim=hidden_dim,
                         num_layers=num_layers,
                         dropout=dropout,
                         frozen=frozen,
                         model_path=model_path,
                         padding_method=padding_method,
                         min_news=min_news,
                         n_heads=n_heads,
                         d_model=d_model)
        assert expert_type in {'gru', 'mlp'}
        self.num_experts = num_experts
        self.expert_type = expert_type

        input_dim = self.d_model  # cross_attn_np 输出维度
        experts = []
        
        if expert_type == 'gru':
            for _ in range(num_experts):
                experts.append(nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, dropout=dropout))
            self.experts = nn.ModuleList(experts)
        elif expert_type == 'mlp':
            self.gru_price = nn.GRU(input_size=price_dim, 
                                   hidden_size=hidden_dim,
                                   num_layers=num_layers, 
                                   batch_first=True,
                                   dropout=dropout)
            for _ in range(num_experts):
                experts.append(nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.Dropout(dropout),
                    nn.LeakyReLU(),
                    nn.Linear(hidden_dim, 1),
                ))
            self.experts = nn.ModuleList(experts)

        self.gate_logits = nn.Parameter(torch.zeros(num_experts))
        self.add_gate = AddGateFusion(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self,
                price: torch.Tensor,
                news: Union[List[torch.Tensor], torch.Tensor],
                mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None):
        N_price, T_price, _ = price.shape

        # 变长准备与过滤（与父类一致）
        news_tensor, mask_n = self._prepare_news_inputs(news, mask, price.device)
        _, news_insufficient = self._compute_news_length_and_filter(mask_n, price.device)
        self.update_count((~news_insufficient).sum().item(), N_price)

        # price 分支
        price_out, _ = self.gru_price(price)
        price_out = price_out[:, -1, :]

        # 新闻分支：Cross-Attn 后进入 MoE
        news_out = torch.zeros((N_price, self.hidden_dim), device=price.device)
        
        if (~news_insufficient).any() and news_tensor.numel() > 0:
            news_valid = news_tensor[~news_insufficient]        # [N_sub, Tn, D_news]
            price_sub = price[~news_insufficient]               # [N_sub, Tp, D_price]
            fused_seq = self.cross_attn_np(news_valid, price_sub)  # [N_sub, Tn, d_model]
            news_out[~news_insufficient] = fused_seq[:, -1, :]
            
            expert_outputs = []  # [N_sub, H]
            if self.expert_type == 'gru':
                for gru in self.experts:  # type: ignore
                    h, _ = gru(fused_seq)
                    expert_outputs.append(h[:, -1, :])
            elif self.expert_type == 'mlp':
                # price GRU
                price_news = self.add_gate(price_out, news_out, (~news_insufficient).unsqueeze(1))
                # 使用 mask 做 mean-pool
                for mlp in self.experts:  # type: ignore
                    expert_outputs.append(mlp(price_news))

            expert_stack = torch.stack(expert_outputs, dim=0)  # [E, N_sub, H]
            expert_out = expert_stack.mean(dim=0)  # [N_sub, H]

        # 融合与输出：保持与父类一致（AddGateFusion + FC）
        out = expert_out.squeeze(-1)

        return {
            'predictions': out,
            'routing_weights': None,
            'hidden_representations': None,
            'top_k_indices': None,
            'routing_weights_flat': None,
        }