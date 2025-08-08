from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  

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
                 min_news: int = 10,):
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
