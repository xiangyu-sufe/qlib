from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


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
                 padding_method: str = 'zero'):
        super().__init__()
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
            self.padding = nn.Parameter(torch.zeros(1, hidden_dim))
            nn.init.xavier_uniform_(self.padding)
        if frozen:
            assert model_path is not None, "model_path must be specified when frozen is True"
            self.gru_price.load_state_dict(torch.load(model_path))
        self.fc_out = nn.Linear(hidden_dim*2, 1)
        
    def forward(self, price: torch.Tensor, news: torch.Tensor, mask: torch.Tensor):
        if self.padding_method == 'zero':
            news[mask.unsqueeze(-1).expand_as(news)] = 0
        elif self.paddding_method == 'learn':
            news[mask.unsqueeze(-1).expand_as(news)] = self.padding
        else:
            raise ValueError(f"Unknown padding method: {self.padding_method}")
        
        price_out, _ = self.gru_price(price) # N , D
        news_out, _ = self.gru_news(news) # N, D
        price_out = price_out[:, -1, :]
        news_out = news_out[:, -1, :]
        out = self.fc_out(torch.cat([price_out, news_out], dim=1)) # N, 1
        
        # 保持一致
        output = {
            'predictions': out,
            'routing_weights': None,
            'hidden_representations': None,
            'top_k_indices': None,
            'routing_weights_flat': None,
        }
        return output
