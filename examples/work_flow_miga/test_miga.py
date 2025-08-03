import torch
from qlib.contrib.model.pytorch_miga_ts import PriceNewsRouter

if __name__ == "__main__":
    N, T, D_price, D_news, H = 5000, 20, 158, 1024, 128
    price = torch.randn(N, T, D_price)
    news = torch.randn(N, T, D_news)

    # 实例化模块并前向
    model = PriceNewsRouter(price_dim=D_price, news_dim=D_news, d_model=H, n_heads=4, d_gru=64)
    output = model(price, news)

    print(output.shape)   