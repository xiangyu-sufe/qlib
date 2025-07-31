import pandas as pd
import os
import glob
from pathlib import Path
from tqdm import tqdm

# 方案2：更详细的处理
def load_and_stack_factors(path):
    parquet_files = list(Path(path).rglob("*.parquet.gzip"))
    
    all_factors = []
    for file_path in tqdm(parquet_files, "reading factors"):
        print(f"处理文件: {file_path.name}")
        factor_name = file_path.name.replace('.parquet.gzip', '')
        # 读取数据
        df = pd.read_parquet(file_path)
        print(f"  原始形状: {df.shape}")
        
        # stack操作
        stacked = df.stack()
        stacked.name = factor_name  # 给值列命名
        
        # 转为DataFrame并添加文件信息
        factor_df = stacked.reset_index()
        factor_df = factor_df.rename(columns={
            'level_0': 'instrument',
            'level_1': 'datetime',
        })
        all_factors.append(factor_df)
    
    # 合并所有因子 - 改为按datetime和instrument进行outer merge
    if not all_factors:
        return pd.DataFrame()

    # 从第一个因子开始
    final_df = all_factors[0]

    # 逐个merge其他因子
    for factor_df in tqdm(all_factors[1:], "merging factors"):
        final_df = pd.merge(
            final_df,
            factor_df,
            on=['datetime', 'instrument'],
            how='outer'
        )

    return final_df

# 使用函数
path = "/DATA/hxy/huatai/Data/minute_factors/processed"
saved_path = "/DATA/hxy/minute_factors_to_qlib/processed"
saved_file = "merged_factors.csv"
os.makedirs(saved_path, exist_ok=True)
merged_factors = load_and_stack_factors(path)
print(f"最终数据形状: {merged_factors.shape}")
print(merged_factors.head())
merged_factors.to_csv(os.path.join(saved_path, saved_file), index=False)