import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from itertools import combinations
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_regression
import lightgbm as lgb

processed_factors_dir = "/home/huxiangyu/GRU/minute_factors_parquet/processed_continue"
close_adj_path = "/home/huxiangyu/GRU/alphamat/20250625/data/features/close_adj_day.parquet.gzip"

def distance_correlation():
    
    ...

def get_label(close_adj, n=1):
    # label = (t+N - t+N-1) / t+N-1
    return (close_adj.shift(-n-1) - close_adj.shift(-1)) / close_adj.shift(-1)


def load_factors_and_label(factors_dir, close_adj_path, n=1):
    files = [f for f in os.listdir(factors_dir) if f.endswith('.parquet') or f.endswith('.parquet.gzip')]
    features = [f.replace('.parquet.gzip', '').replace('.parquet', '') for f in files]
    factor_dfs = {}
    for f, name in zip(files, features):
        df = pd.read_parquet(os.path.join(factors_dir, f))
        factor_dfs[name] = df
    close_adj = pd.read_parquet(close_adj_path)
    close_adj = close_adj.T
    close_adj.index = pd.to_datetime(close_adj.index)
    label = get_label(close_adj, n=n)
    return factor_dfs, features, label


def _pair_corr_worker(f1, f2, factor_dfs, method, min_valid=500):
    df1 = factor_dfs[f1]
    df2 = factor_dfs[f2]
    common_index = df1.index.intersection(df2.index)
    common_cols = df1.columns.intersection(df2.columns)
    if len(common_index) == 0 or len(common_cols) == 0:
        return (f1, f2, np.nan)
    df1_aligned = df1.loc[common_index, common_cols]
    df2_aligned = df2.loc[common_index, common_cols]
    corrs = []
    for date in tqdm(common_index, desc=f'{method}-corr-{f1}-{f2}', leave=False):
        x = df1_aligned.loc[date].values
        y = df2_aligned.loc[date].values
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < min_valid:
            continue
        x_clean = x[mask]
        y_clean = y[mask]
        corr = pd.Series(x_clean).corr(pd.Series(y_clean), method=method)
        if not np.isnan(corr):
            corrs.append(corr)
    avg_corr = np.nanmean(corrs) if corrs else np.nan
    return (f1, f2, avg_corr)


def calc_cross_section_corr_parallel(factor_dfs, features, method='pearson', n_jobs=-1, min_valid=500):
    corr_matrix = pd.DataFrame(index=features, columns=features, dtype=float)
    pairs = list(combinations(features, 2))
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_pair_corr_worker)(f1, f2, factor_dfs, method, min_valid) for f1, f2 in tqdm(pairs, desc=f'特征两两相关性({method})')
    )
    for f1, f2, avg_corr in results:
        corr_matrix.loc[f1, f2] = avg_corr
        corr_matrix.loc[f2, f1] = avg_corr
    np.fill_diagonal(corr_matrix.values, 1.0)
    corr_matrix.to_csv(f'feature_feature_{method}_corr.csv')
    return corr_matrix


def calc_feature_label_corr(factor_dfs, features, label, n_jobs=-1):
    def corr_worker(f):
        df = factor_dfs[f]
        corrs = []
        for date in tqdm(df.index.intersection(label.index), desc=f'Corr-{f}', leave=False):
            common_cols = df.columns.intersection(label.columns)
            if len(common_cols) == 0:
                continue
            x = df.loc[date, common_cols].values
            y = label.loc[date, common_cols].values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 500:
                continue
            x_clean = x[mask]
            y_clean = y[mask]
            corr = pd.Series(x_clean).corr(pd.Series(y_clean), method='spearman')
            if not np.isnan(corr):
                corrs.append(corr)
        return f, np.nanmean(corrs) if corrs else np.nan
    results = Parallel(n_jobs=n_jobs, verbose=10)(delayed(corr_worker)(f) for f in tqdm(features, desc='特征与label相关性'))
    corr_series = pd.Series(dict(results)).sort_values(ascending=False)
    corr_series.to_csv('feature_label_spearman_corr.csv')
    return corr_series


def calc_cross_section_mi(factor_dfs, features, label, n_jobs=-1):
    def mi_worker(f):
        df = factor_dfs[f]
        mi_list = []
        for date in tqdm(df.index.intersection(label.index), desc=f'MI-{f}', leave=False):
            common_cols = df.columns.intersection(label.columns)
            if len(common_cols) == 0:
                continue
            x = df.loc[date, common_cols].values
            y = label.loc[date, common_cols].values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 500:
                continue
            x_clean = x[mask].reshape(-1, 1)
            y_clean = y[mask]
            try:
                mi = mutual_info_regression(x_clean, y_clean, discrete_features=False)
                mi_list.append(mi[0])
            except Exception:
                continue
        return f, np.nanmean(mi_list) if mi_list else np.nan
    results = Parallel(n_jobs=n_jobs, verbose=10)(delayed(mi_worker)(f) for f in tqdm(features, desc='互信息MI'))
    mi_series = pd.Series(dict(results)).sort_values(ascending=False)
    mi_series.to_csv('feature_mi.csv')
    return mi_series


def calc_lgbm_importance(factor_dfs, features, label, n_jobs=-1):
    def lgbm_worker(date):
        try:
            # 首先确定所有特征在该日期的共同股票
            common_cols = None
            for f in features:
                df = factor_dfs[f]
                if date in df.index:
                    if common_cols is None:
                        common_cols = df.columns.intersection(label.columns)
                    else:
                        common_cols = common_cols.intersection(df.columns)
            
            if common_cols is None or len(common_cols) == 0:
                print(f"日期 {date}: 没有共同股票")
                return None
            
            # 构建特征矩阵
            X = []
            for f in features:
                df = factor_dfs[f]
                if date in df.index:
                    x = df.loc[date, common_cols].values
                else:
                    x = np.full(len(common_cols), np.nan)
                X.append(x)
            
            X = np.array(X).T  # shape: n_stock, n_feature
            y = label.loc[date, common_cols].values
            
            # 检查数据质量
            mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            if mask.sum() < 500:
                print(f"日期 {date}: 有效样本数 {mask.sum()} < 500")
                return None
            
            X_clean = X[mask]
            y_clean = y[mask]
            
            # 检查数据是否包含无穷值
            if np.isinf(X_clean).any() or np.isinf(y_clean).any():
                print(f"日期 {date}: 数据包含无穷值")
                return None
            
            # 检查数据是否全为0或常数
            if np.std(y_clean) == 0:
                print(f"日期 {date}: label标准差为0")
                return None
            
            # 训练LightGBM模型
            dtrain = lgb.Dataset(X_clean, y_clean, free_raw_data=False)
            params = {
                'objective': 'regression', 
                'verbosity': -1,
                'random_state': 42,
                'n_estimators': 30,
                'learning_rate': 0.1
            }
            model = lgb.train(params, dtrain, num_boost_round=30, verbose_eval=False)
            imp = model.feature_importance(importance_type='gain')
            
            if len(imp) != len(features):
                print(f"日期 {date}: 特征重要性数量不匹配 {len(imp)} != {len(features)}")
                return None
                
            return imp
            
        except Exception as e:
            print(f"日期 {date}: LightGBM训练失败 - {str(e)}")
            return None
    
    print("开始计算LightGBM重要性...")
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(lgbm_worker)(date) for date in tqdm(label.index, desc='LightGBM重要性')
    )
    
    # 过滤掉None
    valid_results = [r for r in results if r is not None]
    print(f"LightGBM计算完成: 总日期数 {len(results)}, 有效日期数 {len(valid_results)}")
    
    if not valid_results:
        print("警告: 没有有效的LightGBM结果")
        imp_mean = {f: np.nan for f in features}
    else:
        imp_arr = np.stack(valid_results, axis=0)  # shape: (n_valid_dates, n_features)
        imp_mean = {f: np.nanmean(imp_arr[:, i]) for i, f in enumerate(features)}
    
    imp_series = pd.Series(imp_mean).sort_values(ascending=False)
    imp_series.to_csv('feature_lgbm_importance.csv')
    return imp_series


def warn_high_correlation(corr_matrix, threshold=0.7, method_name=""):
    high_corr_pairs = []
    for i in corr_matrix.index:
        for j in corr_matrix.columns:
            if i < j and not pd.isna(corr_matrix.loc[i, j]) and abs(corr_matrix.loc[i, j]) > threshold:
                high_corr_pairs.append((i, j, corr_matrix.loc[i, j]))
    if not high_corr_pairs:
        print(f"[{method_name}] 没有发现相关性超过 {threshold} 的特征对")
        return
    print(f"\n⚠️  [{method_name}] 发现 {len(high_corr_pairs)} 对高相关性特征 (|相关系数| > {threshold}):")
    print("-" * 80)
    high_corr_pairs_sorted = sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)
    for i, (f1, f2, corr) in enumerate(high_corr_pairs_sorted, 1):
        print(f"{i:2d}. {f1:30s} <-> {f2:30s} | 相关系数: {corr:6.3f}")
    print("-" * 80)
    pd.DataFrame(high_corr_pairs_sorted, columns=["feature1", "feature2", "correlation"]).to_csv(f"high_corr_pairs_{method_name}.csv", index=False)


def correlation_redundancy_removal(corr_matrix, importance_series, threshold=0.7, method_name=""):
    features = set(corr_matrix.index)
    corr_pairs = []
    for i in corr_matrix.index:
        for j in corr_matrix.columns:
            if i != j and not pd.isna(corr_matrix.loc[i, j]):
                corr_pairs.append((i, j, abs(corr_matrix.loc[i, j])))
    corr_pairs = sorted(corr_pairs, key=lambda x: x[2], reverse=True)
    removed = set()
    for f1, f2, corr in corr_pairs:
        if corr <= threshold:
            break
        if f1 in features and f2 in features:
            # 保留重要性高的
            if importance_series[f1] >= importance_series[f2]:
                removed.add(f2)
                features.remove(f2)
            else:
                removed.add(f1)
                features.remove(f1)
    pd.Series(list(removed)).to_csv(f'removed_features_{method_name}.csv', index=False)
    pd.Series(list(features)).to_csv(f'selected_features_{method_name}.csv', index=False)
    return list(features), list(removed)


def main():
    n_day = 1  # 可修改为你需要的N天
    print(f"加载特征和label, N={n_day}...")
    factor_dfs, features, label = load_factors_and_label(processed_factors_dir, close_adj_path, n=n_day)
    print("计算特征两两相关性(Pearson)...")
    feature_corr_pearson = calc_cross_section_corr_parallel(factor_dfs, features, method='pearson', n_jobs=-1, min_valid=500)
    # print("计算特征两两相关性(Spearman)...")
    # feature_corr_spearman = calc_cross_section_corr_parallel(factor_dfs, features, method='spearman', n_jobs=-1, min_valid=500)
    print("计算特征与label相关性...")
    feature_label_corr = calc_feature_label_corr(factor_dfs, features, label, n_jobs=-1)
    print("计算互信息...")
    mi_series = calc_cross_section_mi(factor_dfs, features, label, n_jobs=-1)
    # print("计算LightGBM重要性...")
    # lgbm_series = calc_lgbm_importance(factor_dfs, features, label, n_jobs=-1)

    # 警告高相关性特征对
    warn_high_correlation(feature_corr_pearson, threshold=0.7, method_name="pearson")
    # warn_high_correlation(feature_corr_spearman, threshold=0.7, method_name="spearman")

    print("相关性去冗余（优先保留与label相关性高的特征）...")
    selected_features_spearman, removed_features_spearman = correlation_redundancy_removal(feature_corr_pearson, feature_label_corr, threshold=0.7, method_name="spearman")
    selected_features_mi, removed_features_mi = correlation_redundancy_removal(feature_corr_pearson, mi_series, threshold=0.7, method_name="mi")

    print(f"[Spearman] 最终保留特征数: {len(selected_features_spearman)}，被移除特征数: {len(removed_features_spearman)}")
    print(f"[MI]       最终保留特征数: {len(selected_features_mi)}，被移除特征数: {len(removed_features_mi)}")
    print("流程结束。")

if __name__ == "__main__":
    main()
