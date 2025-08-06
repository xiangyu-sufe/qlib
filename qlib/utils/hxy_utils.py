from collections import defaultdict
import functools
import pandas as pd
import numpy as np
from typing import Dict, List
import torch
import re
import os
import lmdb
import tqdm
import pickle
import torch
from qlib.data.dataset import TSDatasetH   # 已在 sys.path 中



root_dir = os.path.expanduser("~")

calendar_path =  f"{root_dir}/GRU/alphamat/20250625/data/calendars/day.csv"
calendar = pd.read_csv(calendar_path)

def compute_grad_norm(model):
    """
        计算模型整体的梯度范数
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    return total_norm

def scale_preserve_sign(x):
    max_abs = np.max(np.abs(x))
    return x / max_abs if max_abs != 0 else x

def scale_preserve_sign_torch(x):
    max_abs = torch.max(torch.abs(x))
    return x / max_abs if max_abs != 0 else x

def compute_layerwise_grad_norm(model):
    """
        计算每层模型的参数
    """
    layer_grad_norms = defaultdict(list)

    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = param.grad.detach().data.norm(2).item()
            layer_grad_norms[name].append(norm)
            
    return layer_grad_norms

def coverage_metric(y_pred, data):
    
    y_true = data.get_label()
    coverage = (y_true <= y_pred).astype(float).mean()
    return 'coverage', coverage, True

def get_label(dataset, segment = "test"):
    """
    获取与预测结果对齐的标签 DataFrame
    """
    handler = dataset.handler
    start, end = dataset.segments[segment]                
    if segment in ('train', 'valid'):
        label_data = handler._learn.loc[slice(start, end), "label"]
    elif segment in ('test'):
        label_data = handler._infer.loc[slice(start, end), "label"]
    else:
        raise ValueError("segment must be 'train', 'valid' or 'test'")    
    return label_data




def is_month_end_trade_day(date):
    """
    检查 date 是否为 calendar 中的月末交易日，如果不是则返回最近的月末交易日。
    Args:
        date: str, datetime, pd.Timestamp
    Returns:
        pd.Timestamp: 月末交易日
        bool: 是否本来就是月末交易日
    """
    cal = pd.to_datetime(calendar['date'])
    cal = pd.DatetimeIndex(cal)
    dt = pd.to_datetime(date)
    # 修正：用 cal.to_series() 作为 groupby 的对象
    month_ends = cal.to_series().groupby([cal.year, cal.month]).max().sort_values()
    is_month_end = dt in month_ends.values
    if is_month_end:
        return dt, True
    year, month = dt.year, dt.month
    print(f"{date} 不是月末交易日，返回最近的月末交易日")
    try:
        month_end = month_ends.loc[(year, month)]
    except KeyError:
        if dt < cal[0]:
            month_end = month_ends.iloc[0]
        else:
            month_end = month_ends.iloc[-1]
    return month_end, False


def load_calendar(path,freq,start_time='',end_time=''):
    '''
    加载日历Calendar

    Parameters
    ----------
    path : str
        数据库所在路径.
    freq : str
        数据频率，day/week/month/quarter.
    start_time : str, optional
        起始日期. The default is ''.
    end_time : str, optional
        结束日期. The default is ''.

    Returns
    -------
    data : (T,) list
        日期向量.

    '''
    data = pd.read_csv(path+'calendars/'+freq+'.csv',index_col=0)
    
    # 列（日期）索引
    if start_time!='' or end_time!='':
        if start_time=='':
            idx_start_time = 0 # 默认首日开始
        else:
            idx_start_time = np.where(pd.to_datetime(data.iloc[:,0])>=pd.to_datetime(start_time))[0][0]
        if end_time=='':
            idx_end_time = data.shape[1] # 默认末日结束
        else:
            idx_end_time = np.where(pd.to_datetime(data.iloc[:,0])<=pd.to_datetime(end_time))[0][-1] + 1
        data = data.iloc[idx_start_time:idx_end_time,:]
        
    data = data.values[:,0].tolist()
    return data

def gen_task(task_config: Dict[str, int],
             daily_date: pd.Series, 
             start_time: str, 
             end_time: str,
             daily_date1: pd.Series = None) -> Dict[int, Dict[str, List[str]]]:
    """
    Args:
        task_config: 任务配置字典, 结构为:
            {
                'train': 训练集长度,
                'valid': 验证集长度,
                'test': 测试集长度,
                'step': 步长,
                'test_buffer': 测试集缓冲区长度
            }
        daily_date: 日频交易日期
        start_time: 测试集的开始时间
        end_time: 测试集的结束时间
        daily_date1: 【数据集】中包含的日期（比如GRU舆情研究就是20170101以后才有数据集）
        
    Returns:
        task_pool: 任务列表, 结构为:
            {
                0: {'train': [训练集开始日期, 训练集结束日期],
                    'valid': [验证集开始日期, 验证集结束日期],
                    'test': [测试集开始日期, 测试集结束日期]}
            }
    """    
    # 若step不存在，则step为test长度
    if 'step' not in task_config.keys():
        task_config['step'] = task_config['test']
        
    # 确保start_time = task_pool[0]['test'][0] + test_buffer
    if 'test_buffer' not in task_config.keys():
        task_config['test_buffer'] = 0
    
    # 初始化
    task_pool = dict()
    
    # 首个测试集首日
    idx_test_start_first = np.where(pd.to_datetime(daily_date)>=pd.to_datetime(start_time))[0][0] - task_config['test_buffer']
    # 末个测试集末日
    idx_test_end_last = np.where(pd.to_datetime(daily_date)<=pd.to_datetime(end_time))[0][-1]
    
    # 测试集首末日
    idx_test = np.arange(idx_test_start_first,idx_test_end_last,task_config['step'])
    if idx_test[-1] != idx_test_end_last:
        idx_test = np.append(idx_test,[idx_test_end_last+1])
    
    # 测试集首末日
    idx_test_start = idx_test[:-1]
    #idx_test_end = idx_test[1:]
    idx_test_end = idx_test_start + task_config['test']
    idx_test_end[idx_test_end>=(idx_test_end_last+1)] = idx_test_end_last+1
    
    # 判断是否含验证集
    if 'valid' in task_config.keys():
        
        if isinstance(task_config['valid'], int) and isinstance(task_config['train'], int):
            # 验证集首日
            idx_valid_start = idx_test_start - task_config['valid']
            
            # 训练集首日
            idx_train_start = idx_valid_start - task_config['train']
        
        elif isinstance(task_config['valid'], float) and isinstance(task_config['train'], float):
            assert daily_date1 is not None, 'daily_date1 must be provided when using float train and valid'
            daily_date_train = np.array([daily_date1[daily_date1 < daily_date[i]].shape[0] for i in idx_test_start])
            
            # 验证集首日
            idx_valid_start = idx_test_start - (task_config['valid'] * daily_date_train).astype(int)
            
            # 训练集首日
            idx_train_start = idx_valid_start - (task_config['train'] * daily_date_train).astype(int)
            
        else:
            raise ValueError('train and valid must be either both int or both float')
        
        idx_train_start[idx_train_start<0] = 0
        
        for i in range(len(idx_test_start)):
            task_pool[i] = {'train': [daily_date[idx_train_start[i]], daily_date[idx_valid_start[i]-1]],
                            'valid': [daily_date[idx_valid_start[i]], daily_date[idx_test_start[i]-1]],
                            'test': [daily_date[idx_test_start[i]], daily_date[idx_test_end[i]-1]]}
    else:
        if isinstance(task_config['train'], int):
            # 训练集首日
            idx_train_start = idx_test_start - task_config['train']
            
        elif isinstance(task_config['train'], float):
            assert daily_date1 is not None, 'daily_date1 must be provided when using float train'
            daily_date_train = daily_date1[daily_date1 < start_time].shape[0]
            idx_train_start = idx_test_start - int(task_config['train'] * daily_date_train)
            
        idx_train_start[idx_train_start<0] = 0
        
        for i in range(len(idx_test_start)):
            task_pool[i] = {'train': [daily_date[idx_train_start[i]], daily_date[idx_test_start[i]-1]],
                            'test': [daily_date[idx_test_start[i]], daily_date[idx_test_end[i]-1]]}
      
    return task_pool

def apply_mask_preserve_norm(grad, mask, method = 'minmax',eps=1e-12):
    # 原始范数
    assert method in ['minmax', 'l2', 'l1'], "method must be 'minmax' or 'l2'"
    if method == 'l2':
        norm_orig = grad.norm(p=2)
        
        # masked 后
        masked_grad = grad * mask
        norm_masked = masked_grad.norm(p=2) + eps  # 加 epsilon 防止除零
        
        # 缩放 masked gradient，使范数与原始一样
        scaled_grad = masked_grad * (norm_orig / norm_masked)
    elif method == 'l1':
        norm_orig = grad.norm(p=1)
        
        # masked 后
        masked_grad = grad * mask
        norm_masked = masked_grad.norm(p=1) + eps
    elif method == 'minmax':
        # 计算原始梯度的最大值和最小值
        min_val = grad.min()
        max_val = grad.max()
        
        # masked 后
        masked_grad = grad * mask
        
        # 计算 masked 梯度的最大值和最小值
        masked_min_val = masked_grad.min()
        masked_max_val = masked_grad.max()
        
        # 缩放 masked gradient，使其在原始梯度的范围内
        scaled_grad = (masked_grad - masked_min_val) / (masked_max_val - masked_min_val + eps) * (max_val - min_val) + min_val
        
    return scaled_grad

def prepare_task_pool(onlyrun_task_id: List[int],
                 task_config,
                 path,
                 start_time, end_time):
    """
    只返回每个task的配置信息，不实际加载数据
    """
    daily_date = load_calendar(path, 'day')
    task_pool = gen_task(task_config, daily_date, start_time, end_time, None)
    only_run_task_pool = {}
    for task_id, (task, segments) in enumerate(task_pool.items()):
        if onlyrun_task_id is not None and task_id not in onlyrun_task_id:
            continue
        only_run_task_pool[task_id] = {
            'segments': segments,
            'task_id': task_id,
        }
        
    return only_run_task_pool


def process_single_factor(factor_data, instrument_df):
    """
    将embedding_vector转换为embedding_0, embedding_1, embedding_2, ...
    """
    factor_data = factor_data[['date', 'stock_code', 'embedding_vector']]
    factor_data['date'] = pd.to_datetime(factor_data['date'])
    factor_data = align_stock_codes(factor_data, instrument_df)
    factor_data['stock_code'] = factor_data['stock_code'].map(convert_stock_code_to_qlib_format)
    stacked = factor_data.set_index(['stock_code', 'date'])
    stacked.index.names = ['instrument', 'datetime']
    stacked = stacked.swaplevel('datetime', 'instrument')
    stacked = stacked.sort_index()
    # 扩张 embedding
    stacked['embed'] = stacked['embedding_vector'].apply(lambda x: ast.literal_eval(x))
    embed_df = pd.DataFrame(stacked["embed"].tolist(), index=stacked.index)
    embed_df = embed_df.add_prefix("embedding_")

    return embed_df

def convert_stock_code_to_qlib_format(stock_code):
    """将完整股票代码转换为qlib格式（如：000001.SZ -> SZ000001）"""
    # 解析股票代码，提取数字部分和后缀
    match = re.match(r'(\d+)\.([A-Z]+)', stock_code)
    if match:
        num_part = match.group(1)
        suffix = match.group(2)
        
        # 根据后缀判断交易所
        if suffix == 'SH':
            return f"SH{num_part}"
        elif suffix == 'SZ':
            return f"SZ{num_part}"
        elif suffix == 'BJ':
            return f"BJ{num_part}"
        else:
            raise ValueError(f"Invalid stock code: {stock_code}")

def align_stock_codes(df, instrument_df):
    """
    根据instrument_path的信息对齐股票代码
    df: stock_code
    instrument_df: stock_code
    """
    # print("正在对齐股票代码...")
     # 将embedding数据中的stock_code补全到6位
    df['stock_code_6digit'] = df['stock_code'].astype(int).astype(str).str.zfill(6)
   
    # 从instrument_df中提取数字部分和后缀
    instrument_df['stock_code_num'] = instrument_df['stock_code'].str.extract(r'(\d+)')[0]
    instrument_df['stock_code_suffix'] = instrument_df['stock_code'].str.extract(r'\.([A-Z]+)')[0]
      
    # 创建映射字典：数字代码 -> 完整代码
    code_mapping = dict(zip(instrument_df['stock_code_num'], instrument_df['stock_code']))
    
    # 检查有多少股票代码在instrument_df中存在
    valid_stocks = df[df['stock_code_6digit'].isin(code_mapping.keys())]
    invalid_stocks = df[~df['stock_code_6digit'].isin(code_mapping.keys())]
    
    # print(f"有效股票数量: {len(valid_stocks)}")
    print(f"无效股票数量: {len(invalid_stocks)}")
    print(invalid_stocks['stock_code_6digit'].unique()[:10])
    # if len(invalid_stocks) > 0:
    #     print("无效股票代码示例:")
    #     
    
    # 只保留有效的股票，并映射到完整的股票代码
    df_aligned = valid_stocks.copy()
    df_aligned['stock_code'] = df_aligned['stock_code_6digit'].map(code_mapping)

    # print(f"对齐后的数据形状: {df_aligned.shape}")
    return df_aligned

def read_label(day=10, method='win+neu+zscore'):
    root_dir = os.path.expanduser('~')
    ser = pd.read_parquet(f'{root_dir}/GRU/Data/labels/csiall_{method}/ret_{day}d.parquet.gzip')
    ser = ser.stack(level=-1)
    ser = ser.swaplevel(0, 1)
    ser.index.names = ['datetime', 'instrument']
    df = ser.to_frame('label')
    # 股票代码转换
    df.index = df.index.set_levels(df.index.levels[1].map(convert_stock_code_to_qlib_format), level=1)
    # 日期处理
    df.index = df.index.set_levels(pd.to_datetime(df.index.levels[0].astype(str)), level=0)
    df.sort_index(inplace=True)
    
    return df

def read_alpha64():
    # 读取自有alpha64 数据
    root_dir = os.path.expanduser('~')
    df = pd.read_pickle(f'{root_dir}/GRU/Data/alpha64/qlib/alpha64.pkl')
    df.columns = df.columns.str.replace(r'\..*$', '', regex=True)
    # 处理股票代码
    df.index = df.index.set_levels(df.index.levels[1].map(convert_stock_code_to_qlib_format), level=1)
    df.index = df.index.set_levels(pd.to_datetime(df.index.levels[0].astype(str)), level=0)
    df.sort_index(inplace=True)
    
    return df

def read_ohlc():
    root_dir = os.path.expanduser('~')
    df = pd.read_pickle(f'{root_dir}/GRU/Data/pv1day/qlib/ohlc.pkl')
    df.columns = df.columns.str.replace(r'\..*$', '', regex=True)
    # 处理股票代码
    df.index = df.index.set_levels(df.index.levels[1].map(convert_stock_code_to_qlib_format), level=1)
    df.index = df.index.set_levels(pd.to_datetime(df.index.levels[0].astype(str)), level=0)
    df.sort_index(inplace=True)
    
    return df
    
def read_minute():
    # 读取自有alpha64 数据
    root_dir = os.path.expanduser('~')
    df = pd.read_pickle(f'{root_dir}/GRU/Data/alpha64/qlib/alpha64.pkl')
    df.columns = df.columns.str.replace(r'\..*$', '', regex=True)
    # 处理股票代码
    df.index = df.index.set_levels(df.index.levels[1].map(convert_stock_code_to_qlib_format), level=1)
    df.index = df.index.set_levels(pd.to_datetime(df.index.levels[0].astype(str)), level=0)    
    
# ==================== 处理新闻新增

# ========== 写入新闻数据 向量索引存储 LMDB

def write_lmdb(df: pd.DataFrame, path="news_feat.lmdb"):
    env = lmdb.open(path, map_size=1024**4)      # 1 TB 上限
    with env.begin(write=True) as txn:
        for (inst, dt), row in tqdm.tqdm(df.iterrows()):
            key = f"{inst}_{dt}".encode()        # e.g. 'SH600000_2017-01-03'
            txn.put(key, pickle.dumps(row.values.astype("float32")))
    env.sync(), env.close()

# =========== 读取新闻数据 带缓存

class NewsStore:
    def __init__(self, path="news_feat.lmdb"):
        self.env = lmdb.open(path, readonly=True, lock=False)
        self.dim = 1024

    # LRU 缓存最近 20*2000 ≈ 4 万条，可调
    @functools.lru_cache(maxsize=60000)
    def _fetch(self, inst, dt):
        key = f"{dt}_{inst}".encode()
        with self.env.begin() as txn:
            val = txn.get(key)
        if val is None:
            return None
        return torch.tensor(pickle.loads(val))      # (1024,)

    def get(self, inst, dt):
        vec = self._fetch(inst, dt)
        if vec is None:
            vec = torch.zeros(self.dim)            # 缺失时用 0 向量
        return vec
    
    @property
    def memory_usage(self):
        return self.env.info().get("map_size") // 1e6

# =============  定义 collate_fn

def make_collate_fn(news_store, step_len, dim_news=1024):
    """
    factory 写法，使 news_store 只实例化一次
    """
    def collate_fn(batch):
        insts, dts_seqs, price = zip(*batch)
        price = torch.stack(price)                     # (B, N, T, Dq)

        news_list, mask_list = [], []
        for ins, dts in zip(insts, dts_seqs):
            news, m = _fetch_news_seq(news_store, ins, dts, dim_news)
            news_list.append(news)
            mask_list.append(m)
        news   = torch.stack(news_list)                # (N, T, Dn)
        n_mask = torch.stack(mask_list)                # ( N, T)  BoolTensor
        # 过滤掉一条新闻都没有的股票
        # N, T, D = price.shape
        # D_n = news.shape[-1]
        # valid_flat = ~(n_mask.all(dim=-1))  # shape: [N]，True 表示这条样本“有有效新闻”
        # price_valid = price[valid_flat].view(1, -1, T, D)  # (N', T, D)
        # news_valid = news[valid_flat].view(1, -1, T, D_n)    # (N', T, Dn)
        # mask_valid = n_mask[valid_flat].view(1, -1, T)    # (N', T)  
        
        # 不过滤
        price_valid = price
        news_valid = news 
        mask_valid = n_mask 
        feat = torch.cat([price_valid, news_valid], dim=-1)        # (N, T, Dq+Dn)
        return feat, mask_valid   # <-- 多返回一个 mask
    
    return collate_fn

def _fetch_news_seq(store, inst, dts_seq, dim_news):
    # inst     单个 instrument
    # dts_seq  单个 dts_seq
    vecs, mask = [], []
    if isinstance(inst, list):
        for ins, dts in zip(inst, dts_seq):
            for dt in dts:
                # 对时间循环
                if dt is None: # 时间为None
                    vecs.append(torch.zeros(dim_news))
                    mask.append(True)           
                else:
                    val = store._fetch(ins, dt)
                    if val is None: # 新闻数据缺失
                        vecs.append(torch.zeros(dim_news))
                        mask.append(True)
                    else:
                        vecs.append(val)
                        mask.append(False)
    elif isinstance(inst, str):
        for dt in dts_seq:
            if dt is None:
                vecs.append(torch.zeros(dim_news))
                mask.append(True)           
            else:
                val = store._fetch(inst, dt)
                if val is None: # 新闻数据缺失
                    vecs.append(torch.zeros(dim_news))
                    mask.append(True)
                else:
                    vecs.append(val)
                    mask.append(False)                   
    vecs = torch.stack(vecs)
    mask = torch.tensor(mask)
    
    return vecs, mask   # (N, T, Dn) , (N, T)

# Dataset

# dataset_wrapper.py


class IndexedSeqDataset(torch.utils.data.Dataset):
    """
    Wrap a TSDatasetH/TSDataSampler to also return (instrument, datetime_seq).
    """
    def __init__(self, tsds: TSDatasetH):
        # If the passed object is a TSDatasetH instance, use its underlying
        # TSDataSampler (ts_sampler) which implements `__getitem__`; otherwise
        # assume the object itself is already a sampler.
        self.ds = tsds.ts_sampler if hasattr(tsds, "ts_sampler") else tsds
        self.step_len = tsds.step_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # 当 `idx` 是单个 int -> 返回单只股票；
        # 当 `idx` 是 list/ndarray -> 返回多个股票的并行样本

        price_feat = self.ds[idx]  # TSDataSampler 支持 list/ndarray

        if isinstance(idx, (list, np.ndarray)):
            insts, dts_seqs = [], []
            for single_idx in idx:
                i, j = self.ds._get_row_col(single_idx)
                rows = list(range(max(i - self.step_len + 1, 0), i + 1))
                rows = [None] * (self.step_len - len(rows)) + rows
                dts = [None if r is None else self.ds.idx_df.index[r] for r in rows]
                inst = self.ds.idx_df.columns[j]
                insts.append(inst)
                dts_seqs.append(dts)
            return insts, dts_seqs, torch.tensor(price_feat)
        else:
            i, j = self.ds._get_row_col(idx)
            rows = list(range(max(i - self.step_len + 1, 0), i + 1))
            rows = [None] * (self.step_len - len(rows)) + rows
            dts = [None if r is None else self.ds.idx_df.index[r] for r in rows]
            inst = self.ds.idx_df.columns[j]
            return inst, dts, torch.tensor(price_feat)
        
        
    
# =============== 

def process_ohlc(ohlc: torch.Tensor):
    # ohlc N, T, 6
    # 最后一列为 volume
            # data_ar[:, :5, :] = data_ar[:, :5, :] / data_ar[:, :1, -1:]

        # data_ar[:, 5, :] = data_ar[:, 5, :] / data_ar[:, 5, -1:]


        # self.data_arr[:, :self.step_len * self.feat_num] = data_ar.reshape((-1, self.feat_num * self.step_len))
    ohlc[:, :, :5] = ohlc[:, :, :5] / ohlc[:, -1:, :1]
    ohlc[:, :, 5] = ohlc[:, :, 5] / ohlc[:, -1:, 5] 
    
    return ohlc


