from collections import defaultdict
import pandas as pd
import numpy as np
from typing import Dict, List

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
    
    label_data = handler._infer.loc[slice(start, end), "label"]
        
    return label_data

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

def prepare_task_pool(onlyrun_task_id,
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