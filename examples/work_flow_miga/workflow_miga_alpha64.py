#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
Qlib provides two kinds of interfaces.
(1) Users could define the Quant research workflow by a simple configuration.
(2) Qlib is designed in a modularized way and supports creating research workflow by code just like building blocks.

The interface of (1) is `qrun XXX.yaml`.  The interface of (2) is script like this, which nearly does the same thing as `qrun XXX.yaml`
"""
import qlib
from qlib.constant import REG_CN
from qlib.contrib.report import analysis_model, analysis_position
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.utils.hxy_utils import (
    get_label, prepare_task_pool, read_alpha64, read_minute, read_label,
    NewsStore, make_collate_fn, write_lmdb, IndexedSeqDataset
)
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData
from qlib.tests.config import CSI300_BENCH, CSI300_GBDT_TASK
from qlib.data.dataset.handler import DataHandlerLP
from torch.utils.data import Sampler, DataLoader
from qlib.data.dataset.loader import StaticDataLoader, QlibDataLoader  

import pandas as pd
import numpy as np
import os


if __name__ == "__main__":
    # 数据参数
    import argparse
    import time
    parser = argparse.ArgumentParser()
    # 实验参数
    parser.add_argument("--onlyrun_task_id", type=int, default=None, help="Only run task id")
    parser.add_argument("--onlyrun_seed_id", type=int, default=0, help="Only run specified seed id")
    parser.add_argument("--pv1pv5", type=int, default=1, help="PV1 or PV5 day setting")
    parser.add_argument("--step_len", type=int, default=20, help="Step length")
    # 数据集长度参数
    parser.add_argument("--train_length", type=int, default=720, help="Training dataset length")
    parser.add_argument("--valid_length", type=int, default=240, help="Validation dataset length")
    parser.add_argument("--test_length", type=int, default=120, help="Test dataset length")

    # 时间范围参数
    parser.add_argument("--start_time", type=str, default="2021-12-31", help="Start time for data")
    parser.add_argument("--end_time", type=str, default="2024-12-31", help="End time for data")
    # 模型一般参数
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--save_path", type=str, default=".")
    parser.add_argument("--lambda_reg", type=float, default=1)
    # MIGA模型结构参数
    parser.add_argument("--d_feat", type=int, default=158, help="Feature dimension")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension for router (auto-calculated as num_groups * num_experts_per_group if not specified)")
    parser.add_argument("--num_groups", type=int, default=4, help="Number of expert groups")
    parser.add_argument("--num_experts_per_group", type=int, default=4, help="Number of experts per group")
    parser.add_argument("--num_heads", type=int, default=2, help="Number of attention heads")
    parser.add_argument("--top_k", type=int, default=4, help="Top-k experts selection")
    parser.add_argument("--expert_output_dim", type=int, default=1, help="Expert output dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    # MIGA损失函数参数
    parser.add_argument("--omega", type=float, default=2e-3, help="Router loss weight")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Expert loss weight")
    parser.add_argument("--omega_scheduler", type=str, default="exp", choices=["none", "exp", "step"], help="omega动态调整策略: none/exp/step")
    parser.add_argument("--omega_decay", type=float, default=0.96, help="指数衰减率（exp模式）")
    args = parser.parse_args()
    save_path = args.save_path
    save_path = os.path.join(f'seed{args.onlyrun_seed_id}', save_path)
    os.makedirs(save_path, exist_ok=True)

    root_dir = os.path.expanduser('~')
    alphamat_path = f'{root_dir}/GRU/alphamat/20250625/data/'
    # provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    # qlib.init(provider_uri=provider_uri, region=REG_CN)
    market = None # 默认用全部
    benchmark = "SH000905"
    # 读取新闻数据
    a = time.time()
    news_lmdb_path = '/home/huxiangyu/.qlib/llm_data/embedding.lmdb'
    if not os.path.exists(news_lmdb_path):
        news_embed = pd.read_pickle('/home/huxiangyu/.qlib/llm_data/embedding.pkl')
        write_lmdb(news_embed, news_lmdb_path)
    else:
        news_store = NewsStore(news_lmdb_path)
    print(f'新闻数据占用内存大小: {news_store.memory_usage} MB')
    # 读取量价数据
    alpha_64 = read_alpha64()
    labels = read_label(day=10, method = 'win+neu+zscore')
    # 选出大于2018年的数据
    alpha_64 = alpha_64.loc[alpha_64.index.get_level_values(0) >= pd.Timestamp("2018-01-01")]
    labels = labels.loc[labels.index.get_level_values(0) >= pd.Timestamp("2018-01-01")]
    data = alpha_64.join(labels, how='left')
    data.columns = pd.MultiIndex.from_tuples([('feature', col) for col in alpha_64.columns] + [('label', col) for col in labels.columns])
    print("读取所有数据用时: ", time.time() - a)
    print(f"量价数据占用内存大小: {data.memory_usage().sum() / 1e6} MB")
    # 创建 DataLoader
    sdl_pkl = StaticDataLoader(config=data)
    
    task_config = {
        'train': args.train_length,
        'valid': args.valid_length,
        'test': args.test_length,
    }

    only_run_task_pool = prepare_task_pool(args.onlyrun_task_id,
                                        task_config,
                                        path = alphamat_path,
                                        start_time = args.start_time,
                                        end_time = args.end_time)
    

    # 根据only_run_task_pool进行迭代
    for task_id, segments in only_run_task_pool.items():
        print(f"开始处理任务 {task_id}")
        segments = segments['segments']
        # 使用segments中的日期
        start_time = segments['train'][0]  # 训练开始日期
        fit_end_time = segments['train'][1]  # 训练结束日期
        val_start_time = segments['valid'][0]  # 验证开始日期
        val_end_time = segments['valid'][1]  # 验证结束日期
        test_start_time = segments['test'][0]  # 测试开始日期
        test_end_time = segments['test'][1]  # 测试结束日期
        
        # 红色打印日期信息
        print(f"\033[31m训练日期: {start_time} - {fit_end_time}\033[0m")
        print(f"\033[31m验证日期: {val_start_time} - {val_end_time}\033[0m")
        print(f"\033[31m测试日期: {test_start_time} - {test_end_time}\033[0m")
        
        print(f"时间范围: 训练({start_time} - {fit_end_time}), 验证({val_start_time} - {val_end_time}), 测试({test_start_time} - {test_end_time})")
        infer_processors = [
            {"class": "ProcessInfHXY", "kwargs": {}}, # 替换为 nan
            # {"class": "CSRankNorm", "kwargs": {"fields_group": "feature", 'parallel':True, 'n_jobs': 60}},
            {"class": "RobustZScoreNorm", 
             "kwargs": {
                        "fields_group": "feature",
                        "fit_start_time": start_time,
                        "fit_end_time": fit_end_time
                        }},
            {"class": "Fillna", 'kwargs': {'fields_group': 'feature'}},
        ]
        # ic label 不用处理
        learn_processors = [
            {"class": "DropnaLabel"},
        ]
        data_handler_config = {
            "start_time": start_time,
            "end_time": test_end_time,
            # "fit_start_time": start_time,
            # "fit_end_time": fit_end_time,
            "instruments": market,
            "data_loader": sdl_pkl,
            "infer_processors":infer_processors,
            "learn_processors":learn_processors,
            "process_type": "append",
            "drop_raw": True,
        }   
            
        task = {
            "model": {
                "class": "MIGA",
                "module_path": "qlib.contrib.model.pytorch_miga_ts",
                "kwargs": {
                    "d_feat": 64, # 模型参数
                    "hidden_size": 16,
                    "num_groups": 2,
                    "num_experts_per_group": 4,
                    "num_heads": 4,
                    "top_k": 4,
                    "expert_output_dim": 1,
                    "num_layers": 2,
                    "dropout": 0.0,
                    "n_epochs": 50, # 训练参数
                    "batch_size": 5000,
                    "lr": args.lr,
                    "early_stop": 10,
                    "metric": "ic",
                    "loss": "miga",
                    "n_jobs": 24,
                    "GPU": 0,
                    "lambda_reg": args.lambda_reg, # 损失参数
                    "omega": args.omega,
                    "epsilon": args.epsilon,
                    "omega_scheduler": args.omega_scheduler,
                    "omega_decay": args.omega_decay,
                    "debug": True,  # Set to True for debugging mode
                    "save_path": save_path,
                    "step_len": args.step_len,
                    "news_store": news_store,
                },
            },
            "dataset": {
                "class": "TSDatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": {
                        "class": "DataHandlerLP",
                        "module_path": "qlib.data.dataset.handler",
                        "kwargs": data_handler_config,
                    },
                    "segments": {
                        "train": (start_time, fit_end_time),
                        "valid": (val_start_time, val_end_time),
                        "test": (test_start_time, test_end_time),
                    },
                    "enable_cache": True ,
                    "step_len": args.step_len, 
                },
            },
        }
        

        # model initialization
        model = init_instance_by_config(task["model"])
        dataset = init_instance_by_config(task["dataset"])
        model.fit(dataset)
        score = model.predict(dataset)
        score.name = 'score'
        score = score.to_frame()
        print(score.head())
        # 测试集的图
        label = get_label(dataset, segment="test")
        label.columns = ["label"]
        
        pred_label = pd.concat([label, score], axis=1, sort=True).reindex(label.index)
        fig, = analysis_position.score_ic_graph(pred_label, show_notebook=False)
        # 保存图
        # fig.savefig(f"{save_path}/score_ic_graph.png")
        fig, = analysis_position.top_score_ic_graph(pred_label, show_notebook=False)
        # fig.savefig(f"{save_path}/top_score_ic_graph.png")