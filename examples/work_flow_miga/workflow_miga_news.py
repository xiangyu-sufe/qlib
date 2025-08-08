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
    NewsStore, make_collate_fn, write_lmdb, IndexedSeqDataset, read_ohlc,
    is_month_end_trade_day
)
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData
from qlib.tests.config import CSI300_BENCH, CSI300_GBDT_TASK
from qlib.data.dataset.handler import DataHandlerLP
from torch.utils.data import Sampler, DataLoader
from qlib.data.dataset.loader import StaticDataLoader, QlibDataLoader  

import pandas as pd
import random, numpy as np, torch
import numpy as np
import os


if __name__ == "__main__":
    # 数据参数
    import argparse

    import time
    parser = argparse.ArgumentParser()
    # 实验参数
    parser.add_argument("--onlyrun_task_id", type=int, nargs="+", default=None, help="Only run task id")
    parser.add_argument("--onlyrun_seed_id", type=int, default=0, help="Only run specified seed id")
    parser.add_argument("--pv1pv5", type=int, default=1, help="PV1 or PV5 day setting")
    parser.add_argument("--step_len", type=int, default=20, help="Step length")
    parser.add_argument("-v", "--version", type=int, default=1, help="Version of the model")
    parser.add_argument("--ohlc", action="store_true",  help="Use ohlc data")
    parser.add_argument("--n_jobs", type=int, default=0, help="Number of jobs for parallel processing")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    # 数据集长度参数
    parser.add_argument("--train_length", type=int, default=720, help="Training dataset length")
    parser.add_argument("--valid_length", type=int, default=240, help="Validation dataset length")
    parser.add_argument("--test_length", type=int, default=120, help="Test dataset length")
    # 时间范围参数
    parser.add_argument("--start_time", type=str, default="2021-12-31", help="Start time for data")
    parser.add_argument("--end_time", type=str, default="2025-05-31", help="End time for data")
    # 模型一般参数
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save_path", type=str, default=".")
    parser.add_argument("--lambda_reg", type=float, default=1)
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--early_stop", type=int, default=10)
    # MIGA模型结构参数
    parser.add_argument("--d_feat", type=int, default=158, help="Feature dimension")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension for router (auto-calculated as num_groups * num_experts_per_group if not specified)")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--num_groups", type=int, default=4, help="Number of expert groups")
    parser.add_argument("--num_experts_per_group", type=int, default=4, help="Number of experts per group")
    parser.add_argument("--num_heads", type=int, default=1, help="Number of attention heads")
    parser.add_argument("--top_k", type=int, default=4, help="Top-k experts selection")
    parser.add_argument("--expert_output_dim", type=int, default=1, help="Expert output dimension")
    parser.add_argument("--use_news", action="store_true", help="是否使用新闻数据")
    parser.add_argument("--padding_method", type=str, default="zero", choices=["zero", "learn"], help="Padding method for news")
    # MIGA损失函数参数
    parser.add_argument("--loss", type=str, default="miga", choices=["miga", "ic"], help="Loss function")
    parser.add_argument("--metric", type=str, default="ic", choices=["ic", "rankic"], help="Metric function")
    parser.add_argument("--omega", type=float, default=2e-2, help="Router loss weight")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Expert loss weight")
    parser.add_argument("--omega_scheduler", type=str, default="exp", choices=["none", "exp", "step"], help="omega动态调整策略: none/exp/step")
    parser.add_argument("--omega_decay", type=float, default=0.96, help="指数衰减率（exp模式）")
    parser.add_argument("--omega_step_epoch", type=int, default=10, help="Step epoch for omega")
    parser.add_argument("--omega_after", type=float, default=1.0, help="Omega after step epoch")
    args = parser.parse_args()
    args.start_time = is_month_end_trade_day(args.start_time)[0]
    args.end_time = is_month_end_trade_day(args.end_time)[0]
    # ---------------- Deterministic seed ----------------
    GLOBAL_SEED = getattr(args, "onlyrun_seed_id", 0) or 0

    def _set_global_deterministic_seed(seed: int = 0):
        """Set seeds for reproducibility across random, numpy, torch (CPU & GPU) and configure deterministic CUDA kernels."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # deterministic behavior for cuDNN / cuBLAS
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    _set_global_deterministic_seed(GLOBAL_SEED)
    # ----------------------------------------------------
    save_path = args.save_path
    save_path = os.path.join(save_path, f'seed{args.onlyrun_seed_id}')
    os.makedirs(save_path, exist_ok=True)

    root_dir = os.path.expanduser('~')
    alphamat_path = f'{root_dir}/GRU/alphamat/20250625/data/'
    market = None # 默认用全部
    benchmark = "SH000905"
    # 读取新闻数据
    if args.use_news:
        a = time.time()
        news_lmdb_path = '/home/huxiangyu/.qlib/llm_data/embedding.lmdb'
        if not os.path.exists(news_lmdb_path):
            news_embed = pd.read_pickle('/home/huxiangyu/.qlib/llm_data/embedding.pkl')
            write_lmdb(news_embed, news_lmdb_path)
        # news_store = NewsStore(news_lmdb_path)
        # print(f'新闻数据占用内存大小: {news_store.memory_usage} MB')
    # 读取量价数据
    if args.ohlc:
        print("读取高开低收数据")
        assert args.d_feat == 6
        a = time.time()
        ohlc = read_ohlc()
        labels = read_label(day=10, method = 'win+neu+zscore')
        data = ohlc.join(labels, how='left')
        
        data.columns = pd.MultiIndex.from_tuples(
            [('feature', col) for col in ohlc.columns] 
            + [('label', col) for col in labels.columns]
            )
        print("读取所有数据用时: ", time.time() - a)
        print(f"量价数据占用内存大小: {data.memory_usage().sum() / 1e6} MB")
    elif args.alpha64:
        a = time.time()
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

        if args.ohlc:
            infer_processors = [
                {"class": "ProcessInfHXY", "kwargs": {}}, # 替换为 nan
            ]
            
            # 不用处理
            learn_processors = [
                {"class": "DropnaLabel"},
            ]    
        elif args.alpha64:
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
                "module_path": "qlib.contrib.model.pytorch_miga_ts_news",
                "kwargs": {
                    "d_feat": args.d_feat, # 模型参数
                    "seed": GLOBAL_SEED, # 种子
                    "hidden_size": args.hidden_dim,
                    "num_groups": args.num_groups,
                    "num_experts_per_group": args.num_experts_per_group,
                    "num_heads": args.num_heads,
                    "top_k": args.top_k,
                    "expert_output_dim": 1,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "n_epochs": args.n_epochs, # 训练参数
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "early_stop": args.early_stop,
                    "metric": args.metric,
                    "loss": args.loss,
                    "n_jobs": args.n_jobs,
                    "GPU": args.gpu,
                    "lambda_reg": args.lambda_reg, # 损失参数
                    "omega": args.omega,
                    "epsilon": args.epsilon,
                    "omega_scheduler": args.omega_scheduler,
                    "omega_decay": args.omega_decay,
                    "debug": True,  # Set to True for debugging mode
                    "save_path": save_path,
                    "step_len": args.step_len,
                    "news_store": news_lmdb_path,
                    "use_news" : args.use_news,
                    "ohlc": args.ohlc,
                    "padding_method": args.padding_method,
                    "version": "B" + str(args.version),
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

        # Ensure DataLoader workers are deterministic
        def _seed_worker(worker_id: int):
            worker_seed = GLOBAL_SEED + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)
        # inject into model so it can be used when constructing DataLoader
        model.extra_worker_init_fn = _seed_worker
        model.debug_timing = True
        model.fit(dataset)
        # 在训练集预测
        # score_train = model.predict_train(dataset)
        # score_train.name = 'score'
        # score_train = score_train.to_frame()
        # label_train = get_label(dataset, segment="train")
        # label_train.columns = ["label"]
        
        # pred_label_train = pd.concat([label_train, score_train], axis=1, sort=True).reindex(label_train.index)
        # fig, = analysis_position.score_ic_graph(pred_label_train, show_notebook=False)
        # # 保存图
        # # fig.savefig(f"{save_path}/score_ic_graph.png")
        # fig, = analysis_position.top_score_ic_graph(pred_label_train, show_notebook=False)
        # fig.savefig(f"{save_path}/top_score_ic_graph.png")
        
        # 在测试集预测
        score = model.predict(dataset)
        score.name = 'score'
        score = score.to_frame()
        print(score.head())
        # 测试集的图
        label = get_label(dataset, segment="test")
        label.columns = ["label"]
        
        pred_label = pd.concat([label, score], axis=1, sort=True).reindex(label.index)
        fig, _ = analysis_position.score_ic_graph(pred_label, show_notebook=False)
        # 保存图
        # fig.savefig(f"{save_path}/score_ic_graph.png")
        # fig, = analysis_position.top_score_ic_graph(pred_label, show_notebook=False)
        # fig.savefig(f"{save_path}/top_score_ic_graph.png")