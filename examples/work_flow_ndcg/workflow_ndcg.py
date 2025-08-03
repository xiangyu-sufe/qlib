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
from qlib.utils.hxy_utils import get_label
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData
from qlib.tests.config import CSI300_BENCH, CSI300_GBDT_TASK
from qlib.data.dataset.handler import DataHandlerLP
from torch.utils.data import Sampler
import pandas as pd
import numpy as np


if __name__ == "__main__":
    # 数据参数
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--save_path", type=str, default=".")
    parser.add_argument("--sigma", type=float, default=3.03)
    parser.add_argument("--start", type=str, required=True, help="Start date for the dataset")
    args = parser.parse_args()
    save_path = args.save_path
    start_time = args.start

    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    market = "csiall"
    benchmark = "SH000905"
    # 
    infer_processors = [
        {"class": "ProcessInfHXY", "kwargs": {}}, # 替换为 nan
        # {"class": "CSRankNorm", "kwargs": {"fields_group": "feature", 'parallel':True, 'n_jobs': 60}},
        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature",}},
        {"class": "Fillna", 'kwargs': {'fields_group': 'feature'}},
    ]
    # 排序学习 label 不用处理
    learn_processors = [
        {"class": "DropnaLabel"},
    ]

    start_time = start_time  # 整个开始日期
    fit_end_time = "2019-12-31" # 训练集结束
    val_start_time = "2020-01-01" # 验证集开始
    val_end_time = "2020-12-31" # 验证集结束
    test_start_time = "2021-01-01" # 测试集开始
    end_time = "2021-12-31" # 整个结束日期
    ###################################
    # train model
    ###################################


    data_handler_config = {
        "start_time": start_time,
        "end_time": end_time,
        "fit_start_time": start_time,
        "fit_end_time": fit_end_time,
        "instruments": market,
        "infer_processors":infer_processors,
        "learn_processors":learn_processors,
        # "infer_processors":[],
        # "learn_processors":[],
        "drop_raw": True,
    }   

    task = {
        "model": {
            "class": "GRUNDCG",
            "module_path": "qlib.contrib.model.pytorch_gru_ts_ndcg",
            "kwargs": {
                "d_feat": 158,
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.0,
                "n_epochs": 1,
                "batch_size": 1,
                "lr": args.lr,
                "early_stop": 10,
                "metric": "ndcg",
                "loss": "cross_entropy",
                "n_jobs": 50,
                "GPU": 0,
                "sigma": args.sigma,
                "n_layer": 5,
                "linear_ndcg": True,
                "debug": True,  # Set to True for debugging mode
                "save_path": save_path
            },
        },
        "dataset": {
            "class": "TSDatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "Alpha158",
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": data_handler_config,
                },
                "segments": {
                    "train": (start_time, fit_end_time),
                    "valid": (val_start_time, val_end_time),
                    "test": (test_start_time, end_time),
                },
                "enable_cache": True ,
                
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
    fig = analysis_position.score_ic_graph(pred_label, show_notebook=False)
    # 保存图
    fig.savefig(f"{save_path}/score_ic_graph.png")
    fig = analysis_position.top_score_ic_graph(pred_label, show_notebook=False)
    fig.savefig(f"{save_path}/top_score_ic_graph.png")