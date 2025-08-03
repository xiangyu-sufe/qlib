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
from qlib.utils import init_instance_by_config, flatten_dict
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
    parser.add_argument("--save_path", type=str, default=".")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    args = parser.parse_args()
    save_path = args.save_path

    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    market = "csiall"
    benchmark = "SH000905"
    # 
    infer_processors = [
        {"class": "ProcessInfHXY", "kwargs": {}}, # 替换为 nan
        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature",}},
        {"class": "Fillna", 'kwargs': {'fields_group': 'feature'}},
    ]
    # MSE PR loss label 不用处理, 截面 zscore 处理
    learn_processors = [
        {"class": "DropnaLabel"},
        {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
    ]

    start_time = "2009-12-31"  # 整个开始日期
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
        "drop_raw": True,
    }   

    task = {
        "model": {
            "class": "GRU",
            "module_path": "qlib.contrib.model.pytorch_gru_ts",
            "kwargs": {
                "d_feat": 158,
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.0,
                "n_epochs": 40,
                "batch_size": 1,
                "lr": args.lr,
                "early_stop": 10,
                "metric": "loss",
                "loss": "ranking",
                "seed": 0,
                "n_jobs": 50,
                "GPU": 0,  # 当使用CUDA_VISIBLE_DEVICES时，总是使用设备0
                "lambda_reg": args.lambda_reg,
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
    pred = model.predict(dataset)
    print(pred.head())