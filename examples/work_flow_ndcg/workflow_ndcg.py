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
    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    market = "csiall"
    benchmark = "SH000905"
    # 学习的参数
    learn_processors = [
        {"class": "ProcessInfHXY", "kwargs": {}}, # 替换为 nan
        {"class": "DropnaLabel"},
        {"class": "CSRankNorm", "kwargs": {"fields_group": "feature", "parallel": True, "n_jobs": 16}},
        {"class": "CSZFillna", "kwargs": {"fields_group":"feature", "parallel": True, "n_jobs": 16}},
    ]
    # 测试集
    infer_processors = [
        {"class": "ProcessInfHXY", "kwargs": {}}, # 替换为 nan
        {"class": "CSRankNorm", "kwargs": {"fields_group": "feature", "parallel": True, "n_jobs": 16}},
        {"class": "CSZFillna", "kwargs": {"fields_group": "feature", "parallel": True, "n_jobs": 16}},
    ]
    start_time = "2011-12-31"  # 整个开始日期
    fit_end_time = "2019-12-31" # 训练集结束
    val_start_time = "2020-01-01" # 验证集开始
    val_end_time = "2021-12-31" # 验证集结束
    test_start_time = "2022-01-01" # 测试集开始
    end_time = "2022-12-31" # 整个结束日期
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
                "dropout": 0.4,
                "n_epochs": 20,
                "lr": 5e-4,
                "early_stop": 10,
                "batch_size": 5000,
                "metric": "ndcg",
                "loss": "cross_entropy",
                "n_jobs": 30,
                "GPU": 0,
                "sigma": 1.0,
                "n_layer": 5,
                "linear_ndcg": True,
                "debug": True,  # Set to True for debugging mode
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
                
            },
        },
    }


    # model initialization
    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])
    model.fit(dataset)