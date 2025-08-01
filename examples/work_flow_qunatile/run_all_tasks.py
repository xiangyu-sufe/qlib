from qlib.model.trainer import task_train
import qlib
from qlib.constant import REG_CN


# 你想测试的不同 processor 组合
processor_configs = [
    {
        "learn_processors": [
            {"class": "ProcessInf", "kwargs": {}},
            {"class": "DropnaLabel"},
            {"class": "CSZFillna", "kwargs": {"fields_group": "feature"}},
            {"class": "CSZScoreNorm", "kwargs": {"fields_group": ["feature"]}}
        ],
        "infer_processors": [
            {"class": "ProcessInf", "kwargs": {}},
            {"class": "CSZFillna", "kwargs": {"fields_group": "feature"}},
            {"class": "CSZScoreNorm", "kwargs": {"fields_group": ["feature"]}}
        ]
    },
    {
        "learn_processors": [
            {"class": "ProcessInf", "kwargs": {}},
            {"class": "DropnaLabel"},
            {"class": "CSZFillna", "kwargs": {"fields_group": "feature"}},
            {"class": "CSRankNorm", "kwargs": {"fields_group": ["feature"]}}
        ],
        "infer_processors": [
            {"class": "ProcessInf", "kwargs": {}},
            {"class": "CSZFillna", "kwargs": {"fields_group": "feature"}},
            {"class": "CSRankNorm", "kwargs": {"fields_group": ["feature"]}}
        ]
    },
    {
        "learn_processors": [
            {"class": "ProcessInf", "kwargs": {}},
            {"class": "DropnaLabel"},
            {"class": "CSZFillna", "kwargs": {"fields_group": "feature"}},
            {"class": "ZScoreNorm", "kwargs": {"fields_group": ["feature"]}}
        ],
        "infer_processors": [
            {"class": "ProcessInf", "kwargs": {}},
            {"class": "CSZFillna", "kwargs": {"fields_group": "feature"}},
            {"class": "ZScoreNorm", "kwargs": {"fields_group": ["feature"]}}
        ]
    },
    {
        "learn_processors": [
            {"class": "ProcessInf", "kwargs": {}},
            {"class": "DropnaLabel"},
            {"class": "CSZFillna", "kwargs": {"fields_group": "feature"}},
            {"class": "RobustZScoreNorm", 
             "kwargs": {"fields_group": ["feature"]}}
        ],
        "infer_processors": [
            {"class": "ProcessInf", "kwargs": {}},
            {"class": "CSZFillna", "kwargs": {"fields_group": "feature"}},
            {"class": "RobustZScoreNorm", 
             "kwargs": {"fields_group": ["feature"]}}
        ]
    },
]

def run_tasks_with_different_processors(processor_configs):
    for i, config in enumerate(processor_configs):
        print(f"Running config {i+1}/{len(processor_configs)}")
        
        # 确保没有活跃的 run
        try:
            import mlflow
            if mlflow.active_run():
                mlflow.end_run()
        except Exception:
            pass
            
        start_time = "2012-06-30"  # 整个开始日期
        fit_end_time = "2020-06-30"  # 训练集结束
        val_start_time = "2020-07-01"  # 验证集开始
        val_end_time = "2021-12-31"  # 验证集结束
        test_start_time = "2022-01-01"  # 测试集开始
        end_time = "2022-12-31"  # 整个结束日期
        # 复制数据配置
        data_handler_config = {
            "start_time": start_time,
            "end_time": end_time,
            "fit_start_time": start_time,
            "fit_end_time": fit_end_time,
            "instruments": "csiall",  # 或者你的 `market` 变量
            "infer_processors": config["infer_processors"],
            "learn_processors": config["learn_processors"],
        }

        task_config = {
            "model": {
                "class": "GRUNDCG",
                "module_path": "qlib.contrib.model.pytorch_gru_ts_ndcg",
                "kwargs": {
                    "d_feat": 158,
                    "hidden_size": 64,
                    "num_layers": 2,
                    "dropout": 0.0,
                    "n_epochs": 30,
                    "lr": 1e-4,
                    "early_stop": 10,
                    "batch_size": 5000,
                    "metric": "ndcg",
                    "loss": "cross_entropy",
                    "n_jobs": 20,
                    "GPU": 0,
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
        provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
        qlib.init(provider_uri=provider_uri, region=REG_CN)        
        # 可以设置不同任务名
        experiment_name = f"processor_test_config_{i}"
            
        # 直接调用 task_train，不包装在 R.start 中
        task_train(task_config, experiment_name=experiment_name)


run_tasks_with_different_processors(processor_configs)