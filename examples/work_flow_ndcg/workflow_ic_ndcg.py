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
from qlib.contrib.report import analysis_position
from qlib.utils import init_instance_by_config
from qlib.utils.hxy_utils import get_label, prepare_task_pool
from torch.utils.data import TensorDataset, DataLoader
import torch
import pandas as pd
import os

if __name__ == "__main__":
    # 数据参数
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--onlyrun_task_id", type=int, default=None, help="Only run task id")
    parser.add_argument("--onlyrun_seed_id", type=int, default=0, help="Only run specified seed id")
    parser.add_argument("--pv1pv5", type=int, default=1, help="PV1 or PV5 day setting")
    parser.add_argument("--fake", action="store_true", default=False, help="Fake data")
    parser.add_argument("--gpu", type=int, default=0,)
    # 数据集长度参数
    parser.add_argument("--train_length", type=int, default=2400, help="Training dataset length")
    parser.add_argument("--valid_length", type=int, default=240, help="Validation dataset length")
    parser.add_argument("--test_length", type=int, default=240, help="Test dataset length")

    # 时间范围参数
    parser.add_argument("--start_time", type=str, default="2019-12-31", help="Start time for data")
    parser.add_argument("--end_time", type=str, default="2024-12-31", help="End time for data")
    # 模型参数
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save_path", type=str, default=".")
    parser.add_argument("--sigma", type=float, default=3.03)
    parser.add_argument("--combine_type", type=str, default="null", ) # 需要设置
    parser.add_argument("--weight", type=float, default=0.7, )
    parser.add_argument("--step_len", type=int, default=10, )
    args = parser.parse_args()
    save_path = args.save_path
    save_path = os.path.join(save_path, f'seed{args.onlyrun_seed_id}')
    os.makedirs(save_path, exist_ok=True)

    root_dir = os.path.expanduser('~')
    alphamat_path = f'{root_dir}/GRU/alphamat/20250625/data/'
    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    market = "csiall"
    benchmark = "SH000905"

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

    # 数据参数
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

    filter_pipe = [
        {
            "filter_type": "ExpressionDFilter",
            "rule_expression": "$volume > 1e-5",
            "filter_start_time": None,
            "filter_end_time": None,
            "keep": False
        }
    ]
    # 创建数据处理器缓存
    handler_cache_path = os.path.join(save_path, "handler_cache.pkl")
    
    # 检查是否已有缓存
    if os.path.exists(handler_cache_path):
        print(f"✅ 找到数据处理器缓存: {handler_cache_path}")
        # 使用缓存的处理器配置
        handler_config = f"file://{handler_cache_path}"
    else:
        print("🔄 创建新的数据处理器...")
        # 使用整个时间范围创建处理器，这样可以覆盖所有任务
        data_handler_config = {
            "start_time": args.start_time,
            "end_time": args.end_time,
            "fit_start_time": args.start_time,
            "fit_end_time": args.end_time,
            "instruments": market,
            "infer_processors": infer_processors,
            "learn_processors": learn_processors,
            "drop_raw": True,
            "filter_pipe": filter_pipe,
        }
        
        # 创建并保存处理器
        from qlib.contrib.data.handler import Alpha158
        handler = Alpha158(**data_handler_config)
        handler.to_pickle(handler_cache_path, dump_all=True)
        print(f"✅ 数据处理器已保存到: {handler_cache_path}")
        handler_config = f"file://{handler_cache_path}"
    
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

        task = {
            "model": {
                "class": "GRUNDCG",
                "module_path": "qlib.contrib.model.pytorch_gru_ts_ic_ndcg",
                "kwargs": {
                    "d_feat": 158,
                    "hidden_size": 64,
                    "num_layers": 2,
                    "dropout": 0.0,
                    "n_epochs": 50,
                    "batch_size": 5000,
                    "lr": args.lr,
                    "early_stop": 10,
                    "metric": "ndcg",
                    "loss": "ic",
                    "n_jobs": 24,
                    "GPU": args.gpu,
                    "seed": args.onlyrun_seed_id,
                    "sigma": args.sigma,
                    "n_layer": 5,
                    "weight": args.weight,
                    "linear_ndcg": False,
                    "combine_type":args.combine_type,
                    "debug": True,  # Set to True for debugging mode
                    "save_path": f"{save_path}/task_{task_id}"
                },
            },
            "dataset": {
                "class": "TSDatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": handler_config,  # 使用缓存的处理器
                    "segments": {
                        "train": (start_time, fit_end_time),
                        "valid": (val_start_time, val_end_time),
                        "test": (test_start_time, test_end_time),
                    },
                    "step_len": args.step_len,
                },
            },
        }
        

        # model initialization
        if args.fake:
            model = init_instance_by_config(task["model"])
            print("使用假数据测试 model")
            N, T, D = 5000, 20, 159
            batch_size = 10
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            features = torch.randn(N, T, D, dtype=torch.float32, device=device)
            # 标签数据: (N,) = (1000,)
            labels = torch.randn(N, dtype=torch.float32, device=device)
            
            # 创建数据集和数据加载器
            dataset = TensorDataset(features, labels)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            model.train_epoch(dataloader)
            model.test_epoch(dataloader)
        else:
            model = init_instance_by_config(task["model"])
            dataset = init_instance_by_config(task["dataset"])
            model.fit(dataset)
            score = model.predict(dataset)
            score.name = 'score'
            # 保存预测值
            score.to_csv(f"{save_path}/task_{task_id}/test.csv")
            score = score.to_frame()
            print(score.head())
            # 测试集的图
            label = get_label(dataset, segment="test")
            label.columns = ["label"]
            
            pred_label = pd.concat([label, score], axis=1, sort=True).reindex(score.index)
            
            # 确保保存目录存在
            import os
            os.makedirs(f"{save_path}/task_{task_id}", exist_ok=True)
            
            # 保存图片 - 使用多种方法
            try:
                fig, = analysis_position.score_ic_graph(pred_label, show_notebook=False)
                # 方法1: 尝试保存为PNG
                try:
                    fig.write_image(f"{save_path}/task_{task_id}/score_ic_graph.png")
                    print("✅ 成功保存 score_ic_graph.png")
                except Exception as png_error:
                    print(f"⚠️ PNG保存失败: {png_error}")
                    # 方法2: 保存为HTML文件
                    try:
                        fig.write_html(f"{save_path}/task_{task_id}/score_ic_graph.html")
                        print("✅ 成功保存 score_ic_graph.html (可在浏览器中查看)")
                    except Exception as html_error:
                        print(f"⚠️ HTML保存失败: {html_error}")
                        # 方法3: 保存为JSON文件
                        try:
                            fig.write_json(f"{save_path}/task_{task_id}/score_ic_graph.json")
                            print("✅ 成功保存 score_ic_graph.json (plotly格式)")
                        except Exception as json_error:
                            print(f"❌ 所有保存方法都失败: {json_error}")
            except Exception as e:
                print(f"❌ 生成 score_ic_graph 时出错: {e}")
                
            try:
                fig, = analysis_position.top_score_ic_graph(pred_label, show_notebook=False)
                # 方法1: 尝试保存为PNG
                try:
                    fig.write_image(f"{save_path}/task_{task_id}/top_score_ic_graph.png")
                    print("✅ 成功保存 top_score_ic_graph.png")
                except Exception as png_error:
                    print(f"⚠️ PNG保存失败: {png_error}")
                    # 方法2: 保存为HTML文件
                    try:
                        fig.write_html(f"{save_path}/task_{task_id}/top_score_ic_graph.html")
                        print("✅ 成功保存 top_score_ic_graph.html (可在浏览器中查看)")
                    except Exception as html_error:
                        print(f"⚠️ HTML保存失败: {html_error}")
                        # 方法3: 保存为JSON文件
                        try:
                            fig.write_json(f"{save_path}/task_{task_id}/top_score_ic_graph.json")
                            print("✅ 成功保存 top_score_ic_graph.json (plotly格式)")
                        except Exception as json_error:
                            print(f"❌ 所有保存方法都失败: {json_error}")
            except Exception as e:
                print(f"❌ 生成 top_score_ic_graph 时出错: {e}")
            
            print(f"任务 {task_id} 完成")