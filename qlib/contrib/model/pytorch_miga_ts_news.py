# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function
from collections import defaultdict
import warnings
from typing import Dict, List, Tuple, Optional, DefaultDict

import numpy as np
from tqdm import tqdm
import pandas as pd
import copy
from ...utils import get_or_create_path
from ...log import get_module_logger

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

from .pytorch_utils import count_parameters
from ...model.base import Model
from ...data.dataset.handler import DataHandlerLP
from ...model.utils import ConcatDataset
from ...data.dataset.weight import Reweighter
from ..loss.loss import (ic_loss, rankic_loss,
                         topk_ic_loss, topk_rankic_loss,
                           ranking_loss, pairwise_loss, mse)
from ..loss.miga import MIGALoss
from qlib.utils.hxy_utils import (compute_grad_norm, 
                                  compute_layerwise_grad_norm,
                                  IndexedSeqDataset,
                                  make_collate_fn,)

from colorama import Fore, Style, init
import matplotlib.pyplot as plt
import os


# class DailyBatchSampler(Sampler):
#     def __init__(self, data_source):
#         self.data_source = data_source
#         # calculate number of samples in each batch
#         self.daily_count = (
#             pd.Series(index=self.data_source.get_index()).groupby("datetime", group_keys=False).size().values
#         )
#         self.daily_index = np.roll(np.cumsum(self.daily_count), 1)  # calculate begin index of each batch
#         self.daily_index[0] = 0

#     def __iter__(self):
#         for idx, count in zip(self.daily_index, self.daily_count):
#             yield np.arange(idx, idx + count)

#     def __len__(self):
#         return len(self.data_source)
    
    
class DailyBatchSampler(Sampler):
    """
    Yield all rows of the same trading day as one batch,
    independent of the index sort order.
    """
    def __init__(self, data_source):
        self.data_source = data_source
        # 把 datetime -> 行号数组 建立映射
        dts = self.data_source.get_index().get_level_values("datetime")
        self.groups = defaultdict(list)
        for pos, dt in enumerate(dts):
            self.groups[dt].append(pos)
        # 交易日按时间排序，保证训练顺序一致
        self.order = sorted(self.groups.keys())

    def __iter__(self):
        for dt in self.order:
            yield np.array(self.groups[dt])

    def __len__(self):
        return len(self.data_source)
    

class MIGA(Model):
    """
    MIGA模型专用训练器，继承自基础Trainer类
    重写fit函数以适配MIGA模型的特殊输出格式和损失函数
    """
    
    def __init__(
        self,
        d_feat=6, # 模型参数
        hidden_size=64,
        num_groups: int = 4,
        num_experts_per_group: int = 4,
        num_heads: int = 8,
        top_k: int = 2,
        expert_output_dim: int = 1,
        num_layers=2,
        dropout=0.0, # 训练参数
        n_epochs=200,
        lr=0.001,
        metric="ic",
        batch_size=2000,
        early_stop=20,
        loss="miga",
        optimizer="adam",
        n_jobs=10,
        GPU=0,
        seed=None,
        lambda_reg=0.1,
        omega=0.1,
        epsilon=1,
        omega_scheduler=None,
        omega_decay=0.96,
        debug=False,
        save_path=None,
        step_len=1,
        news_store=None,
        **kwargs,
    ):
        # Set logger.
        self.logger = get_module_logger("GRU")
        self.logger.info("GRU pytorch version...")
        
        # set hyper-parameters.
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.num_experts_per_group = num_experts_per_group
        self.num_heads = num_heads
        self.top_k = top_k
        self.expert_output_dim = expert_output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.loss = loss
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.optimizer_type = optimizer
        self.n_jobs = n_jobs
        self.GPU = GPU
        self.seed = seed if seed is not None else 42
        self.lambda_reg = lambda_reg
        self.debug = debug
        self.save_path = save_path
        self.step_len = step_len
        self.news_store = news_store
        
        if  self.loss == "miga":
            self.omega = omega
            self.epsilon = epsilon
            self.omega_scheduler = omega_scheduler
            self.omega_decay = omega_decay
            self.Miga_loss = MIGALoss(omega=self.omega, epsilon=self.epsilon)
            # 定义omega_scheduler
            if self.omega_scheduler == "exp":
                def omega_scheduler(epoch):
                    # 指数衰减
                    # 但超过一定 epoch 数不再变化
                    if epoch < 10:
                        return self.omega * (self.omega_decay ** epoch)
                    else:
                        return self.omega * (self.omega_decay ** 10)     
            elif self.omega_scheduler == "step":
                def omega_scheduler(epoch):
                    return self.omega if epoch < self.omega_step_epoch else self.omega_after
            else:
                omega_scheduler = None
        else:
            raise ValueError("unknown loss `%s`" % self.loss)    
            
        self.logger.info(
            "MIGA parameters setting:"
            "\nd_feat : {}"
            "\nhidden_size : {}"
            "\nnum_groups : {}"
            "\nnum_experts_per_group : {}"
            "\nnum_heads : {}"
            "\ntop_k : {}"
            "\nexpert_output_dim : {}"
            "\nomega : {}"
            "\nepsilon : {}"
            "\nomega_scheduler : {}"
            "\nomega_decay : {}"
            "\nnum_layers : {}"
            "\ndropout : {}"
            "\nn_epochs : {}"
            "\nlr : {}"
            "\nmetric : {}"
            "\nbatch_size : {}"
            "\nearly_stop : {}"
            "\noptimizer : {}"
            "\nloss_type : {}"
            "\ndevice : {}"
            "\nn_jobs : {}"
            "\nuse_GPU : {}"
            "\nseed : {}"
            "\ndebug : {}"
            "\nsave_path : {}".format(
                d_feat,
                hidden_size,
                num_groups,
                num_experts_per_group,
                num_heads,
                top_k,
                expert_output_dim,
                omega,
                epsilon,
                omega_scheduler,
                omega_decay,
                num_layers,
                dropout,
                n_epochs,
                lr,
                metric,
                batch_size,
                early_stop,
                optimizer.lower(),
                loss,
                self.device,
                n_jobs,
                self.use_gpu,
                self.seed,
                self.debug,
                self.save_path
            )
        )
    
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        
        # 路由器设置
        
        router = PriceNewsRouter(
            price_dim=self.d_feat,
            news_dim=1024,
            d_model=self.hidden_size,
            n_heads=self.num_heads, # 这里的 num_heads 共用了
            d_gru=self.hidden_size,
            dropout=self.dropout
        )

        # 定义 MIGA 新闻模型
        self.MIGA_model = MIGANewsModel(
            input_dim=self.d_feat,
            news_dim=1024,
            num_groups=self.num_groups,
            num_experts_per_group=self.num_experts_per_group,
            num_heads=self.num_heads,
            top_k=self.top_k,
            expert_output_dim=self.expert_output_dim,
            router=router,
        )
        self.logger.info("model:\n{:}".format(self.MIGA_model))
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.MIGA_model)))

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.MIGA_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.MIGA_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.MIGA_model.to(self.device)
    

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label, weight):
        loss = weight * (pred - label) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label, hidden,weight=None):
        mask = ~torch.isnan(label)
        return self.Miga_loss(pred[mask], label[mask], hidden)

    def metric_fn(self, pred, label, name, topk=None):
        mask = torch.isfinite(label)

        if name in ("", "loss", "ic", "rankic", "topk_ic", "topk_rankic"):
            if name == "ic":
                return -ic_loss(pred[mask], label[mask]).item()
            elif name == "rankic":
                return -rankic_loss(pred[mask], label[mask]).item()
            elif name == "topk_ic":
                if topk is None:
                    warnings.warn("topk must be specified for topk_ic metric, return nan")
                    return torch.nan
                return -topk_ic_loss(pred[mask], label[mask], k=topk).item()
            elif name == "topk_rankic":
                if topk is None:
                    warnings.warn("topk must be specified for topk_ic metric, return nan")
                    return torch.nan
                return -topk_rankic_loss(pred[mask], label[mask], k=topk).item()
            elif name == "loss":
                return -self.loss_fn(pred[mask], label[mask]).item()

        raise ValueError("unknown metric `%s`" % name)
        

    def train_epoch(self, data_loader):
        self.MIGA_model.train()
        tot_loss_list = []
        result: DefaultDict[str, np.floating] = defaultdict(lambda: np.float64(np.nan))        
        epoch_grad_norms = []
        epoch_grad_norms_layer = []
        mse_loss_list = []
        pairwise_loss_list = []
        expert_loss_list = []
        router_loss_list = []

            
        for data, news_mask in tqdm(data_loader):
            data.squeeze_(0) # 去除横截面 dim
            news_mask.squeeze_(0)
            price_feature = data[:,:, :self.d_feat].to(self.device)
            label = data[:, -1, -1].to(self.device)
            news_feature = data[:,:, self.d_feat+1:].to(self.device)
            news_mask = news_mask.to(self.device)
            
            output = self.MIGA_model(price_feature, news_feature, news_mask)
            pred = output['predictions']
            hidden_representations = output['hidden_representations']
            
            loss_dict = self.loss_fn(pred, label, hidden_representations,)
            self.train_optimizer.zero_grad()
            loss = loss_dict['total_loss']
            loss.backward()
            
            torch.nn.utils.clip_grad_value_(self.MIGA_model.parameters(), 3.0)
            self.train_optimizer.step()

            if self.debug:
                # 计算MSE 和 pairwise loss 相对大小
                with torch.no_grad():
                    expert_loss = loss_dict['expert_loss'].item()
                    router_loss = loss_dict['router_loss'].item()
                    mse_loss = mse(pred, label).item()
                    pr_loss = pairwise_loss(pred, label).item()
                    mse_loss_list.append(mse_loss)
                    pairwise_loss_list.append(pr_loss)
                    expert_loss_list.append(expert_loss)
                    router_loss_list.append(router_loss)
                # 计算梯度范数
                grad_norm = compute_grad_norm(self.MIGA_model)
                grad_norm_layer = compute_layerwise_grad_norm(self.MIGA_model)
                epoch_grad_norms.append(grad_norm)
                epoch_grad_norms_layer.append(grad_norm_layer)

            tot_loss_list.append(loss_dict['total_loss'].item())

        result["train"] = np.mean(tot_loss_list)

        if self.debug:
            self.logger.debug(f"{Fore.RED} MSE Loss: {np.mean(mse_loss_list):.6f}, Pairwise Loss: {np.mean(pairwise_loss_list):.6f}{Style.RESET_ALL}")
            self.logger.debug(f"{Fore.RED} Expert Loss: {np.mean(expert_loss_list):.6f}, Router Loss: {np.mean(router_loss_list):.6f}{Style.RESET_ALL}")
            avg_grad_norm = np.mean(epoch_grad_norms)
            self.logger.debug(f"{Fore.RED}Epoch Avg Grad Norm: {avg_grad_norm:.6f}{Style.RESET_ALL}")
            # 计算每层的平均梯度范数
            if epoch_grad_norms_layer:
                layer_names = epoch_grad_norms_layer[0].keys()
                for layer_name in layer_names:
                    layer_norms = []
                    for batch_layer in epoch_grad_norms_layer:
                        if isinstance(batch_layer[layer_name], list):
                            layer_norms.extend(batch_layer[layer_name])
                        else:
                            layer_norms.append(batch_layer[layer_name])
                    avg_layer_norm = np.mean(layer_norms)
                    self.logger.debug(f"Epoch Avg {layer_name} Grad Norm: {avg_layer_norm:.6f}")

        return result
    

    def test_epoch(self, data_loader):
        self.MIGA_model.eval()

        scores = []
        losses = []
        ic_scores = []
        rankic_scores = []
        topk_ic_scores = []
        topk_rankic_scores = []
        
  
        for data, news_mask in data_loader:
            data.squeeze_(0) # 去除横截面 dim
            news_mask.squeeze_(0)
            price_feature = data[:,:, :self.d_feat].to(self.device)
            label = data[:, -1, -1].to(self.device)
            news_feature = data[:,:, self.d_feat+1:].to(self.device)
            news_mask = news_mask.to(self.device)
            
            with torch.no_grad():
                output = self.MIGA_model(price_feature.float(), news_feature.float(), news_mask)
                pred = output['predictions']
                hidden_representations = output['hidden_representations']
                loss_dict = self.loss_fn(pred, label, hidden_representations)
                loss = loss_dict['total_loss'].item()
                expert_loss = loss_dict['expert_loss'].item()
                router_loss = loss_dict['router_loss'].item()
                # score = self.metric_fn(pred, label, name = 'loss') # 暂时没想好是否应该用这个做 score
                ic_score = self.metric_fn(pred, label, "ic")
                score = ic_score 
                rankic_score = self.metric_fn(pred, label, "rankic")
                topk_ic_score = self.metric_fn(pred, label, "topk_ic", topk=5)
                topk_rankic_score = self.metric_fn(pred, label, "topk_rankic", topk=5)

                losses.append(loss)
                scores.append(score)
                ic_scores.append(ic_score)
                rankic_scores.append(rankic_score)
                topk_ic_scores.append(topk_ic_score)
                topk_rankic_scores.append(topk_rankic_score)

        result: DefaultDict[str, np.floating] = defaultdict(lambda: np.float64(np.nan))        
        result["loss"] = np.mean(losses)
        result["score"] = np.mean(scores)
        result["ic"] = np.mean(ic_scores)
        result["rankic"] = np.mean(rankic_scores)
        result["topk_ic"] = np.mean(topk_ic_scores)
        result["topk_rankic"] = np.mean(topk_rankic_scores)

        return result
    


    def fit(
        self,
        dataset,
        evals_result=dict(),
        save_path=None,
        reweighter=None,
    ):
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        self.train_index = dl_train.get_index()
        self.val_index = dl_valid.get_index()
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        dl_train.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
        dl_valid.config(fillna_type="ffill+bfill")  # process nan brought by dataloader

        # daily batch sampler
        sampler_train = DailyBatchSampler(dl_train)
        sampler_valid = DailyBatchSampler(dl_valid)
        # index 下
        dl_train = IndexedSeqDataset(dl_train)
        dl_valid = IndexedSeqDataset(dl_valid)

        if reweighter is None:
            wl_train = np.ones(len(dl_train))
            wl_valid = np.ones(len(dl_valid))
        elif isinstance(reweighter, Reweighter):
            wl_train = reweighter.reweight(dl_train)
            wl_valid = reweighter.reweight(dl_valid)
        else:
            raise ValueError("Unsupported reweighter type.")

        train_loader = DataLoader(
            dl_train,
            sampler=sampler_train,
            num_workers=self.n_jobs,
            collate_fn=make_collate_fn(self.news_store, self.step_len),
            drop_last=True,
        )
        valid_loader = DataLoader(
            dl_valid,
            sampler=sampler_valid,
            num_workers=self.n_jobs,
            collate_fn=make_collate_fn(self.news_store, self.step_len),
            drop_last=True,
        )

        save_path = get_or_create_path(save_path)
        
        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []
        evals_result["train_score"] = []
        evals_result["valid_score"] = []
        # train
        self.logger.info("training...")
        self.fitted = True
        best_param = None
        
        for step in tqdm(range(self.n_epochs)):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            result = self.train_epoch(train_loader)
            evals_result["train"].append(result["train"])
            evals_result["train_score"].append(result["score"])
            self.logger.info("evaluating...")
            result = self.test_epoch(valid_loader)
            
            self.logger.info(
                f"{Fore.GREEN}"
                f"valid loss: {result['loss']:.6f}, valid score: {result['score']:.6f}\n"
                f"ic: {result['ic']:.6f}, rankic: {result['rankic']:.6f}, "
                f"topk_ic: {result['topk_ic']:.6f}, topk_rankic: {result['topk_rankic']:.6f}"
                f"{Style.RESET_ALL}"
            )
            val_score = result["score"]
            evals_result["valid"].append(result["loss"])
            evals_result["valid_score"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.MIGA_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        if best_param is not None:
            self.MIGA_model.load_state_dict(best_param)
            torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()
        # 可视化损失
        self.visualize_evals_result(evals_result)        
        
    
    
    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        self.test_index = dl_test.get_index()
        test_loader = DataLoader(dl_test, batch_size=self.batch_size, num_workers=self.n_jobs)
        self.MIGA_model.eval()
        preds = []

        for data in test_loader:
            data.squeeze_(0) # 去除横截面 dim
            feature = data[:, :, 0:-1].to(self.device)

            with torch.no_grad():
                output = self.MIGA_model(feature.float())
                pred = output['predictions'].detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=self.test_index)
    

    def visualize_evals_result(self, evals_result, save_path=None, train_index=None, val_index=None, test_index=None):
        """
        重写可视化函数，专门为MIGA模型设计
        训练和验证损失分开画，每个图显示total、expert、router三条曲线
        新增：在图下方显示训练集、验证集、测试集的时间区间。
        """
        import matplotlib.pyplot as plt
        import os

        def _get_time_range(index):
            if index is None:
                return "N/A"
            if hasattr(index, "get_level_values"):
                try:
                    dates = index.get_level_values("datetime")
                except Exception:
                    dates = index
            else:
                dates = index
            if len(dates) == 0:
                return "N/A"
            return f"{str(min(dates))[:10]} ~ {str(max(dates))[:10]}"

        train_range = _get_time_range(train_index)
        val_range = _get_time_range(val_index)
        test_range = _get_time_range(test_index)

        best_epoch = evals_result.get("best_epoch", None)

        # 检查数据是否存在
        has_train = "train" in evals_result and len(evals_result["train"]) > 0
        has_valid = "valid" in evals_result and len(evals_result["valid"]) > 0
        has_train_score = "train_score" in evals_result and len(evals_result["train_score"]) > 0
        has_valid_score = "valid_score" in evals_result and len(evals_result["valid_score"]) > 0

        if not (has_train or has_valid):
            print("No loss data to visualize")
            return

        # 创建子图 - 2x2布局
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 第一个子图：训练损失（双y轴）
        if has_train:
            epochs = range(len(evals_result["train"]))

            # 左y轴：Total Loss 和 Expert Loss
            ax1.plot(epochs, evals_result["train"], label="Total Loss", color='blue', linewidth=2)
            if "train_expert" in evals_result and len(evals_result["train_expert"]) > 0:
                ax1.plot(epochs, evals_result["train_expert"], label="Expert Loss", color='green', linewidth=2)

            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Total & Expert Loss", color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.grid(True, alpha=0.3)

            # 右y轴：Router Loss
            if "train_router" in evals_result and len(evals_result["train_router"]) > 0:
                ax1_right = ax1.twinx()
                ax1_right.plot(epochs, evals_result["train_router"], label="Router Loss", color='orange', linewidth=2)
                ax1_right.set_ylabel("Router Loss", color='orange')
                ax1_right.tick_params(axis='y', labelcolor='orange')

            # 合并图例
            lines1, labels1 = ax1.get_legend_handles_labels()
            if "train_router" in evals_result and len(evals_result["train_router"]) > 0:
                lines2, labels2 = ax1_right.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            else:
                ax1.legend()

        ax1.set_title("Training Loss Components")

        # 第二个子图：验证损失（双y轴）
        if has_valid:
            epochs = range(len(evals_result["valid"]))

            # 左y轴：Total Loss 和 Expert Loss
            ax2.plot(epochs, evals_result["valid"], label="Total Loss", color='red', linewidth=2)
            if "valid_expert" in evals_result and len(evals_result["valid_expert"]) > 0:
                ax2.plot(epochs, evals_result["valid_expert"], label="Expert Loss", color='darkgreen', linewidth=2)

            # 标记最佳epoch（在Total Loss上）
            if best_epoch is not None and best_epoch < len(evals_result["valid"]):
                ax2.scatter(
                    best_epoch,
                    evals_result["valid"][best_epoch],
                    label="Best Epoch",
                    color='purple',
                    s=100,
                    zorder=10
                )

            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Total & Expert Loss", color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.grid(True, alpha=0.3)

            # 右y轴：Router Loss
            if "valid_router" in evals_result and len(evals_result["valid_router"]) > 0:
                ax2_right = ax2.twinx()
                ax2_right.plot(epochs, evals_result["valid_router"], label="Router Loss", color='darkorange', linewidth=2)
                ax2_right.set_ylabel("Router Loss", color='darkorange')
                ax2_right.tick_params(axis='y', labelcolor='darkorange')

            # 合并图例
            lines1, labels1 = ax2.get_legend_handles_labels()
            if "valid_router" in evals_result and len(evals_result["valid_router"]) > 0:
                lines2, labels2 = ax2_right.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            else:
                ax2.legend()

        ax2.set_title("Validation Loss Components")

        # 第三个子图：训练评分
        if has_train_score:
            epochs = range(len(evals_result["train_score"]))
            ax3.plot(epochs, evals_result["train_score"], label="Train Score (IC)", color='blue', linewidth=2)

        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Score (IC)")
        ax3.set_title("Training Score")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 第四个子图：验证评分
        if has_valid_score:
            epochs = range(len(evals_result["valid_score"]))
            ax4.plot(epochs, evals_result["valid_score"], label="Valid Score (IC)", color='red', linewidth=2)

            # 标记最佳epoch
            if best_epoch is not None and best_epoch < len(evals_result["valid_score"]):
                ax4.scatter(
                    best_epoch,
                    evals_result["valid_score"][best_epoch],
                    label="Best Epoch",
                    color='purple',
                    s=100,
                    zorder=10
                )

        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Score (IC)")
        ax4.set_title("Validation Score")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 调整子图间距
        plt.tight_layout()

        # 添加总标题
        if best_epoch is not None:
            title_info = f"MIGA Training Results - Best Epoch: {best_epoch + 1}"
            if has_valid:
                title_info += f" (Val Loss: {evals_result['valid'][best_epoch]:.4f})"
            if has_valid_score:
                title_info += f" (Val Score: {evals_result['valid_score'][best_epoch]:.4f})"
            fig.suptitle(title_info, fontsize=16, y=0.98)
        else:
            fig.suptitle("MIGA Training Results", fontsize=16, y=0.98)

        # 在图下方添加时间区间
        fig.text(0.5, 0.01, f"Train: {train_range}    Valid: {val_range}    Test: {test_range}", ha='center', fontsize=12)

        # 保存图片
        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(
                os.path.join(save_path, "miga_training_results.png"),
                dpi=300,
                bbox_inches='tight'
            )
            print(f"📊 MIGA训练结果图已保存到: {os.path.join(save_path, 'miga_training_results.png')}")

        plt.close()



class MIGAModel(nn.Module):
    """
    Mixture of Expert with Group Aggregation (MIGA)
    """
    def __init__(
        self,
        input_dim: int,
        num_groups: int = 4,
        num_experts_per_group: int = 4,
        num_heads: int = 8,
        top_k: int = 2,
        expert_output_dim: int = 1,
        router: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_groups = num_groups
        self.num_experts_per_group = num_experts_per_group
        self.num_heads = num_heads
        self.top_k = top_k
        self.expert_output_dim = expert_output_dim
        self.hidden_dim = self.num_groups * self.num_experts_per_group
        self.batch_norm = nn.BatchNorm1d(1)
        # Router
        self.router = router 

        
        # Expert groups
        self.expert_groups = nn.ModuleList([
            nn.ModuleList([
                Expert(self.hidden_dim, expert_output_dim)
                for _ in range(num_experts_per_group)
            ]) for _ in range(num_groups)
        ])
        
        # Inner group attention modules
        self.inner_group_attentions = nn.ModuleList([
            InnerGroupAttention(self.num_experts_per_group, num_heads)
            for _ in range(num_groups)
        ])
        
        # Final output projection
        self.output_projection = nn.Linear(expert_output_dim, 1)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [N, T, D] - N stocks, T time steps, D features
        Returns:
            predictions: [N, 1] - stock return predictions
            routing_weights: routing weights for analysis
        """
        batch_size = x.size(0)

        # Get router outputs - only hidden representations
        hidden_representations = self.router(x)  # [N, hidden_dim]

        # Select top-k dimensions/features from hidden representations
        top_k_values, top_k_indices = torch.topk(hidden_representations, self.top_k, dim=1)  # [N, k]

        # Create mask for selected features
        mask = torch.zeros_like(hidden_representations)
        mask.scatter_(1, top_k_indices, 1.0)

        # Apply mask and softmax to get routing weights
        masked_hidden = hidden_representations.masked_fill(mask == 0, float('-inf'))
        routing_weights_flat = F.softmax(masked_hidden, dim=1)  # [N, hidden_dim]

        # Reshape to match group structure (assuming hidden_dim = num_groups * num_experts_per_group)
        routing_weights = routing_weights_flat.view(batch_size, self.num_groups, self.num_experts_per_group)
        routing_weights = routing_weights.view(batch_size, -1)
        # Process each group
        all_group_outputs = []

        for group_idx in range(self.num_groups):
            # Get expert outputs for this group
            expert_outputs = []

            for expert_idx in range(self.num_experts_per_group):
                expert = self.expert_groups[group_idx][expert_idx]
                expert_output = expert(hidden_representations)  # [N, expert_output_dim]
                expert_output = expert_output.squeeze()
                assert expert_output.dim() == 1, f"expert_output.dim() = {expert_output.dim()}"
                expert_outputs.append(expert_output)

            # Stack expert outputs within the group for attention
            group_output = torch.stack(expert_outputs, dim=1)  # [N, num_experts_per_group, expert_output_dim]

            # Apply inner group attention
            aggregated_output = self.inner_group_attentions[group_idx](group_output)  # [N, num_experts_per_group, expert_output_dim]

            all_group_outputs.append(aggregated_output)

        # Concatenate all groups along the expert dimension
        all_outputs = torch.cat(all_group_outputs, dim=1)  # [N, num_groups * num_experts_per_group]

        # Apply routing weights for final weighted aggregation
        weighted_outputs = all_outputs * routing_weights  # [N, num_groups * num_experts_per_group, expert_output_dim]
        predictions = torch.sum(weighted_outputs, dim=1)  # [N]

        return {
            'predictions': predictions,
            'routing_weights': routing_weights,  # [N, num_groups, num_experts_per_group]
            'hidden_representations': hidden_representations,
            'top_k_indices': top_k_indices,  # [N, k] - 选中的专家索引
            'routing_weights_flat': routing_weights_flat  # [N, total_experts] - 展平的路由权重
        }

class MIGANewsModel(nn.Module):
    """
    Mixture of Expert with Group Aggregation (MIGA)
    """
    def __init__(
        self,
        input_dim: int,
        news_dim: int = 1024,
        num_groups: int = 4,
        num_experts_per_group: int = 4,
        num_heads: int = 8,
        top_k: int = 2,
        expert_output_dim: int = 1,
        router: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.news_dim = news_dim
        self.num_groups = num_groups
        self.num_experts_per_group = num_experts_per_group
        self.num_heads = num_heads
        self.top_k = top_k
        self.expert_output_dim = expert_output_dim
        self.hidden_dim = self.num_groups * self.num_experts_per_group
        self.batch_norm = nn.BatchNorm1d(1)
        # Router
        self.router = router 
        self.hidden_to_gate = nn.Linear(self.hidden_dim, self.num_groups * self.num_experts_per_group)

        # Expert groups
        self.expert_groups = nn.ModuleList([
            nn.ModuleList([
                Expert(self.hidden_dim, expert_output_dim)
                for _ in range(num_experts_per_group)
            ]) for _ in range(num_groups)
        ])
        
        # Inner group attention modules
        self.inner_group_attentions = nn.ModuleList([
            InnerGroupAttention(self.num_experts_per_group, num_heads)
            for _ in range(num_groups)
        ])
        
        # Final output projection
        self.output_projection = nn.Linear(expert_output_dim, 1)
        
    def forward(self, price_feature: torch.Tensor, 
                news_feature: torch.Tensor, 
                news_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            price_feature: [N, T, D] - N stocks, T time steps, D features
            news_feature: [N, T, D] - N stocks, T time steps, D features
            news_mask: [N, T] - N stocks, T time steps
        Returns:
            predictions: [N, 1] - stock return predictions
            routing_weights: routing weights for analysis
        """
        batch_size = price_feature.size(0)

        # Get router outputs - only hidden representations
        hidden_representations = self.router(price_feature, news_feature, news_mask)  # [N, hidden_dim]

        # Select top-k dimensions/features from hidden representations
        top_k_values, top_k_indices = torch.topk(hidden_representations, self.top_k, dim=1)  # [N, k]

        # Create mask for selected features
        mask = torch.zeros_like(hidden_representations)
        mask.scatter_(1, top_k_indices, 1.0)

        # Apply mask and softmax to get routing weights
        masked_hidden = hidden_representations.masked_fill(mask == 0, float('-inf'))
        routing_weights_flat = F.softmax(masked_hidden, dim=1)  # [N, hidden_dim]

        # Reshape to match group structure (assuming hidden_dim = num_groups * num_experts_per_group)
        routing_weights = routing_weights_flat.view(batch_size, self.num_groups, self.num_experts_per_group)
        routing_weights = routing_weights.view(batch_size, -1)
        # Process each group
        all_group_outputs = []

        for group_idx in range(self.num_groups):
            # Get expert outputs for this group
            expert_outputs = []

            for expert_idx in range(self.num_experts_per_group):
                expert = self.expert_groups[group_idx][expert_idx]
                expert_output = expert(hidden_representations)  # [N, expert_output_dim]
                expert_output = expert_output.squeeze()
                assert expert_output.dim() == 1, f"expert_output.dim() = {expert_output.dim()}"
                expert_outputs.append(expert_output)

            # Stack expert outputs within the group for attention
            group_output = torch.stack(expert_outputs, dim=1)  # [N, num_experts_per_group, expert_output_dim]

            # Apply inner group attention
            aggregated_output = self.inner_group_attentions[group_idx](group_output)  # [N, num_experts_per_group, expert_output_dim]

            all_group_outputs.append(aggregated_output)

        # Concatenate all groups along the expert dimension
        all_outputs = torch.cat(all_group_outputs, dim=1)  # [N, num_groups * num_experts_per_group]

        # Apply routing weights for final weighted aggregation
        weighted_outputs = all_outputs * routing_weights  # [N, num_groups * num_experts_per_group, expert_output_dim]
        predictions = torch.sum(weighted_outputs, dim=1)  # [N]

        return {
            'predictions': predictions,
            'routing_weights': routing_weights,  # [N, num_groups, num_experts_per_group]
            'hidden_representations': hidden_representations,
            'top_k_indices': top_k_indices,  # [N, k] - 选中的专家索引
            'routing_weights_flat': routing_weights_flat  # [N, total_experts] - 展平的路由权重
        }


class CrossAttentionAggregator(nn.Module):
    def __init__(self, input_dim_ts,
                 input_dim_news,
                 hidden_dim, 
                 num_heads: int = 4):
        super().__init__()
        self.ts_proj = nn.Linear(input_dim_ts, hidden_dim)
        self.news_proj = nn.Linear(input_dim_news, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # Time attention pooling
        self.time_score = nn.Linear(hidden_dim, 1)

    def forward(self, ts_input, news_input, news_mask=None):
        # Step 1: Projection
        ts = self.ts_proj(ts_input)       # (N, T, H)
        news = self.news_proj(news_input) # (N, T, H)

        # Step 2: Cross Attention (ts attends to news)
        attn_out, _ = self.attn(query=ts, key=news, value=news, key_padding_mask=news_mask)

        # Step 3: Pooling over time
        scores = self.time_score(attn_out)        # (N, T, 1)
        weights = torch.softmax(scores, dim=1)    # (N, T, 1)
        final = (attn_out * weights).sum(dim=1)   # (N, H)

        return final  # 每只股票一个向量


class PriceNewsCrossAttn(nn.Module):
    def __init__(self, d_price, d_news, d_model, n_heads, d_gru, dropout=0.1):
        super().__init__()
        self.price_proj = nn.Linear(d_price, d_model) 
        self.news_proj = nn.Linear(d_news, d_model) 
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)

        self.resid_dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_gru, batch_first=True)

    def forward(self, price, news, mask):
        """
        price: [N, T, D_p]
        news:  [N, T, D_n]
        mask:  [N, T]
        return: [N, T, d_model]
        """
        # 1. 投影
        price_proj = self.price_proj(price)  # [N, T, d_model]
        news_proj = self.news_proj(news)    # [N, T, d_model]

        # 2. Cross-Attention: q=price, k/v=news
        # nn.MultiheadAttention expects input as [batch, seq_len, embed_dim]
        attn_out, _ = self.cross_attn(query=price_proj,
                                      key=news_proj,
                                      value=news_proj,
                                      key_padding_mask=mask)

        # 3. Residual + Dropout
        fused = self.ln(price_proj + self.resid_dropout(attn_out))  # [N, T, d_model]

        # 4. GRU
        out, _ = self.gru(fused)  # [N, T, d_model]

        return out[:, -1, :]


class PriceNewsRouter(nn.Module):
    """
    
    """
    def __init__(
        self,
        price_dim:int=158,
        news_dim:int=1024,
        d_model:int=128,
        n_heads:int=4,
        d_gru:int=64,
        dropout:float=0.0,
    ):
        super().__init__()
        self.price_dim = price_dim
        self.news_dim = news_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.price_cross_attn = PriceNewsCrossAttn(price_dim, news_dim, d_model, n_heads, d_gru, dropout)

    def forward(self, price, news, mask):
        """
        price: [N, T, D_p]
        news:  [N, T, D_n]
        mask:  [N, T]
        return: [N, T, d_model]
        """
        return self.price_cross_attn(price, news, mask)
        
        

class Router(nn.Module):
    """
    Router for cross-sectional encoding and generating routing weights.
    目前只用了 GRU
    """
    def __init__(self, 
                input_dim: int, 
                hidden_dim: int,
                num_groups: int, 
                num_experts_per_group: int,
                model = 'gru'):
        super().__init__()
        self.input_dim = input_dim
        self.num_groups = num_groups
        self.num_experts_per_group = num_experts_per_group
        self.hidden_dim = hidden_dim
        self.fc_dim = self.num_experts_per_group * self.num_groups       
        self.fc = nn.Linear(self.hidden_dim, self.fc_dim)
        self.activation = nn.LeakyReLU()
        
        if  model == 'gru':
            self.model = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=0.0,
            )
        elif model == 'lstm':
            self.model = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=0.0,
            )
        else:
            raise NotImplementedError(f"model {model} is not supported!")
                
        # Cross-sectional encoder
        # self.encoder = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU()
        # )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, T, D] - N stocks, T time steps, D features
        Returns:
            hidden: [N, hidden_dim] - encoded representations
            routing_weights: [N, num_groups, num_experts_per_group] - routing weights
        """
        # Use the last time step for cross-sectional encoding
        hidden, _ = self.model(x)
        hidden = hidden[:,-1,:]
        hidden = self.activation(hidden)
        
        return self.fc(hidden)
             


class Expert(nn.Module):
    """
    Individual expert (linear layer as mentioned in paper)
    """
    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, input_dim] - input features
        Returns:
            output: [N, output_dim] - expert predictions
        """
        return self.linear(x)

class InnerGroupAttention(nn.Module):
    """
    Multi-head self-attention for aggregating experts within a group
    """
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.w_q = nn.Linear(hidden_dim, hidden_dim)
        self.w_k = nn.Linear(hidden_dim, hidden_dim)
        self.w_v = nn.Linear(hidden_dim, hidden_dim)
        self.w_o = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, num_experts] - concatenated expert outputs
        Returns:
            output: [N, num_experts] - attention-aggregated outputs
        """
        batch_size, num_experts = x.size()
        
        # Generate Q, K, V
        Q = self.w_q(x)  # [N, num_experts]
        K = self.w_k(x)  # [N, num_experts]
        V = self.w_v(x)  # [N, num_experts]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attended_values = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, num_experts
        )
        
        # Output projection
        output = self.w_o(attended_values)
        
        return output

