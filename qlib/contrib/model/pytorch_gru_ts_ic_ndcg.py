# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# add daily batch sampler
# add NDCG loss function

from __future__ import division
from __future__ import print_function

from collections import defaultdict
import os 
import numpy as np
from tqdm import tqdm
import pandas as pd
import copy
from ...utils import get_or_create_path
from ...log import get_module_logger
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Sampler

from .pytorch_utils import count_parameters
from ...model.base import Model
from ...data.dataset.handler import DataHandlerLP
from ...model.utils import ConcatDataset
from ...data.dataset.weight import Reweighter
from ..loss.ndcg import (compute_lambda_gradients, calculate_ndcg_optimized, ranknet_cross_entropy_loss,
                         compute_delta_ndcg)
from ..loss.loss import ic_loss, rankic_loss, topk_ic_loss, topk_rankic_loss
from qlib.utils.color import *
from qlib.utils.hxy_utils import (
    compute_grad_norm, compute_layerwise_grad_norm, 
    apply_mask_preserve_norm, scale_preserve_sign_torch,
    process_ohlc,
)
from colorama import Fore, Style, init
import matplotlib.pyplot as plt
import logging
import sys

init(autoreset=True)

class DailyBatchSampler(Sampler):
    """
        HXY 修正
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
        return len(self.order)


class GRUNDCG(Model):
    """GRU Model

    Parameters
    ----------
    d_feat : int
        input dimension for each time step
    metric: str
        the evaluation metric used in early stop
    optimizer : str
        optimizer name
    GPU : str
        the GPU ID(s) used for training
    """

    def __init__(
        self,
        d_feat=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        n_epochs=200,
        lr=0.001,
        metric="",
        batch_size=2000,
        early_stop=20,
        step_len=20,
        loss="mse",
        optimizer="adam",
        n_jobs=10,
        GPU=0,
        seed=None,
        sigma=1.0,
        n_layer=10,
        linear_ndcg=False,
        debug=False,
        save_path=None,
        weight=0.7,
        combine_type='mult',
        ohlc=False,
        **kwargs,
    ):
        # Set logger.
        self.logger = get_module_logger("GRU")
        self.logger.setLevel(logging.DEBUG)
        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.DEBUG)
            self.logger.addHandler(handler)
        self.logger.info("GRU pytorch version...")

        # set hyper-parameters.
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        assert self.loss == "ic", "loss must be ic"
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.n_jobs = n_jobs
        self.seed = seed
        self.sigma = sigma
        self.n_layer = n_layer  
        self.linear_ndcg = linear_ndcg
        self.debug = debug
        self.save_path = save_path
        self.combine_type = combine_type
        self.weight = weight
        self.ohlc = ohlc
        self.step_len = step_len
        if self.ohlc:
            self.logger.info(Fore.RED + "使用OHLC数据, 默认为前 6 个特征" + Style.RESET_ALL)
        self.logger.info(Fore.RED + "use GPU: %s" % str(self.device) + Style.RESET_ALL)
        self.logger.info(Fore.RED + ("Debug Mode" if self.debug else "RUN Mode") + Style.RESET_ALL)
        self.logger.info(
            "GRU parameters setting:"
            "\nd_feat : {}"
            "\nhidden_size : {}"
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
            "\nsigma : {}"
            "\nn_layer : {}"
            "\nlinear_ndcg : {}"
            "\ncombine_type : {}"
            "\nweight : {}"
            "\nstep_len : {}"
            "\nsave_path : {}".format(
                d_feat,
                hidden_size,
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
                seed,
                sigma,
                n_layer,
                linear_ndcg,
                combine_type,
                weight,
                step_len,
                save_path,
            )
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.GRU_model = GRUModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        self.logger.info("model:\n{:}".format(self.GRU_model))
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.GRU_model)))

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.GRU_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.GRU_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.GRU_model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label, weight):
        loss = weight * (pred - label) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label, weight=None):
        mask = ~torch.isnan(label)

        if weight is None:
            weight = torch.ones_like(label)

        if self.loss == "mse":
            return self.mse(pred[mask], label[mask], weight[mask])
        elif self.loss == "ic":
            return ic_loss(pred[mask], label[mask])
        elif self.loss == 'rankic':
            return rankic_loss(pred[mask], label[mask])
        raise ValueError("unknown loss `%s`" % self.loss)

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
        self.GRU_model.train()
        # Debug模式下记录每个batch的梯度信息
        result = defaultdict(lambda : np.nan)
        if self.debug:
            epoch_grad_norms = []
            epoch_grad_norms_layer = []

        for i, (data, weight) in tqdm(enumerate(data_loader)):
            if i <= self.step_len:
                # warm up
                continue
                
            data.squeeze_(0) # 去除横截面 dim
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)
            if self.ohlc:
                feature = process_ohlc(feature)
            pred = self.GRU_model(feature.float())
            # 这里使用NDCG @k来计算损失
            pred.requires_grad_(True)
            # 清零梯度
            self.train_optimizer.zero_grad()
            if self.loss == "ic":
                # 路线 2
                with torch.no_grad():
                    # 计算每个样本的ndcg变化
                    lambda_grads = compute_delta_ndcg(pred.detach(), label.detach(), self.n_layer, sigma=self.sigma, linear=self.linear_ndcg)
                    lambda_grads = lambda_grads / lambda_grads.sum()
                loss = self.loss_fn(pred, label)
                grad = torch.autograd.grad(loss, pred, create_graph=False)[0]
                if self.combine_type == 'mult': # 相乘形式
                    # lambda_grads = apply_mask_preserve_norm(grad, lambda_grads, method = 'minmax')
                    lambda_grads = lambda_grads * grad * 1000
                elif self.combine_type == 'null':
                    lambda_grads = grad
                elif self.combine_type == 'add': # 相加形式
                    # 计算 lambda 梯度
                    lambda_grads = compute_lambda_gradients(label.detach(), pred.detach(), self.n_layer, self.sigma, self.linear_ndcg)
                    # 检查梯度是否有效
                    check_grad = torch.sum(lambda_grads).item()
                    if check_grad == float('inf') or np.isnan(check_grad):
                        print("Warning: Invalid lambda gradients detected")
                        print("lambda_grads_sum:", check_grad)
                        lambda_grads = torch.zeros_like(lambda_grads)
                    # 加上ic 的梯度
                    lambda_grads = scale_preserve_sign_torch(lambda_grads)
                    grad = scale_preserve_sign_torch(grad)
                    lambda_grads = (1-self.weight) * lambda_grads + grad * self.weight
                else:
                    raise ValueError(f"Unknown combine_type: {self.combine_type}")
            else:
                raise ValueError(f"Unknown loss: {self.loss}")
            pred.backward(lambda_grads)                
            torch.nn.utils.clip_grad_norm_(self.GRU_model.parameters(), 3.0) 
            # 手动更新梯度
            # with torch.no_grad():
            #     lr = self.train_optimizer.param_groups[0]['lr']
            #     for p in self.GRU_model.parameters():
            #         if p.grad is not None:
            #             p.data -= lr * p.grad
            # 优化器更新梯度
            self.train_optimizer.step()
            # 更新完后记录下梯度
            if self.debug:
                grad_norm = compute_grad_norm(self.GRU_model)
                grad_norm_layer = compute_layerwise_grad_norm(self.GRU_model)
                epoch_grad_norms.append(grad_norm)
                epoch_grad_norms_layer.append(grad_norm_layer)
                # # 打印梯度信息
                # print(f"Batch Grad Norm: {grad_norm:.6f}")
                # # 处理每层梯度范数（compute_layerwise_grad_norm返回的是字典，每个键对应一个参数的梯度范数列表）
                # layer_info_parts = []
                # for name, norms in grad_norm_layer.items():
                #     if isinstance(norms, list) and len(norms) > 0:
                #         avg_norm = np.mean(norms)
                #         layer_info_parts.append(f"{name}: {avg_norm:.6f}")
                #     else:
                #         layer_info_parts.append(f"{name}: {norms:.6f}")
                # layer_info = ", ".join(layer_info_parts)
                # print(f"Layer Grad Norms: {layer_info}")

                # Debug模式下记录梯度信息
        if self.debug:
            # 打印epoch级别的梯度统计
            avg_grad_norm = np.mean(epoch_grad_norms)
            print(f"Epoch Avg Grad Norm: {avg_grad_norm:.6f}")

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
                    print(f"Epoch Avg {layer_name} Grad Norm: {avg_layer_norm:.6f}")

        return result
    
    def test_epoch(self, data_loader):
        self.GRU_model.eval()

        scores = []
        losses = []
        ic_scores = []
        rankic_scores = []
        topk_ic_scores = []
        topk_rankic_scores = []

        for data, weight in data_loader:
            data.squeeze_(0) # 去除横截面 dim
            feature = data[:, :, 0:-1].to(self.device)
            # feature[torch.isnan(feature)] = 0
            label = data[:, -1, -1].to(self.device)
            if self.ohlc:
                feature = process_ohlc(feature)
            with torch.no_grad():
                pred = self.GRU_model(feature.float())
                # 计算RankNet交叉熵损失（仅用于观察）
                loss = ranknet_cross_entropy_loss(pred, label, sigma=self.sigma)
                # 计算NDCG
                score = calculate_ndcg_optimized(label, pred, self.n_layer, self.linear_ndcg)
                losses.append(loss.item())
                # 计算 IC
                ic_score = self.metric_fn(pred, label, "ic")
                rankic_score = self.metric_fn(pred, label, "rankic")
                topk_ic_score = self.metric_fn(pred, label, "topk_ic", topk=self.n_layer)
                topk_rankic_score = self.metric_fn(pred, label, "topk_rankic", topk=self.n_layer)
                # append scores
                scores.append(score.item())
                ic_scores.append(ic_score)
                rankic_scores.append(rankic_score)
                topk_ic_scores.append(topk_ic_score)
                topk_rankic_scores.append(topk_rankic_score)

        result = defaultdict(lambda : np.nan)
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

        if reweighter is None:
            wl_train = np.ones(len(dl_train))
            wl_valid = np.ones(len(dl_valid))
        elif isinstance(reweighter, Reweighter):
            wl_train = reweighter.reweight(dl_train)
            wl_valid = reweighter.reweight(dl_valid)
        else:
            raise ValueError("Unsupported reweighter type.")

        train_loader = DataLoader(
            ConcatDataset(dl_train, wl_train),
            batch_sampler=sampler_train,
            num_workers=self.n_jobs,
        )
        valid_loader = DataLoader(
            ConcatDataset(dl_valid, wl_valid),
            batch_sampler=sampler_valid,
            num_workers=self.n_jobs,
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

        for step in tqdm(range(self.n_epochs)):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            result = self.train_epoch(train_loader) 
            # evals_result["train"].append(result["train"]) 
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
            evals_result["valid"].append(result["rankic"])
            evals_result["valid_score"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.GRU_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.GRU_model.load_state_dict(best_param)
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
        # 这里不能用dailysampler，否则 index 对不上
        test_loader = DataLoader(dl_test, batch_size=self.batch_size, num_workers=self.n_jobs)
        self.GRU_model.eval()
        preds = []

        for data in test_loader:
            data.squeeze_(0) # 去除横截面 dim
            feature = data[:, :, 0:-1].to(self.device)
            if self.ohlc:
                feature = process_ohlc(feature)
            with torch.no_grad():
                pred = self.GRU_model(feature.float()).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=self.test_index)

    def visualize_evals_result(self, evals_result,):
        """
        可视化训练和验证损失曲线。
        分别绘制loss和score的图表，分别监测训练和验证指标。
        新增：在图下方显示训练集、验证集、测试集的时间区间。
        """
        self.logger.info("visualizing evals result...")
        self.logger.info(f"save_path: {self.save_path}")
        def _get_time_range(index):
            """提取时间区间"""
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

        best_epoch = evals_result.get("best_epoch", None)
        has_train = "train" in evals_result.keys() and len(evals_result["train"]) > 0
        has_valid = "valid" in evals_result.keys() and len(evals_result["valid"]) > 0
        has_train_score = "train_score" in evals_result.keys() and len(evals_result["train_score"]) > 0
        has_valid_score = "valid_score" in evals_result.keys() and len(evals_result["valid_score"]) > 0
        
        if has_train or has_valid or has_train_score or has_valid_score or best_epoch is not None:
            # 创建子图
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))

            # 第一个子图：Loss曲线
            if has_train:
                ax1.plot(evals_result["train"], label="Train Loss", color='blue', linewidth=2)
            if has_valid:
                ax1.plot(evals_result["valid"], label="Valid Loss", color='red', linewidth=2)

            # 标记最佳epoch（基于验证损失）
            if best_epoch is not None and has_valid:
                ax1.scatter(
                    best_epoch,
                    evals_result["valid"][best_epoch],
                    label="Best Epoch",
                    color='green',
                    s=100,
                    zorder=10
                )

            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.set_title(f"{self.metric} Loss - Training vs Validation")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 第二个子图：Score曲线
            if has_train_score:
                ax2.plot(evals_result["train_score"], label="Train Score", color='blue', linewidth=2)
            if has_valid_score:
                ax2.plot(evals_result["valid_score"], label="Valid Score", color='red', linewidth=2)

            # 标记最佳epoch（基于验证分数）
            if best_epoch is not None and has_valid_score:
                ax2.scatter(
                    best_epoch,
                    evals_result["valid_score"][best_epoch],
                    label="Best Epoch",
                    color='green',
                    s=100,
                    zorder=10
                )

            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Score")
            ax2.set_title(f"{self.metric} Score - Training vs Validation")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 调整子图间距，为时间区间信息留出空间
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)

            # 添加时间区间信息
            train_range = _get_time_range(self.train_index)
            val_range = _get_time_range(self.val_index)

            time_info = f"Train: {train_range}  |  Valid: {val_range}"
            fig.text(0.5, 0.02, time_info, ha='center', va='bottom', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))

            # 添加总标题
            if best_epoch is not None:
                title_info = f"Best Epoch: {best_epoch}"
                if has_valid:
                    title_info += f" (Loss: {evals_result['valid'][best_epoch]:.4f})"
                if has_valid_score:
                    title_info += f" (Score: {evals_result['valid_score'][best_epoch]:.4f})"
                fig.suptitle(title_info, fontsize=14, y=0.95)
            
            if self.save_path is not None and self.save_path != "":
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)
                plt.savefig(
                    os.path.join(self.save_path, "evals_result.png"),
                    dpi=300,
                    bbox_inches='tight'
                )
            plt.close()



class GRUModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 1)

        self.d_feat = d_feat

    def forward(self, x):
        x = x.squeeze()  # remove the time dimension
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze()
