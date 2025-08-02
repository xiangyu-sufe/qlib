# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# add daily batch sampler
# add NDCG loss function

from __future__ import division
from __future__ import print_function

from collections import defaultdict 
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
from ..loss.ndcg import compute_lambda_gradients, calculate_ndcg_optimized, ranknet_cross_entropy_loss
from ..loss.loss import ic_loss, rankic_loss, topk_ic_loss, topk_rankic_loss
from qlib.utils.color import *
from qlib.utils.hxy_utils import compute_grad_norm, compute_layerwise_grad_norm
from colorama import Fore, Style, init

init(autoreset=True)
class DailyBatchSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        # calculate number of samples in each batch
        self.daily_count = (
            pd.Series(index=self.data_source.get_index()).groupby("datetime", group_keys=False).size().values
        )
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)  # calculate begin index of each batch
        self.daily_index[0] = 0

    def __iter__(self):
        for idx, count in zip(self.daily_index, self.daily_count):
            yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.data_source)


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
        loss="mse",
        optimizer="adam",
        n_jobs=10,
        GPU=0,
        seed=None,
        sigma=1.0,
        n_layer=10,
        linear_ndcg=False,
        debug=False,
        **kwargs,
    ):
        # Set logger.
        self.logger = get_module_logger("GRU")
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
        assert self.loss == "cross_entropy", "loss must be cross_entropy"
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.n_jobs = n_jobs
        self.seed = seed
        self.sigma = sigma
        self.n_layer = n_layer  
        self.linear_ndcg = linear_ndcg
        self.debug = debug
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
            "\nlinear_ndcg : {}".format(
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

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label, name, topk=None):
        mask = torch.isfinite(label)

        if name in ("", "loss", "ic", "rankic", "topk_ic", "topk_rankic"):
            if name == "ic":
                return -ic_loss(pred[mask], label[mask])
            elif name == "rankic":
                return -rankic_loss(pred[mask], label[mask])
            elif name == "topk_ic":
                if topk is None:
                    warnings.warn("topk must be specified for topk_ic metric, return nan")
                    return torch.nan
                return -topk_ic_loss(pred[mask], label[mask], k=topk)
            elif name == "topk_rankic":
                if topk is None:
                    warnings.warn("topk must be specified for topk_ic metric, return nan")
                    return torch.nan
                return -topk_rankic_loss(pred[mask], label[mask], k=topk)
            elif name == "loss":
                return -self.loss_fn(pred[mask], label[mask])

        raise ValueError("unknown metric `%s`" % name)

    def train_epoch(self, data_loader):
        self.GRU_model.train()
        # Debug模式下记录每个batch的梯度信息
        if self.debug:
            epoch_grad_norms = []
            epoch_grad_norms_layer = []

        for data, weight in data_loader:
            data.squeeze_(0) # 去除横截面 dim
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)
            pred = self.GRU_model(feature.float())
            # 这里使用NDCG @k来计算损失
            pred.requires_grad_(True)
            # 清零梯度
            self.train_optimizer.zero_grad()
            with torch.no_grad():
                # 计算NDCG
                lambda_grads = compute_lambda_gradients(label, pred.detach(), self.n_layer, self.sigma, self.linear_ndcg)
                # 检查梯度是否有效
                check_grad = torch.sum(lambda_grads).item()
                if check_grad == float('inf') or np.isnan(check_grad):
                    print("Warning: Invalid lambda gradients detected")
                    print("lambda_grads_sum:", check_grad)
                    lambda_grads = torch.zeros_like(lambda_grads)
            pred.backward(lambda_grads)                
            torch.nn.utils.clip_grad_norm_(self.GRU_model.parameters(), 3.0)
            # 手动更新梯度
            with torch.no_grad():
                lr = self.train_optimizer.param_groups[0]['lr']
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.data -= lr * p.grad
            # 优化器更新梯度
            # self.train_optimizer.step()
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
                topk_ic_score = self.metric_fn(pred, label, "topk_ic", topk=5)
                topk_rankic_score = self.metric_fn(pred, label, "topk_rankic", topk=5)
                # append scores
                scores.append(score.item())
                ic_scores.append(ic_score.item())
                rankic_scores.append(rankic_score.item())
                topk_ic_scores.append(topk_ic_score.item())
                topk_rankic_scores.append(topk_rankic_score.item())

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
            sampler=sampler_train,
            num_workers=self.n_jobs,
            drop_last=True,
        )
        valid_loader = DataLoader(
            ConcatDataset(dl_valid, wl_valid),
            sampler=sampler_valid,
            num_workers=self.n_jobs,
            drop_last=True,
        )

        save_path = get_or_create_path(save_path)

        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in tqdm(range(self.n_epochs)):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(train_loader)
            self.logger.info("evaluating...")
            # train_loss, train_score = self.test_epoch(train_loader)
            result = self.test_epoch(valid_loader)
            self.logger.info(
                f"{Fore.GREEN}"
                f"valid loss: {result['loss']:.6f}, valid score: {result['score']:.6f}\n"
                f"ic: {result['ic']:.6f}, rankic: {result['rankic']:.6f}, "
                f"topk_ic: {result['topk_ic']:.6f}, topk_rankic: {result['topk_rankic']:.6f}"
                f"{Style.RESET_ALL}"
            )
            # evals_result["train"].append(train_score)
            val_score = result["score"]
            evals_result["valid"].append(val_score)

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

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        sampler_test = DailyBatchSampler(dl_test)
        test_loader = DataLoader(dl_test, sampler=sampler_test, num_workers=self.n_jobs)
        self.GRU_model.eval()
        preds = []

        for data in test_loader:
            feature = data[:, :, 0:-1].to(self.device)

            with torch.no_grad():
                pred = self.GRU_model(feature.float()).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=dl_test.get_index())


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
