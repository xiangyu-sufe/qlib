# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function
from collections import defaultdict
import warnings

import numpy as np
from tqdm import tqdm
import pandas as pd
import copy
from ...utils import get_or_create_path
from ...log import get_module_logger

import torch
import torch.nn as nn
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
from qlib.utils.hxy_utils import compute_grad_norm, compute_layerwise_grad_norm

from colorama import Fore, Style, init

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


class GRU(Model):
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
        lambda_reg=0.1,
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
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.n_jobs = n_jobs
        self.seed = seed
        self.debug = debug
        if self.loss == "ranking":
            assert lambda_reg is not None, "lambda must be provided for ranking loss"
            self.lambda_reg = lambda_reg        

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
            "\ndebug : {}".format(
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
                debug,
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
        elif self.loss == "ranking":
            return ranking_loss(pred[mask], label[mask], self.lambda_reg)

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
            mse_loss_list = []
            pairwise_loss_list = []
            tot_loss_list = []

        for data, weight in data_loader:
            data.squeeze_(0) # 去除横截面 dim
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device).float()

            pred = self.GRU_model(feature.float())
            loss = self.loss_fn(pred, label, weight.to(self.device))

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.GRU_model.parameters(), 3.0)
            self.train_optimizer.step()

            if self.debug:
                # 计算MSE 和 pairwise loss 相对大小
                with torch.no_grad():
                    mse_loss = mse(pred, label).item()
                    pr_loss = pairwise_loss(pred, label).item()
                    mse_loss_list.append(mse_loss)
                    pairwise_loss_list.append(pr_loss)
                    tot_loss_list.append(loss.item())
                # 计算梯度范数
                grad_norm = compute_grad_norm(self.GRU_model)
                grad_norm_layer = compute_layerwise_grad_norm(self.GRU_model)
                epoch_grad_norms.append(grad_norm)
                epoch_grad_norms_layer.append(grad_norm_layer)

                # Debug模式下记录梯度信息
        if self.debug:
            # 打印epoch级别的梯度统计
            # f"{Fore.GREEN}"
            print(f"{Fore.RED} MSE Loss: {np.mean(mse_loss_list):.6f}, Pairwise Loss: {np.mean(pairwise_loss_list):.6f}{Style.RESET_ALL}")
            print(f"{Fore.RED} Total loss: {np.mean(tot_loss_list):.6f}{Style.RESET_ALL}")
            avg_grad_norm = np.mean(epoch_grad_norms)
            print(f"{Fore.RED}Epoch Avg Grad Norm: {avg_grad_norm:.6f}{Style.RESET_ALL}")

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
            label = data[:, -1, -1].to(self.device).float()

            with torch.no_grad():
                pred = self.GRU_model(feature.float())
                # 计算RankNet交叉熵损失（仅用于观察）
                loss = self.loss_fn(pred, label, )
                losses.append(loss.item())
                score = self.metric_fn(pred, label, name = 'loss')
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
            result = self.test_epoch(valid_loader)
            self.logger.info(
                f"{Fore.GREEN}"
                f"valid loss: {result['loss']:.6f}, valid score: {result['score']:.6f}\n"
                f"ic: {result['ic']:.6f}, rankic: {result['rankic']:.6f}, "
                f"topk_ic: {result['topk_ic']:.6f}, topk_rankic: {result['topk_rankic']:.6f}"
                f"{Style.RESET_ALL}"
            )
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
        test_loader = DataLoader(dl_test, batch_size=self.batch_size, num_workers=self.n_jobs)
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
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze()
