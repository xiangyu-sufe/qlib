# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# add daily batch sampler
# add NDCG loss function

# train loss  
# train score  ic 
# val loss 
# val score  ic 

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
from torch.utils.data import BatchSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .pytorch_utils import count_parameters
from ...model.base import Model
from ...data.dataset.handler import DataHandlerLP
from ...model.utils import ConcatDataset
from ...data.dataset.weight import Reweighter
from ..loss.ndcg import (compute_lambda_gradients, calculate_ndcg_optimized, ranknet_cross_entropy_loss,
                         compute_delta_ndcg)
from ..loss.loss import ic_loss, rankic_loss, topk_ic_loss, topk_rankic_loss, topk_return
from qlib.utils.color import *
from qlib.utils.hxy_utils import (
    compute_grad_norm, compute_layerwise_grad_norm, process_ohlc_cuda, process_minute_cuda,
    apply_mask_preserve_norm, process_ohlc_batchnorm, scale_preserve_sign_torch,
    process_ohlc_minmax, process_ohlc_inf_nan_fill0_cuda, process_ohlc_batchwinsor,
    visualize_evals_result_general, 
)
from qlib.utils.timing import timing
from colorama import Fore, Style, init
import matplotlib.pyplot as plt
import logging
import sys

init(autoreset=True)

class DailyBatchSampler(BatchSampler):
    """
    Yield all rows of the same trading day as one batch,
    independent of the index sort order.
    """
    def __init__(self, data_source):
        self.data_source = data_source
        # 把 datetime -> 行号数组 建立映射
        original_index = self.data_source.get_index()
        dts = original_index.get_level_values("datetime")
        self.groups = defaultdict(list)
        for pos, dt in enumerate(dts):
            self.groups[dt].append(pos)
        # 交易日按时间排序，保证训练顺序一致
        self.order = sorted(self.groups.keys())

        # 按训练顺序把行号拼成一个完整的新索引顺序
        new_pos_list = []
        for dt in self.order:
            new_pos_list.extend(self.groups[dt])
        # 根据行号列表重新排序原 MultiIndex
        self.new_index = original_index[new_pos_list]

    @property
    def reordered_index(self):
        return self.new_index
        
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
        batch_size=2000,
        early_stop=20,
        step_len=20,
        loss="mse",
        metric="",
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
        minute=True,
        display_list=['loss', 'ic', 'rankic', 'ndcg', 'topk_return'],
        id=0,
        **kwargs,
    ):
        # Set logger.
        assert metric in display_list, "metric must be in display_list"
        self.logger = get_module_logger("GRU")
        self.logger.setLevel(logging.DEBUG)
        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.DEBUG)
            self.logger.addHandler(handler)
        self.logger.info("GRU pytorch version...")
        self.logger.addHandler(logging.FileHandler(f"{save_path}/train.log"))
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
        self.minute = minute
        self.step_len = step_len
        self.display_list = display_list
        if self.ohlc:
            self.logger.info(Fore.RED + "使用OHLC数据, 默认为前 6 个特征" + Style.RESET_ALL)
        if self.minute:
            self.logger.info(Fore.RED + "使用分钟数据" + Style.RESET_ALL)
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
        
        self.lr_scheduler = ReduceLROnPlateau(
            optimizer=self.train_optimizer,
            mode='max',
            factor = 0.2,
            patience = 3, 
            min_lr = 1e-5           # 学习率下限
        )
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

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label, name, topk=None):
        mask = torch.isfinite(label) 

        if name in ("", "loss", "ic", "rankic", "topk_ic", "topk_rankic", "ndcg", "topk_return"):
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
            elif name == "ndcg":
                return calculate_ndcg_optimized(label[mask], pred[mask], self.n_layer, linear=self.linear_ndcg).item()
            elif name == "topk_return":
                return topk_return(pred[mask], label[mask], k=self.n_layer).item()

        raise ValueError("unknown metric `%s`" % name)
    
    
    @timing
    def train_epoch(self, data_loader):
        self.GRU_model.train()
        # Debug模式下记录每个batch的梯度信息
        result = defaultdict(list)
        result_agg = defaultdict(lambda : np.nan)
        if self.debug:
            epoch_grad_norms = []
            epoch_grad_norms_layer = []

        for i, (data, weight) in enumerate(data_loader):
            if i <= self.step_len:
                # warm up
                continue
                
            data.squeeze_(0) # 去除横截面 dim
            feature = data[:, :, 0:self.d_feat].to(self.device)
            label = data[:, -1, self.d_feat].to(self.device)
            if self.ohlc:
                # 这里是分钟数据没有归一化
                # 使用 ohlc 数据
                # 先时序归一化+ winsor + batchnorm + fill0
                if self.minute:
                    feature = process_minute_cuda(feature)
                else:
                    feature = process_ohlc_cuda(feature)
                feature = process_ohlc_batchwinsor(feature)
                feature = process_ohlc_batchnorm(feature) 
                feature = process_ohlc_inf_nan_fill0_cuda(feature)
            #     # 或者 对 volume winsor 后  minmax 归一化 + ffill + bfill
            # if self.ohlc:
            #     feature = process_ohlc_cuda(feature)
            #     feature[:, :, :6] = process_ohlc_batchwinsor(feature[:, :, :6])
            #     feature[:, :, :6] = process_ohlc_batchnorm(feature[:, :, :6])
            #     feature = process_ohlc_inf_nan_fill0_cuda(feature)
            pred = self.GRU_model(feature.float())
            # 这里使用NDCG @k来计算损失
            pred.requires_grad_(True)
            # 清零梯度
            self.train_optimizer.zero_grad()
            if self.loss == "ic":
                # 路线 2
                with torch.no_grad():
                    # 计算每个样本的ndcg变化
                    lambda_grads = compute_delta_ndcg(pred.detach(), label.detach(), 1, sigma=self.sigma, linear=self.linear_ndcg)
                    lambda_grads = lambda_grads / lambda_grads.sum()
                loss = self.loss_fn(pred, label)
                grad = torch.autograd.grad(loss, pred, create_graph=True)[0]
                grad_norm = grad.norm(2)
                if self.combine_type == 'mult': # 相乘形式
                    lambda_grads = apply_mask_preserve_norm(grad, lambda_grads, method = 'l2')
                    # lambda_grads = lambda_grads * grad * 100
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
                    lambda_grads = lambda_grads / lambda_grads.norm(2) * grad_norm # 保证梯度范数一致
                else:
                    raise ValueError(f"Unknown combine_type: {self.combine_type}")
            else:
                raise ValueError(f"Unknown loss: {self.loss}")
            pred.backward(lambda_grads)                
            torch.nn.utils.clip_grad_norm_(self.GRU_model.parameters(), 3.0) 
            # 手动更新梯度
            self.logger.debug(f"\n 手动更新梯度")
            with torch.no_grad():
                lr = self.train_optimizer.param_groups[0]['lr']
                for p in self.GRU_model.parameters():
                    if p.grad is not None:
                        p.data -= lr * p.grad
            # 优化器更新梯度
            # self.logger.debug(f"\n 优化器{self.optimizer}更新梯度")
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
            # 计算一些指标
            for name in self.display_list:
                result['train_'+name].append(self.metric_fn(pred, label, name = name))
            

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
                    print(f"Epoch Avg {layer_name} Grad Norm: {avg_layer_norm:.6f}, \
                          Epoch Avg {layer_name} Grad Norm: Ratio: {avg_layer_norm/avg_grad_norm:.6f}")

        for name in self.display_list:
            result_agg['train_'+name] = np.mean(result['train_'+name])
        
        return result_agg
    
    
    @timing
    def test_epoch(self, data_loader):
        self.GRU_model.eval()
        result = defaultdict(list)
        result_agg = defaultdict(lambda : np.nan)
        for data, weight in data_loader:
            data.squeeze_(0) # 去除横截面 dim
            feature = data[:, :, :self.d_feat].to(self.device)
            # feature[torch.isnan(feature)] = 0
            label = data[:, -1, self.d_feat].to(self.device)
            if self.ohlc:
                if self.minute:
                    feature = process_minute_cuda(feature)
                else:
                    feature = process_ohlc_cuda(feature)
                feature = process_ohlc_batchwinsor(feature)
                feature = process_ohlc_batchnorm(feature) 
                feature = process_ohlc_inf_nan_fill0_cuda(feature)
            # if self.ohlc:
            #     feature = process_ohlc_cuda(feature)
            #     feature[:, :, :6] = process_ohlc_batchwinsor(feature[:, :, :6])
            #     feature[:, :, :6] = process_ohlc_batchnorm(feature[:, :, :6])
            #     feature = process_ohlc_inf_nan_fill0_cuda(feature)
            with torch.no_grad():
                pred = self.GRU_model(feature.float())
                # 计算RankNet交叉熵损失（仅用于观察）
                loss = ranknet_cross_entropy_loss(pred, label, sigma=self.sigma)
                # 计算NDCG
                score = calculate_ndcg_optimized(label, pred, self.n_layer, self.linear_ndcg)

            for name in self.display_list:
                result['val_'+name].append(self.metric_fn(pred.detach(), label.detach(), name = name))
        
        for name in self.display_list:
            result_agg['val_'+name] = np.mean(result['val_'+name])
        
        return result_agg
    
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

        save_path = get_or_create_path(self.save_path)

        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = {}
        evals_result["valid"] = {}
        
        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            train_result = self.train_epoch(train_loader) 
            self.logger.info("evaluating...")
            valid_result = self.test_epoch(valid_loader)
            
            for k, v in train_result.items():
                if k not in evals_result["train"]:
                    evals_result["train"][k] = []
                evals_result["train"][k].append(v)
                self.logger.info(
                    f"{Fore.RED}"
                    f"train {k}: {v:.6f}"
                    f"{Style.RESET_ALL}"
                )

            for k, v in valid_result.items():
                if k not in evals_result["valid"]:
                    evals_result["valid"][k] = []
                evals_result["valid"][k].append(v)
                self.logger.info(
                    f"{Fore.GREEN}"
                    f"valid {k}: {v:.6f}"
                    f"{Style.RESET_ALL}"
                )

            val_score = valid_result['val_'+self.metric]
            # 更新学习率
            self.lr_scheduler.step(val_score)
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
        torch.save(best_param, os.path.join(save_path, 'model.pt'))

        if self.use_gpu:
            torch.cuda.empty_cache()
        # 可视化损失
        visualize_evals_result_general(evals_result, list(range(self.n_epochs)), best_epoch,
                                       self.train_index, self.val_index, save_path, self.logger)

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        sampler_test = DailyBatchSampler(dl_test)
        self.test_index = sampler_test.reordered_index
        # 这里不能用dailysampler，否则 index 对不上
        test_loader = DataLoader(dl_test, batch_sampler=sampler_test, num_workers=self.n_jobs)
        self.GRU_model.eval()
        preds = []

        for data in test_loader:
            data.squeeze_(0) # 去除横截面 dim
            feature = data[:, :, :self.d_feat].to(self.device)
            if self.ohlc:
                if self.minute:
                    feature = process_minute_cuda(feature)
                else:
                    feature = process_ohlc_cuda(feature)
                feature = process_ohlc_batchwinsor(feature)
                feature = process_ohlc_batchnorm(feature) 
                feature = process_ohlc_inf_nan_fill0_cuda(feature)
            # if self.ohlc:
            #     feature = process_ohlc_cuda(feature)
            #     feature[:, :, :6] = process_ohlc_batchwinsor(feature[:, :, :6])
            #     feature[:, :, :6] = process_ohlc_batchnorm(feature[:, :, :6])
            #     feature = process_ohlc_inf_nan_fill0_cuda(feature)
            with torch.no_grad():
                pred = self.GRU_model(feature.float()).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=self.test_index)



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
        if torch.isnan(x).any():
            raise ValueError("GRU input contains NaN values")
        out, _ = self.rnn(x)
        if torch.isnan(out).any():
            raise ValueError("GRU output contains NaN values")
        out = self.fc_out(out[:, -1, :]).squeeze()
        if torch.isnan(out).any():
            raise ValueError("FC output contains NaN values")
        return out
