# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function
from collections import defaultdict
import warnings
from typing import Dict, List, Tuple, Optional, DefaultDict, Union
import gc

import sys
import logging
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
from torch.utils.data.sampler import Sampler, BatchSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .pytorch_utils import count_parameters
from ...model.base import Model
from ...data.dataset.handler import DataHandlerLP
from ...model.utils import ConcatDataset
from ...data.dataset.weight import Reweighter
from ..loss.loss import (ic_loss, rankic_loss,
                         topk_ic_loss, topk_rankic_loss,
                           ranking_loss, pairwise_loss, mse,
                            topk_return)
from ..loss.ndcg import calculate_ndcg_optimized
from ..loss.miga import MIGALoss
from qlib.utils.hxy_utils import (compute_grad_norm, 
                                  compute_layerwise_grad_norm,
                                  IndexedSeqDataset,
                                  make_collate_fn,
                                  make_varlen_collate_fn,
                                  process_ohlc_cuda,
                                  process_minute_cuda,
                                  process_ohlc_batchwinsor,
                                  process_ohlc_batchnorm,
                                  process_ohlc_inf_nan_fill0_cuda, visualize_evals_result_general,
                                  VarLenIndexedSeqDataset
                                  )
from qlib.utils.timing import timing
from qlib.contrib.hxy_model.pytorch_miga_ts import (
    MIGAB1, MIGAB1VarLen, MIGAB2VarLenCrossAttn,
    MIGAB3VarLenMoE, MIGAB4VarLenCrossAttnAvgMoE,
    MIGAB5VarLenMoEGateTop
)
from colorama import Fore, Style, init
import matplotlib.pyplot as plt
import os
import random

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
    

def seed_worker(worker_id: int):
    """Set deterministic seeds for each DataLoader worker.
    Uses torch.initial_seed() to derive a distinct seed per worker, and applies it to
    Python's random and NumPy RNGs as recommended by PyTorch docs.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class MIGA(Model):
    """
    MIGA模型专用训练器，继承自基础Trainer类
    重写fit函数以适配MIGA模型的特殊输出格式和损失函数
    """
    
    def __init__(
        self,
        d_feat=6, # 模型参数
        hidden_size=64,
        num_groups: Optional[int] = None,
        num_experts: int = 1,
        expert_type: str = "gru",
        num_experts_per_group: Optional[int] = None,   
        num_heads: int = 4, # 头数
        d_model: int = 64, # attention 的维度
        top_k: int = 2,
        expert_output_dim: int = 1,
        num_layers=2,
        dropout=0.0, # 训练参数
        n_epochs=200,
        lr=0.001,
        metric="ic",
        batch_size=2000,
        early_stop=20,
        loss="ic",
        optimizer="adam",
        n_jobs=0,
        GPU=0,
        seed=None,
        lambda_reg=0.1,
        n_layer=10,
        linear_ndcg=False,
        omega=0.1,
        epsilon=1,
        omega_scheduler=None,
        omega_decay=0.96,
        omega_step_epoch=None,
        omega_after=None,
        debug=False,
        save_path=None,
        step_len=1,
        news_store=None,
        use_news=True,
        version=None,
        padding_method="zero",
        ohlc=False,
        minute=False,
        display_list=['loss', 'ic', 'rankic', 'ndcg', 'topk_return'],
        **kwargs,
    ):
        # Set logger.
        assert version is not None, "version must be specified"
        self.logger = get_module_logger("MIGA News")
        self.logger.setLevel(logging.DEBUG)
        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.DEBUG)
            self.logger.addHandler(handler)
        self.logger.addHandler(logging.FileHandler(f"{save_path}/train.log"))
        self.logger.info(f"MIGA News pytorch version {version}...")
     
        # set hyper-parameters.
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.d_model = d_model
        self.num_groups = num_groups
        self.num_experts = num_experts
        self.expert_type = expert_type
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
        self.optimizer = optimizer
        self.n_jobs = n_jobs
        self.GPU = GPU
        self.seed = seed if seed is not None else 42
        self.lambda_reg = lambda_reg
        self.debug = debug
        self.save_path = save_path
        self.step_len = step_len
        self.news_store_path = news_store
        self.use_news = use_news
        self.padding_method = padding_method
        self.ohlc = ohlc
        self.minute = minute
        self.display_list = display_list
        self.omega = omega
        self.epsilon = epsilon
        self.omega_scheduler = omega_scheduler
        self.omega_decay = omega_decay
        self.omega_step_epoch = omega_step_epoch
        self.omega_after = omega_after
        self.n_layer = n_layer
        self.linear_ndcg = linear_ndcg

        if  self.loss == "miga":
            self.Miga_loss = MIGALoss(omega=self.omega, epsilon=self.epsilon)
            self.display_list += ['router_loss', ]
            # 定义omega_scheduler
            if self.omega_scheduler == "exp":
                def omega_scheduler_func(epoch):
                    # 指数衰减
                    # 但超过一定 epoch 数不再变化
                    if epoch < 10:
                        return self.omega * (self.omega_decay ** epoch)
                    else:
                        return self.omega * (self.omega_decay ** 10)     
            elif self.omega_scheduler == "step":
                def omega_scheduler_func(epoch):
                    return self.omega if epoch < self.omega_step_epoch else self.omega_after
            else:
                omega_scheduler_func = None
        elif self.loss == "ic":
            self.logger.info("Use IC loss")
        else:
            raise ValueError("unknown loss `%s`" % self.loss)    
            
        self.logger.info(
            "MIGA parameters setting:"
            "\nd_feat : {}"
            "\nhidden_size : {}"
            "\nd_model : {}"
            "\nnum_experts : {}"
            "\nnum_groups : {}"
            "\nnum_experts_per_group : {}"
            "\nnum_heads : {}"
            "\nohlc : {}"
            "\minute : {}"
            "\ntop_k : {}"
            "\nexpert_output_dim : {}"
            "\nomega : {}"
            "\nepsilon : {}"
            "\nomega_scheduler : {}"
            "\nomega_decay : {}"
            "\nomega_step_epoch : {}"
            "\nomega_after : {}"
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
            "\n\033[31mpadding_method : {}\033[0m"
            "\nsave_path : {}".format(
                self.d_feat,
                self.hidden_size,
                self.d_model,
                self.num_experts,
                self.num_groups,
                self.num_experts_per_group,
                self.num_heads,
                self.ohlc,
                self.minute,
                self.top_k,
                self.expert_output_dim,
                self.omega,
                self.epsilon,
                self.omega_scheduler,
                self.omega_decay,
                self.omega_step_epoch,
                self.omega_after,
                self.num_layers,
                self.dropout,
                self.n_epochs,
                self.lr,
                self.metric,
                self.batch_size,
                self.early_stop,
                self.optimizer.lower(),
                self.loss,
                self.device,
                self.n_jobs,
                self.use_gpu,
                self.seed,
                self.debug,
                self.padding_method,
                self.save_path,
            )
        )
    
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            # 确保CUDA操作的确定性
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        self.ablation_study(version)
        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.MIGA_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.MIGA_model.parameters(), lr=self.lr)
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
        self.MIGA_model.to(self.device)
        self.logger.info("model:\n{:}".format(self.MIGA_model))
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.MIGA_model)))
        
        # 路由器设置
        # if self.use_news:
        #     self.logger.info(f"{Fore.RED}使用新闻数据{Style.RESET_ALL}")
        #     router = PriceNewsRouter(
        #         price_dim=self.d_feat,
        #         news_dim=1024,
        #         d_model=self.hidden_size,
        #         n_heads=self.num_heads, # 这里的 num_heads 共用了
        #     d_gru=self.hidden_size,
        #     dropout=self.dropout
        #     )

        #     # 定义 MIGA 新闻模型
        #     self.MIGA_model = MIGANewsModel(
        #         input_dim=self.d_feat,
        #         d_gru=self.hidden_size,
        #         news_dim=1024,
        #         num_groups=self.num_groups,
        #         num_experts_per_group=self.num_experts_per_group,
        #         num_heads=self.num_heads,
        #         top_k=self.top_k,
        #         expert_output_dim=self.expert_output_dim,
        #         router=router,
        #     )
        # else:
        #     self.logger.info(f"{Fore.BLUE}不使用新闻数据{Style.RESET_ALL}")
        #     router = Router(
        #         input_dim=self.d_feat,
        #         hidden_dim=self.hidden_size,
        #         num_groups=self.num_groups,
        #         num_experts_per_group=self.num_experts_per_group,
        #     )
        #     self.MIGA_model = MIGAModel(
        #         input_dim=self.d_feat,
        #         num_groups=self.num_groups,
        #         num_experts_per_group=self.num_experts_per_group,
        #         num_heads=self.num_heads,
        #         top_k=self.top_k,
        #         expert_output_dim=self.expert_output_dim,
        #         router=router,
        #     )


    
    def ablation_study(self, version = "B1"):
        # 消融实验
        assert version in ("B1", "B2", "B3", "B4", "B5"), "version must be in (B1, B2, B3, B4, B5)"
        if version == "B1":
            self.MIGA_model = MIGAB1VarLen(
                price_dim=self.d_feat,
                news_dim=1024,
                hidden_dim=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                frozen=False,
                model_path=None,
                padding_method=self.padding_method,
                )
        elif version == "B2":
            self.MIGA_model = MIGAB2VarLenCrossAttn(
                price_dim=self.d_feat,
                news_dim=1024,
                hidden_dim=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                frozen=False,
                model_path=None,
                padding_method=self.padding_method,        
                n_heads = self.num_heads,
                d_model = self.d_model,
            )
        elif version == "B3":
            self.MIGA_model = MIGAB3VarLenMoE(
                price_dim=self.d_feat,
                news_dim=1024,
                hidden_dim=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                frozen=False,
                model_path=None,
                padding_method=self.padding_method,
                n_heads=self.num_heads,
                d_model=self.d_model,
                num_experts=self.num_experts,
                expert_type=self.expert_type,
            )
        elif version == "B4":
            self.MIGA_model = MIGAB4VarLenCrossAttnAvgMoE(
                price_dim=self.d_feat,
                news_dim=1024,
                hidden_dim=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                frozen=False,
                model_path=None,
                padding_method=self.padding_method,
                n_heads=self.num_heads,
                d_model=self.d_model,
                num_experts=self.num_experts,
            )
        elif version == "B5":
            self.MIGA_model = MIGAB5VarLenMoEGateTop(
                price_dim=self.d_feat,
                news_dim=1024,
                hidden_dim=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                frozen=False,
                model_path=None,
                padding_method=self.padding_method,
                n_heads=self.num_heads,
                d_model=self.d_model,
                num_experts=self.num_experts,
                expert_type=self.expert_type,
                topk=self.top_k,
            )
        else:
            raise ValueError("...")

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def load_model(self):
        self.MIGA_model.load_state_dict(torch.load(
            os.path.join(self.save_path, 'model.pt'), 
            map_location='cpu',
        ))
        self.fitted = True

    def mse(self, pred, label, weight):
        loss = weight * (pred - label) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label, hidden=None, weight=None)->Union[torch.Tensor, Dict[str, torch.Tensor]]:
        mask = ~torch.isnan(label)
        if self.loss == "miga": 
            return self.Miga_loss(pred[mask], label[mask], hidden)
        elif self.loss == 'ic':
            return ic_loss(pred[mask], label[mask])
        else:
            raise ValueError("unknown loss `%s`" % self.loss)
        
        
    def metric_fn(self, pred, label, name, hidden=None, topk=None):
        mask = torch.isfinite(label)

        if name in ("", "loss", "ic", "rankic", "ndcg", "topk_return", "router_loss"):
            if name == "ic":
                return -ic_loss(pred[mask], label[mask]).item()
            elif name == "rankic":
                return -rankic_loss(pred[mask], label[mask]).item()
            elif name == "ndcg":
                return calculate_ndcg_optimized(label[mask], pred[mask], self.n_layer, linear=self.linear_ndcg).item()
            elif name == "topk_return":
                return topk_return(pred[mask], label[mask], k=self.n_layer).item()
            elif name == "loss":
                if self.loss == "miga":
                    loss_dict = self.loss_fn(pred[mask], label[mask], hidden=hidden)
                    if isinstance(loss_dict, dict):
                        return -loss_dict['total_loss'].item()
                    else:
                        raise ValueError("MIGA Loss return a torch.Tensor, but expect a dict")
                else:
                    loss = self.loss_fn(pred[mask], label[mask], hidden=hidden)
                    if isinstance(loss, torch.Tensor):
                        return -loss.item()
                    else:
                        raise ValueError(f"Loss {self.loss} but expect a torch.Tensor")
            elif name == "router_loss":
                assert self.loss == "miga"
                loss_dict = self.loss_fn(pred[mask], label[mask], hidden=hidden)
                if isinstance(loss_dict, dict):
                    return -loss_dict['router_loss'].item()
                else:
                    raise ValueError("MIGA Loss return a torch.Tensor, but expect a dict")
                
        raise ValueError("unknown metric `%s`" % name)
        
    @timing
    def train_epoch(self, data_loader):
        import time
        
        
        # torch.cuda.synchronize()        # 开始前清空队列
        # start = time.time()
        
        self.MIGA_model.train()
        # Debug模式下记录每个batch的梯度信息
        result = defaultdict(list)
        result_agg = defaultdict(lambda : np.nan)
        if self.debug:
            epoch_grad_norms = []
            epoch_grad_norms_layer = []

        total_len = 0
        count = 0
        # pbar = tqdm(data_loader, desc="training...", file=sys.stdout)
        for i, (data, news, news_mask) in enumerate(data_loader):
            if i < self.step_len:
                continue
            # ------------- IO 计时开始 ------------
            
            # 去除横截面 dim
            data.squeeze_(0) 
            news.squeeze_(0)
            news_mask.squeeze_(0)
            # 取出量价、新闻、mask
            feature = data[:,:, :self.d_feat].to(self.device, non_blocking=True)
            label = data[:, -1, self.d_feat].to(self.device, non_blocking=True)
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
            news_feature = news.to(self.device, non_blocking=True)
            news_mask = news_mask.to(self.device, non_blocking=True)
            # torch.cuda.synchronize()            # 保证搬运结束
            
            # ------------- IO 计时结束 ------------
            
            # ---------- ② GPU 计时开始 ----------
            ts_io = time.time()
            if self.ohlc:
                # 使用 ohlc 数据
                # 先时序归一化+ winsor + batchnorm + fill0
                feature = process_ohlc_cuda(feature)
                feature = process_ohlc_batchwinsor(feature)
                feature = process_ohlc_batchnorm(feature)
                feature = process_ohlc_inf_nan_fill0_cuda(feature)
            io_time = time.time() - ts_io    
            t1 = time.time()
            if self.use_news:
                output = self.MIGA_model(feature, news_feature, news_mask)
            else:
                output = self.MIGA_model(feature)
            
            pred = output['predictions']
            routing_weights = output['routing_weights']
            if self.loss == "miga":
                loss_dict = self.loss_fn(pred, label, routing_weights)
            else:
                loss_dict = self.loss_fn(pred, label)
            if isinstance(loss_dict, dict):
                loss = loss_dict['total_loss']
            else:
                loss = loss_dict
            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.MIGA_model.parameters(), 3.0)
            self.train_optimizer.step()
            
            # torch.cuda.synchronize()            # GPU 全部算完
            # gpu_time = time.time() - t1         # ---------- ② GPU 计时结束 ----------

            # if getattr(self, "debug_timing", False):
            #     total = io_time + gpu_time
            #     print(f"step {i:04d} | IO {io_time*1e3:.1f} ms "
            #           f"| GPU {gpu_time*1e3:.1f} ms | IO% {io_time/total*100:.1f}")
            
            if self.debug:
                # 计算梯度范数
                grad_norm = compute_grad_norm(self.MIGA_model)
                grad_norm_layer = compute_layerwise_grad_norm(self.MIGA_model)
                epoch_grad_norms.append(grad_norm)
                epoch_grad_norms_layer.append(grad_norm_layer)

            total_len += len(data)
            count += 1
            # pbar.set_postfix({"Average Length": total_len / count})
            # 记录损失
            for name in self.display_list:
                result['train_'+name].append(self.metric_fn(pred.detach(), 
                                                            label.detach(), 
                                                            hidden = routing_weights,
                                                            name = name))

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
            result_agg['train_'+name] = float(np.mean(result['train_'+name]))

        # torch.cuda.synchronize()        # 确保 GPU 全部算完
        # print(f"epoch {i} wall-time: {time.time()-start:.2f}s")
        return result_agg
    
    @timing
    def test_epoch(self, data_loader):
        self.MIGA_model.eval()
        result = defaultdict(list)
        result_agg = defaultdict(lambda : np.nan)
        
        total_len = 0
        count = 0
        # pbar = tqdm(data_loader, desc="evaluating...") 
        for data, news, news_mask in data_loader:
            # 去除横截面 dim
            data.squeeze_(0) 
            news.squeeze_(0)
            news_mask.squeeze_(0)
            feature = data[:,:, :self.d_feat].to(self.device, non_blocking=True)
            label = data[:, -1, self.d_feat].to(self.device, non_blocking=True)
            news_feature = news.to(self.device, non_blocking=True)
            news_mask = news_mask.to(self.device, non_blocking=True)
            if self.ohlc:
                # 使用 ohlc 数据
                # 先时序归一化+ winsor + batchnorm + fill0
                feature = process_ohlc_cuda(feature)
                feature = process_ohlc_batchwinsor(feature)
                feature = process_ohlc_batchnorm(feature)
                feature = process_ohlc_inf_nan_fill0_cuda(feature)            
            with torch.no_grad():
                if self.use_news:
                    output = self.MIGA_model(feature.float(), news_feature.float(), news_mask)
                else:
                    output = self.MIGA_model(feature.float())
                pred = output['predictions']
                routing_weights = output['routing_weights']

                total_len += len(data)
                count += 1
                # pbar.set_postfix({"Average Length": total_len / count})
            for name in self.display_list:
                result['val_'+name].append(self.metric_fn(pred.detach(), label.detach(), hidden = routing_weights, name = name))
        
        for name in self.display_list:
            result_agg['val_'+name] = float(np.mean(result['val_'+name]))

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
    
        dl_train = VarLenIndexedSeqDataset(dl_train,news_store_path=self.news_store_path)
        dl_valid = VarLenIndexedSeqDataset(dl_valid,news_store_path=self.news_store_path)

        if reweighter is None:
            wl_train = np.ones(len(dl_train))
            wl_valid = np.ones(len(dl_valid))
        elif isinstance(reweighter, Reweighter):
            wl_train = reweighter.reweight(dl_train)
            wl_valid = reweighter.reweight(dl_valid)
        else:
            raise ValueError("Unsupported reweighter type.")

        # Ensure deterministic DataLoader worker seeding via generator and worker_init_fn
        _base_seed = int(self.seed) if self.seed is not None else 0
        _generator = torch.Generator()
        _generator.manual_seed(_base_seed)

        train_loader = DataLoader(
            dl_train,
            batch_sampler=sampler_train,
            num_workers=self.n_jobs,
            collate_fn=make_varlen_collate_fn(),
            pin_memory=True,
            prefetch_factor=2 if self.n_jobs > 0 else None,
            persistent_workers=True,
            worker_init_fn=seed_worker,
            generator=_generator,
        )
        valid_loader = DataLoader(
            dl_valid,
            batch_sampler=sampler_valid,
            num_workers=self.n_jobs,
            collate_fn=make_varlen_collate_fn(),
            pin_memory=True,
            prefetch_factor=2 if self.n_jobs > 0 else None,
            persistent_workers=True,
            worker_init_fn=seed_worker,
            generator=_generator,
        )
        # next(iter(train_loader)) # 预加载
        # next(iter(valid_loader)) # 预加载
        save_path = get_or_create_path(self.save_path)
        
        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = {}
        evals_result["valid"] = {}
        # train
        self.logger.info("training...\n")
        self.fitted = True
        best_param = None
        
        for step in tqdm(range(self.n_epochs)):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            if step == 0:
                self.MIGA_model.reset_count()
            train_result = self.train_epoch(train_loader)
            if step == 0:
                self.logger.info("Train News Coverage Ratio: %.6lf", self.MIGA_model.news_coverage_ratio)
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
            self.logger.info(
                f"{Fore.BLUE}"
                f"Learning Rate: {self.train_optimizer.param_groups[0]['lr']}"
                f"{Style.RESET_ALL}"
            )
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
            torch.save(best_param, os.path.join(save_path, 'model.pt'))

        if self.use_gpu:
            torch.cuda.empty_cache()
        # 可视化损失
        visualize_evals_result_general(evals_result,
                                       list(range(self.n_epochs)),
                                       best_epoch,
                                       self.train_index, 
                                       self.val_index,
                                       save_path, 
                                       self.logger)
        self.logger.info("回收train loader, valid loader...")
        self.train_loader = None 
        self.valid_loader = None
        gc.collect()

    
    
    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        sampler_test = DailyBatchSampler(dl_test)
        dl_test = IndexedSeqDataset(dl_test, news_store_path=self.news_store_path)


        test_loader = DataLoader(dl_test,
                                 batch_sampler=sampler_test,
                                 num_workers=self.n_jobs,
                                 collate_fn=make_collate_fn(),
                                 worker_init_fn=seed_worker,
                                 generator=torch.Generator().manual_seed(int(self.seed) if self.seed is not None else 0))
        self.test_index = sampler_test.reordered_index
        self.MIGA_model.eval()
        preds = []

        for data, news, news_mask in test_loader:
            # 去除横截面 dim
            data.squeeze_(0) 
            news.squeeze_(0)
            news_mask.squeeze_(0)
            feature = data[:,:, :self.d_feat].to(self.device, non_blocking=True)
            # label = data[:, -1, self.d_feat].to(self.device, non_blocking=True)
            news_feature = news.to(self.device, non_blocking=True)
            news_mask = news_mask.to(self.device, non_blocking=True)
            if self.ohlc:
                # 使用 ohlc 数据
                # 先时序归一化+ winsor + batchnorm + fill0
                feature = process_ohlc_cuda(feature)
                feature = process_ohlc_batchwinsor(feature)
                feature = process_ohlc_batchnorm(feature)
                feature = process_ohlc_inf_nan_fill0_cuda(feature)            
            with torch.no_grad():
                if self.use_news:
                    output = self.MIGA_model(feature.float(), news_feature.float(), news_mask)
                else:
                    output = self.MIGA_model(feature.float())
                pred = output['predictions'].cpu().numpy()
                

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=self.test_index)
    
    def predict_train(self, dataset, segment = 'train'):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")        

        dl = dataset.prepare(segment, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl.config(fillna_type="ffill+bfill") # 可以填充
        index = dl.get_index()
        dl = IndexedSeqDataset(dl, news_store_path=self.news_store_path)

        test_loader = DataLoader(dl,
                                 batch_size=self.batch_size,
                                 num_workers=self.n_jobs,
                                 collate_fn=make_collate_fn())
        self.MIGA_model.eval()
        preds = []

        for data, news_mask in test_loader:
            data.squeeze_(0) # 去除横截面 dim
            news_mask.squeeze_(0)
            feature = data[:, :, :self.d_feat].to(self.device)
            news_feature = data[:, :, self.d_feat+1:].to(self.device)
            news_mask = news_mask.to(self.device)
            if self.ohlc:
                # 使用 ohlc 数据
                # 先时序归一化+ winsor + batchnorm + fill0
                feature = process_ohlc_cuda(feature)
                feature = process_ohlc_batchwinsor(feature)
                feature = process_ohlc_batchnorm(feature)
                feature = process_ohlc_inf_nan_fill0_cuda(feature)            

            with torch.no_grad():
                if self.use_news:
                    output = self.MIGA_model(feature.float(), news_feature.float(), news_mask)
                else:
                    output = self.MIGA_model(feature.float())
                pred = output['predictions'].detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)





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
        d_gru: int,
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
        self.d_gru = d_gru
        self.news_dim = news_dim
        self.num_groups = num_groups
        self.num_experts_per_group = num_experts_per_group
        self.num_heads = num_heads
        self.top_k = top_k
        self.expert_output_dim = expert_output_dim
        self.hidden_dim = self.num_groups * self.num_experts_per_group
        # self.batch_norm = nn.BatchNorm1d(1)
        # Router
        self.router = router 
        self.hidden_to_gate = nn.Linear(self.d_gru, self.hidden_dim)

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
        # 降维
        hidden_representations = self.hidden_to_gate(hidden_representations)
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
        self.NO_NEWS_TOKEN = torch.nn.Parameter(torch.zeros(1, d_news))
        nn.init.xavier_uniform_(self.NO_NEWS_TOKEN)
        self.price_proj = nn.Linear(d_price, d_model) 
        self.news_proj = nn.Linear(d_news, d_model) 
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        
        self.resid_dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)
        self.gru = nn.GRU(input_size=2*d_model, hidden_size=d_gru, batch_first=True)

    def forward(self, price, news, mask):
        """
        price: [N, T, D_p]
        news:  [N, T, D_n]
        mask:  [N, T]
        return: [N, T, d_model]
        """
        has_real_news = (~mask).any(dim=1).float().unsqueeze(-1)  # shape: [N, 1]
        # 方案一 学习一个 NO NEWS TOKEN
        all_masked = mask.all(dim=1)  # shape: [N], bool
        news[all_masked, -1, :] = self.NO_NEWS_TOKEN # 最后一个设为 NO NEWS TOKEN 
        mask[all_masked, -1] = False
        # 1. 投影
        # 
        price_proj = self.price_proj(price)  # [N, T, d_model]
        news_proj = self.news_proj(news)    # [N, T, d_model]

        # 2. Cross-Attention: q=price, k/v=news
        # nn.MultiheadAttention expects input as [batch, seq_len, embed_dim]
        attn_out, _ = self.cross_attn(query=price_proj,
                                      key=news_proj,
                                      value=news_proj,
                                      key_padding_mask=mask)

        # 3. Residual + Dropout
        attn_out = self.ln(price_proj + self.resid_dropout(attn_out))  # [N, T, d_model]
        fused = torch.cat([price_proj, attn_out], dim=-1)  # [N, T, 2*d_model]
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
        # self.activation = nn.LeakyReLU()

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
        self.fc = nn.Linear(self.hidden_dim, self.fc_dim)
                
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
        # hidden = self.activation(hidden)
        
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

