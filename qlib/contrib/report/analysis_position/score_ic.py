# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
import numpy as np

from ..graph import ScatterGraph
from ..utils import guess_plotly_rangebreaks


def _get_score_ic(pred_label: pd.DataFrame):
    """

    :param pred_label:
    :return:
    """
    concat_data = pred_label.copy()
    concat_data.dropna(axis=0, how="any", inplace=True)
    _ic = concat_data.groupby(level="datetime", group_keys=False).apply(
        lambda x: x["label"].corr(x["score"])
    )
    _rank_ic = concat_data.groupby(level="datetime", group_keys=False).apply(
        lambda x: x["label"].corr(x["score"], method="spearman")
    )
    return pd.DataFrame({"ic": _ic, "rank_ic": _rank_ic})


def _get_top_score_ic(pred_label: pd.DataFrame, n_layer: int = 5):
    """
    计算多头 IC，根据指定的层数计算不同百分位的 IC 和 RankIC
    
    Parameters
    ----------
    pred_label : pd.DataFrame
        包含 'score' 和 'label' 列的 DataFrame，索引为 MultiIndex 
        (instrument, datetime)
    n_layer : int, default=5
        层数，用于计算百分位。例如 n_layer=5 表示取前 20% 的预测值
        
    Returns
    -------
    pd.DataFrame
        包含不同百分位的 IC 和 RankIC 的 DataFrame
    """
    concat_data = pred_label.copy()
    concat_data.dropna(axis=0, how="any", inplace=True)
    
    # 计算百分位阈值
    percentile = 100 / n_layer  # 例如 n_layer=5 时，percentile=20
    
    def calculate_percentile_ic(group):
        """计算指定百分位的 IC 和 RankIC"""
        # 获取当前百分位的阈值
        threshold = group["score"].quantile(1 - percentile / 100)
        
        # 筛选出大于等于阈值的预测值（前 percentile%）
        mask = group["score"] >= threshold
        filtered_group = group[mask]
        
        if len(filtered_group) < 2:  # 至少需要2个点才能计算相关性
            return pd.Series({"ic": np.nan, "rank_ic": np.nan})
        
        # 计算 IC 和 RankIC
        ic = filtered_group["label"].corr(filtered_group["score"])
        rank_ic = filtered_group["label"].corr(
            filtered_group["score"], method="spearman"
        )
        
        return pd.Series({"ic": ic, "rank_ic": rank_ic})
    
    # 按时间分组计算
    result = concat_data.groupby(level="datetime", group_keys=False).apply(calculate_percentile_ic)
    
    # 重命名列以区分不同层数
    result.columns = [f"ic_top_{percentile:.0f}pct", f"rank_ic_top_{percentile:.0f}pct"]
    
    return result


def top_score_ic_graph(pred_label: pd.DataFrame, n_layer: int = 5, show_notebook: bool = True, **kwargs) -> [list, tuple]:
    """
    多头 Score IC 图表
    
    Parameters
    ----------
    pred_label : pd.DataFrame
        包含 'score' 和 'label' 列的 DataFrame，索引为 MultiIndex (instrument, datetime)
    n_layer : int, default=5
        层数，用于计算百分位
    show_notebook : bool, default=True
        是否在 notebook 中显示图表
    **kwargs : 
        其他参数传递给 ScatterGraph
        
    Returns
    -------
    list or tuple
        如果 show_notebook 为 True，在 notebook 中显示；否则返回 plotly.graph_objs.Figure 列表
    """
    _ic_df = _get_top_score_ic(pred_label, n_layer)
    
    # 计算 IC 和 RankIC 的均值
    ic_mean = _ic_df.iloc[:, 0].mean()  # 第一列是 IC
    rank_ic_mean = _ic_df.iloc[:, 1].mean()  # 第二列是 RankIC
    
    # 格式化标题，包含均值信息
    percentile = 100 / n_layer
    title = (
        f"Top Score IC (Top {percentile:.0f}% IC Mean: {ic_mean:.4f}, "
        f"Top {percentile:.0f}% RankIC Mean: {rank_ic_mean:.4f})"
    )
    print(title)

    _figure = ScatterGraph(
        _ic_df,
        layout=dict(
            title=title,
            xaxis=dict(
                tickangle=45, 
                rangebreaks=kwargs.get("rangebreaks", guess_plotly_rangebreaks(_ic_df.index))
            ),
        ),
        graph_kwargs={"mode": "lines+markers"},
    ).figure
    if show_notebook:
        ScatterGraph.show_graph_in_notebook([_figure])
    else:
        return (_figure,)


def score_ic_graph(pred_label: pd.DataFrame, show_notebook: bool = True, **kwargs) -> [list, tuple]:
    """score IC

        Example:


            .. code-block:: python

                from qlib.data import D
                from qlib.contrib.report import analysis_position
                pred_df_dates = pred_df.index.get_level_values(level='datetime')
                features_df = D.features(
                    D.instruments('csi500'), 
                    ['Ref($close, -2)/Ref($close, -1)-1'], 
                    pred_df_dates.min(), 
                    pred_df_dates.max()
                )
                features_df.columns = ['label']
                pred_label = pd.concat(
                    [features_df, pred], axis=1, sort=True
                ).reindex(features_df.index)
                analysis_position.score_ic_graph(pred_label)


    :param pred_label: index is **pd.MultiIndex**, index name is **[instrument, datetime]**; columns names is **[score, label]**.


            .. code-block:: python

                instrument  datetime        score         label
                SH600004  2017-12-11     -0.013502       -0.013502
                            2017-12-12   -0.072367       -0.072367
                            2017-12-13   -0.068605       -0.068605
                            2017-12-14    0.012440        0.012440
                            2017-12-15   -0.102778       -0.102778


    :param show_notebook: whether to display graphics in notebook, the default is **True**.
    :return: if show_notebook is True, display in notebook; else return **plotly.graph_objs.Figure** list.
    """
    _ic_df = _get_score_ic(pred_label)

    # 计算 IC 和 RankIC 的均值
    ic_mean = _ic_df["ic"].mean()
    rank_ic_mean = _ic_df["rank_ic"].mean()
    
    # 格式化标题，包含均值信息
    title = (
        f"Score IC (IC Mean: {ic_mean:.4f}, "
        f"RankIC Mean: {rank_ic_mean:.4f})"
    )
    print(title)

    _figure = ScatterGraph(
        _ic_df,
        layout=dict(
            title=title,
            xaxis=dict(
                tickangle=45, 
                rangebreaks=kwargs.get("rangebreaks", guess_plotly_rangebreaks(_ic_df.index))
            ),
        ),
        graph_kwargs={"mode": "lines+markers"},
    ).figure
    if show_notebook:
        ScatterGraph.show_graph_in_notebook([_figure])
    else:
        return _figure, title
