
import qlib
import pandas as pd
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import flatten_dict
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import TSDatasetH

provider_uri = "/DATA/hxy/qlib/cn_data/"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)


qdl_config = {
    "class": "QlibDataLoader",
    "module_path": "qlib.data.dataset.loader",
    "kwargs": {
        "config": {
            "feature": (['EMA($close, 10)', 'EMA($close, 30)'], ['EMA10', 'EMA30'] ),
            "label": (['Ref($close, -11)/Ref($close, -1) - 1',],['RET_1',]),
        },
        "freq": 'day',
    },
}


handler = Alpha158(instruments='all', start_time='2010-01-01', end_time='2025-07-30')

# 使用HDF5格式保存（适合大数据）
print("计算并保存特征数据到HDF5...")
features = handler.fetch(col_set="feature")
features.to_hdf("/DATA/hxy/qlib/alpha158/alpha158_features.h5", key="features", mode="w")

labels = handler.fetch(col_set="label") 
labels.to_hdf("/DATA/hxy/qlib/alpha158/alpha158_features.h5", key="labels", mode="a")

print("数据已保存到HDF5文件")