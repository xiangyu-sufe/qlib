import qlib
import pandas as pd
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import flatten_dict

# use default data
# NOTE: need to download data from remote: python scripts/get_data.py qlib_data_cn --target_dir ~/.qlib/qlib_data/cn_data
provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir

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

qdl = init_instance_by_config(qdl_config)
market = 'csi300'
df = qdl.load(instruments=market, start_time='2020-05-01', end_time='2020-05-31')
print(df)