"""
自定义数据
"""

import qlib
import pandas as pd
from qlib.constant import REG_CN
from qlib.data.dataset.loader import StaticDataLoader, QlibDataLoader, DataLoader, NestedDataLoader

from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import ZScoreNorm, Fillna, DropnaLabel
from qlib.utils import init_instance_by_config
from qlib.utils.hxy_utils import convert_stock_code_to_qlib_format
# 读取QLIB 数据构造 label
provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)
qdl = QlibDataLoader(config=(['Ref($close, -11)/Ref($close, -1) - 1'],['label'])) 
label = qdl.load()
print(label.head())


print(df.head())

df = df.join(label, how='left')
df.dropna(subset=['label'], inplace=True)
print(df.head())
# dataset 
sdl_pkl = StaticDataLoader(config=df)

def read_alpha64():
    # 读取自有alpha64 数据
    root_dir = os.path.expanduser('~')
    df = pd.read_pickle(f'{root_dir}/GRU/Data/alpha64/qlib/alpha64.pkl')
    df.columns = df.columns.str.replace(r'\..*$', '', regex=True)
    # 处理股票代码
    df.index = df.index.set_levels(df.index.levels[1].map(convert_stock_code_to_qlib_format), level=1)
    df.index = df.index.set_levels(pd.to_datetime(df.index.levels[0].astype(str)), level=0)