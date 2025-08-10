import qlib
import numpy as np
import pandas as pd
from qlib.constant import REG_CN
from qlib.data.dataset.loader import StaticDataLoader, QlibDataLoader, DataLoader  
from qlib.data.dataset_xsec import CSDataSampler
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import ZScoreNorm, Fillna, DropnaLabel
from qlib.utils import init_instance_by_config
from torch.utils.data import Sampler

from qlib.utils.hxy_utils import read_ohlc

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


df = read_ohlc()
df.head()
df = df.loc[(slice("2016-01-01","2021-02-01"), ),:]
ds = CSDataSampler(df,
              start = "2016-01-01",
              end =  "2021-02-01",
              step_len = 20)
date_index = df.index.get_level_values(0).unique()
daily_sampler = DailyBatchSampler(ds)
for i, idx in enumerate(daily_sampler):
    print(idx)
    print(df.loc[(date_index[i],slice(None)),:])
    data = ds.__getitem__(idx)
    
    print(data[:,-1,:])
    print(data.shape)
    
