from qlib.config import REG_CN
import qlib
import time

qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

from qlib.data import D

start = time.time()
for _ in range(10):
    df = D.features(['SH600519'], ['$close', '$volume'], start_time='2020-01-01', end_time='2020-12-31')
print(f"Elapsed: {time.time() - start:.2f}s")
