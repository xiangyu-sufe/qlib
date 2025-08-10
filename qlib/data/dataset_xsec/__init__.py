from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from ..dataset.handler import DataHandler, DataHandlerLP  # type: ignore
from ..dataset import DatasetH  # type: ignore
from ...utils import time_to_slc_point  # type: ignore
from ...log import get_module_logger  # type: ignore
from ..dataset.utils import get_level_index  # type: ignore
from ...utils import np_ffill, lazy_sort_index  # type: ignore


class CSDataSampler:
    """
    Datetime-major variant of TSDataSampler.

    - Data index: MultiIndex [datetime, instrument], sorted by (datetime, instrument).
    - Sample definition: identical to TSDataSampler — one instrument at one day with a time window.
    - Difference from TSDataSampler: memory layout is datetime-major (same-day stocks are contiguous).

    Returns from __getitem__ follow TSDataSampler: a numpy array window of shape
    [step_len, n_columns] for a single sample, or [batch, step_len, n_columns] for a batch.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        start,
        end,
        step_len: int,
        fillna_type: str = "none",
        dtype=None,
        flt_data: Optional[pd.Series] = None,
        feature_cols: Optional[List[str]] = None,
        label_cols: Optional[List[str]] = None,
    ):
        assert get_level_index(data, "datetime") == 0, "expect MultiIndex level 0 to be datetime"
        # Ensure sorted by (datetime, instrument)
        self.data = data.sort_index().copy()
        # Optional dtype cast
        if dtype is not None:
            self.data = self.data.astype(dtype)

        self.step_len = int(step_len)
        self.fillna_type = fillna_type
        self.start = start
        self.end = end
        self.logger = get_module_logger("CSDataSampler")

        # Build data_arr (datetime-major) and append a NaN row for padding
        kwargs = {"object": self.data}
        if dtype is not None:
            kwargs["dtype"] = dtype
        self.data_arr = np.array(**kwargs)

        nan_row = np.full((1, self.data_arr.shape[1]), np.nan, dtype=self.data_arr.dtype)
        self.data_arr = np.append(self.data_arr, nan_row, axis=0)
        self.nan_idx = self.data_arr.shape[0] - 1

        # Universe and calendar
        self.calendar = self.data.index.get_level_values(0).unique().sort_values()
        self.instruments = self.data.index.get_level_values(1).unique().tolist()

        # Optional sample-level filtering before slicing start/end
        self.flt_data = flt_data
        assert flt_data is None, "CSDataSampler does not support sample-level filtering"
        # Build indices
        self.idx_df, idx_map_dict = self.build_index(self.data)
        self.data_index = deepcopy(self.data.index)
        # Convert dict -> array of (row, col) aligned by sorted key of real_idx
        self.idx_map = self.idx_map2arr(idx_map_dict)
        # Slice by start/end
        self.idx_map, self.data_index = self.slice_idx_map_and_data_index(
            self.idx_map, self.idx_df, self.data.index, self.start, self.end
        )
        # Apply boolean filtering if provided
        if self.flt_data is not None:
            self.idx_map = self.flt_idx_map(self.flt_data, self.idx_map)

        # Cache array view of idx_df for fast window construction
        self.idx_arr = np.array(self.idx_df.values, dtype=np.float64)
        del self.data  # save memory

    def build_index(self, data: pd.DataFrame):
        """
        Build idx_df and idx_map dict just like TSDataSampler, but keep rows=datetime, cols=instrument.

        idx_df: DataFrame with shape [n_dates, n_instruments], each cell is the flat row index of data_arr
                (i.e., positional index in data with MultiIndex [datetime, instrument]). Missing pairs are NaN.
        idx_map: dict mapping real_idx (flat row index) -> (row_idx, col_idx) in idx_df
        """
        # Series of flat indices aligned with MultiIndex
        idx_df = pd.Series(range(data.shape[0]), index=data.index, dtype=object)
        # Ensure sorted rows/cols
        idx_df = lazy_sort_index(idx_df.unstack())
        # NOTE: the correctness of `__getitem__` depends on columns sorted here
        idx_df = lazy_sort_index(idx_df, axis=1)

        # Build mapping real_idx -> (row_idx, col_idx)
        idx_map: Dict[int, Tuple[int, int]] = {}
        for i, (_, row) in tqdm(enumerate(idx_df.iterrows()), "Building idx_map By Day"):
            for j, real_idx in enumerate(row):
                if not np.isnan(real_idx):
                    idx_map[int(real_idx)] = (i, j)
        return idx_df, idx_map

    @staticmethod
    def slice_idx_map_and_data_index(idx_map: np.ndarray, idx_df: pd.DataFrame, data_index: pd.MultiIndex, start, end):
        """
        Slice idx_map (2D array of [row_idx, col_idx]) by start/end dates using idx_df row index.
        Returns filtered idx_map and corresponding data_index values for those samples.
        """
        if start is None and end is None:
            return idx_map, data_index
        start_row_idx, end_row_idx = idx_df.index.slice_locs(start=start, end=end)
        mask = (idx_map[:, 0] >= start_row_idx) & (idx_map[:, 0] < end_row_idx)
        return idx_map[mask], data_index

    @staticmethod
    def idx_map2arr(idx_map: Dict[int, Tuple[int, int]]):
        """
        Convert dict {real_idx: (row, col)} to an array of shape [N, 2] sorted by real_idx ascending.
        """
        if len(idx_map) == 0:
            return np.empty((0, 2), dtype=np.int32)
        keys = np.array(sorted(idx_map.keys()), dtype=np.int64)
        rows = np.fromiter((idx_map[k][0] for k in keys), dtype=np.int32, count=len(keys))
        cols = np.fromiter((idx_map[k][1] for k in keys), dtype=np.int32, count=len(keys))
        return np.stack([rows, cols], axis=1)

    @staticmethod
    def flt_idx_map(flt_data: pd.Series, idx_map: np.ndarray):
        """
        Filter samples using a boolean Series aligned to data.index (flat rows). Keep samples
        whose corresponding real_idx is True. Since idx_map is built sorted by real_idx,
        we can index by the same sorted order.
        """
        if flt_data is None:
            return idx_map
        flt_data = flt_data.astype(bool)
        # positions where real_idx is kept
        kept_pos = np.flatnonzero(flt_data.values)
        if kept_pos.size == 0:
            return np.empty((0, 2), dtype=np.int32)
        # idx_map ordering follows sorted real_idx (0..N-1) — intersect
        # For safety, compute mask by building a boolean array of size len(flt_data)
        mask = np.zeros(len(flt_data), dtype=bool)
        mask[kept_pos] = True
        # real_idx sequence equals sorted keys; build that sequence length
        # We assume data_index is original data.index; its positional order matches real_idx
        real_idx_seq = np.arange(len(flt_data), dtype=np.int64)
        sel = mask[real_idx_seq]
        return idx_map[sel]

    def config(self, **kwargs):
        """
        Update configuration. For TS-like sampler, changing fillna_type only affects how indices
        are forward/backward filled in _get_indices; no need to rebuild data structures.
        """
        if "fillna_type" in kwargs:
            self.fillna_type = kwargs["fillna_type"]
        for k, v in kwargs.items():
            if k != "fillna_type":
                setattr(self, k, v)

    def __len__(self):
        return len(self.idx_map)

    def _get_indices(self, row: int, col: int) -> np.array:
        """
        Get flat indices of self.data_arr for the time window ending at (row, col) in idx_df.
        """
        indices = self.idx_arr[max(row - self.step_len + 1, 0) : row + 1, col]
        if len(indices) < self.step_len:
            indices = np.concatenate([np.full((self.step_len - len(indices),), np.nan), indices])

        if self.fillna_type == "ffill":
            indices = np_ffill(indices)
        elif self.fillna_type == "ffill+bfill":
            indices = np_ffill(np_ffill(indices)[::-1])[::-1]
        else:
            assert self.fillna_type == "none"
        return indices

    def _get_row_col(self, idx) -> Tuple[int]:
        """
        Resolve the (row, col) in idx_df for a given sample index, consistent with TSDataSampler.
        idx can be an int (sample ordinal) or a tuple (date, instrument).
        """
        if isinstance(idx, (int, np.integer)):
            real_idx = idx
            if 0 <= real_idx < len(self.idx_map):
                i, j = self.idx_map[real_idx]
            else:
                raise KeyError(f"{real_idx} is out of [0, {len(self.idx_map)})")
        elif isinstance(idx, tuple):
            date, inst = idx
            date = pd.Timestamp(date)
            # rows and cols are sorted
            i = np.searchsorted(self.idx_df.index.values, date, side="right") - 1
            j = np.searchsorted(self.idx_df.columns.values, inst, side="left")
        else:
            raise NotImplementedError("Unsupported index type")
        return i, j

    def __getitem__(self, idx: Union[int, Tuple[object, str], List[int]]):
        mtit = (list, np.ndarray)
        if isinstance(idx, mtit):
            indices = [self._get_indices(*self._get_row_col(i)) for i in idx]
            indices = np.concatenate(indices)
        else:
            indices = self._get_indices(*self._get_row_col(idx))

        indices = np.nan_to_num(indices.astype(np.float64), nan=self.nan_idx).astype(int)

        if (np.diff(indices) == 1).all():
            data = self.data_arr[indices[0] : indices[-1] + 1]
        else:
            data = self.data_arr[indices]
        if isinstance(idx, mtit):
            data = data.reshape(-1, self.step_len, *data.shape[1:])
        return data

    # Compatibility helpers
    def get_index(self) -> pd.MultiIndex:
        """
        Align with TSDataSampler.get_index(): return MultiIndex in order <datetime, instrument>.
        Useful for external code that needs to know the index mapping.
        """
        return self.data_index

    @property
    def empty(self) -> bool:
        """Mimic pandas.DataFrame.empty: True if no indexable samples."""
        return len(self) == 0


class CSDatasetH(DatasetH):
    """
    DatasetH subclass for datetime-major TS-like sampling via CSDataSampler.
    Keeps <datetime, instrument> primary ordering and TS-compatible sample semantics.
    """

    DEFAULT_STEP_LEN = 30

    def __init__(
        self,
        handler: Union[Dict, DataHandler],
        segments: Dict[str, Tuple],
        fetch_kwargs: Dict = {},
        step_len: int = DEFAULT_STEP_LEN,
        **kwargs,
    ):
        self.step_len = step_len
        super().__init__(handler=handler, segments=segments, fetch_kwargs=fetch_kwargs, **kwargs)

    def config(self, handler_kwargs: dict = None, step_len: Optional[int] = None, **kwargs):
        if step_len is not None:
            self.step_len = int(step_len)
        super().config(handler_kwargs=handler_kwargs, **kwargs)

    def _prepare_seg(
        self,
        slc,
        col_set=DataHandler.CS_ALL,
        data_key=DataHandlerLP.DK_I,
        fillna_type: str = "none",
        feature_cols: Optional[List[str]] = None,
        label_cols: Optional[List[str]] = None,
        flt_col: Optional[str] = None,
        **kwargs,
    ) -> CSDataSampler:
        # fetch dataframe from handler
        if hasattr(self, "fetch_kwargs"):
            df = self.handler.fetch(slc, col_set=col_set, data_key=data_key, **self.fetch_kwargs)
        else:
            df = self.handler.fetch(slc, col_set=col_set, data_key=data_key)

        # df index expected as <datetime, instrument>
        df = df.sort_index()

        flt_data = None
        if flt_col is not None and flt_col in df.columns:
            flt_data = df[flt_col].astype(bool)
            # keep features/labels only (drop flt column from features unless explicitly wanted)
            df = df.drop(columns=[flt_col])

        # detect default feature/label columns if not specified
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != "label"]
        if label_cols is None and "label" in df.columns and data_key != DataHandlerLP.DK_I:
            label_cols = ["label"]

        # derive start/end from slc
        try:
            start, end = slc.start, slc.stop
        except Exception:
            # If slc is a tuple like (start, end) or a named segment tuple
            if isinstance(slc, (list, tuple)):
                try:
                    start, end = slc[0]
                except Exception:
                    start, end = None, None
            else:
                start, end = None, None

        return CSDataSampler(
            data=df,
            start=start,
            end=end,
            step_len=self.step_len,
            fillna_type=fillna_type,
            dtype=None,
            flt_data=flt_data,
            feature_cols=feature_cols,
            label_cols=label_cols,
        )

    def prepare(
        self,
        segments: Union[List[str], Tuple[str], str, slice, pd.Index],
        col_set=DataHandler.CS_ALL,
        data_key=DataHandlerLP.DK_I,
        **kwargs,
    ) -> Union[List[CSDataSampler], CSDataSampler]:
        # Delegate to DatasetH.prepare, which will call our _prepare_seg under the hood
        return super().prepare(segments=segments, col_set=col_set, data_key=data_key, **kwargs)

class TSDatasetH(DatasetH):
    """
    (T)ime-(S)eries Dataset (H)andler


    Convert the tabular data to Time-Series data

    Requirements analysis

    The typical workflow of a user to get time-series data for an sample
    - process features
    - slice proper data from data handler:  dimension of sample <feature, >
    - Build relation of samples by <time, instrument> index
        - Be able to sample times series of data <timestep, feature>
        - It will be better if the interface is like "torch.utils.data.Dataset"
    - User could build customized batch based on the data
        - The dimension of a batch of data <batch_idx, feature, timestep>
    """

    DEFAULT_STEP_LEN = 30

    def __init__(self, step_len=DEFAULT_STEP_LEN, **kwargs):
        self.step_len = step_len
        super().__init__(**kwargs)

    def config(self, **kwargs):
        if "step_len" in kwargs:
            self.step_len = kwargs.pop("step_len")
        super().config(**kwargs)

    def setup_data(self, **kwargs):
        super().setup_data(**kwargs)
        # make sure the calendar is updated to latest when loading data from new config
        cal = self.handler.fetch(col_set=self.handler.CS_RAW).index.get_level_values("datetime").unique()
        self.cal = sorted(cal)

    @staticmethod
    def _extend_slice(slc: slice, cal: list, step_len: int) -> slice:
        # Dataset decide how to slice data(Get more data for timeseries).
        start, end = slc.start, slc.stop
        start_idx = bisect.bisect_left(cal, pd.Timestamp(start))
        pad_start_idx = max(0, start_idx - step_len)
        pad_start = cal[pad_start_idx]
        return slice(pad_start, end)

    def _prepare_seg(self, slc: slice, **kwargs) -> CSDataSampler:
        """
        split the _prepare_raw_seg is to leave a hook for data preprocessing before creating processing data
        NOTE: TSDatasetH only support slc segment on datetime !!!
        """
        dtype = kwargs.pop("dtype", None)
        if not isinstance(slc, slice):
            slc = slice(*slc)
        start, end = slc.start, slc.stop
        flt_col = kwargs.pop("flt_col", None)
        # TSDatasetH will retrieve more data for complete time-series

        ext_slice = self._extend_slice(slc, self.cal, self.step_len)
        data = super()._prepare_seg(ext_slice, **kwargs)

        flt_kwargs = deepcopy(kwargs)
        if flt_col is not None:
            flt_kwargs["col_set"] = flt_col
            flt_data = super()._prepare_seg(ext_slice, **flt_kwargs)
            assert len(flt_data.columns) == 1
        else:
            flt_data = None

        tsds = CSDataSampler(
            data=data,
            start=start,
            end=end,
            step_len=self.step_len,
            dtype=dtype,
            flt_data=flt_data,
        )
        return tsds



__all__ = [
    "CSDataSampler",
    "CSDatasetH",
    "TSDatasetH"
]
