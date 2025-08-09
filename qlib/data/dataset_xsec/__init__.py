from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import pandas as pd

from ..dataset.handler import DataHandler, DataHandlerLP  # type: ignore
from ..dataset import DatasetH  # type: ignore
from ...utils import time_to_slc_point  # type: ignore
from ...log import get_module_logger  # type: ignore
from ..dataset.utils import get_level_index  # type: ignore


class CSDataSampler:
    """
    Cross-Sectional (datetime-first) Data Sampler.

    - data is indexed by MultiIndex [datetime, instrument] and sorted by <datetime, instrument>.
    - Each __getitem__(i) returns the cross-section (one date) with step_len historical window per instrument:
        X: [N_inst_t, step_len, n_feat]
        y: [N_inst_t, step_len, n_y] if label_cols specified; else None
        index: List[(datetime, instrument)] length N_inst_t
    - fillna_type: 'none' | 'ffill' | 'ffill+bfill' (per-instrument along time)
    - One batch = one date (recommended); no padding required.
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
        # ensure sorted by (datetime, instrument)
        self.data = data.sort_index().copy()
        if feature_cols is None:
            feature_cols = list(self.data.columns)
        self.feature_cols = feature_cols
        self.label_cols = label_cols

        # Optional filtering by flt_data (aligned with data index)
        if flt_data is not None:
            flt_idx = flt_data[flt_data.astype(bool)].index
            self.data = self.data.loc[self.data.index.intersection(flt_idx)]

        self.step_len = int(step_len)
        self.fillna_type = fillna_type
        self.start = start
        self.end = end

        # Build calendar (ordered unique dates) within extended window so that step_len backfill works
        all_dates = self.data.index.get_level_values(0).unique().sort_values()
        # keep full calendar; we'll filter indexable dates later
        self.calendar = all_dates

        # Instruments universe
        self.instruments = self.data.index.get_level_values(1).unique().tolist()

        # Cast to numpy-friendly types if requested
        if dtype is not None:
            self.data = self.data.astype(dtype)

        # Precompute per-instrument time series DataFrames for efficient slicing
        # Each: index = dates, columns = feature_cols (+ label_cols if provided will be handled on-demand)
        self._per_inst: Dict[str, pd.DataFrame] = {}
        for inst, df_inst in self.data.groupby(level=1):
            ts = df_inst.droplevel(1).reindex(self.calendar)  # align to full calendar
            # Apply fillna strategy on features (and labels later similarly)
            if self.fillna_type in ("ffill", "ffill+bfill"):
                ts = ts.ffill()
                if self.fillna_type == "ffill+bfill":
                    ts = ts.bfill()
            self._per_inst[inst] = ts

        # Determine indexable dates: those with at least one instrument having a full step_len window
        idxable_dates = []
        for i in range(self.step_len - 1, len(self.calendar)):
            dt = self.calendar[i]
            w_start = self.calendar[i - self.step_len + 1]
            # check existence for any instrument
            ok_any = False
            for inst in self.instruments:
                ts = self._per_inst.get(inst)
                if ts is None:
                    continue
                window = ts.loc[w_start:dt, self.feature_cols]
                if len(window) == self.step_len and (self.fillna_type != "none" or not window.isna().any().any()):
                    ok_any = True
                    break
            if ok_any:
                idxable_dates.append(dt)
        # Apply start/end on indexable dates
        if start is not None:
            idxable_dates = [d for d in idxable_dates if d >= pd.Timestamp(start)]
        if end is not None:
            idxable_dates = [d for d in idxable_dates if d <= pd.Timestamp(end)]

        self.idx_dates: List[pd.Timestamp] = idxable_dates
        self.logger = get_module_logger("CSDataSampler")
        self.logger.info(f"CSDataSampler built with {len(self.idx_dates)} cross-sectional samples; step_len={self.step_len}")

    def __len__(self):
        return len(self.idx_dates)

    def __getitem__(self, i: int):
        dt = self.idx_dates[i]
        start_pos = self.calendar.get_loc(dt)
        w_start = self.calendar[start_pos - self.step_len + 1]
        X_list = []
        y_list = [] if self.label_cols is not None else None
        index_list: List[Tuple[pd.Timestamp, str]] = []
        for inst in self.instruments:
            ts = self._per_inst.get(inst)
            if ts is None:
                continue
            win_feat = ts.loc[w_start:dt, self.feature_cols]
            if len(win_feat) != self.step_len:
                continue
            # if no filling allowed, ensure no NaN
            if self.fillna_type == "none" and win_feat.isna().any().any():
                continue
            X_list.append(win_feat.values[np.newaxis, ...])  # [1, step_len, F]
            index_list.append((dt, inst))
            if y_list is not None:
                win_y = ts.loc[w_start:dt, self.label_cols]
                if self.fillna_type == "none" and win_y.isna().any().any():
                    # if label missing, drop this instrument
                    X_list.pop()
                    index_list.pop()
                    continue
                y_list.append(win_y.values[np.newaxis, ...])  # [1, step_len, Y]
        if len(X_list) == 0:
            # In rare case, no instrument valid on this date; raise or return empty
            n_y = 0 if self.label_cols is None else len(self.label_cols)
            return {
                "x": np.empty((0, self.step_len, len(self.feature_cols)), dtype=float),
                "y": None if y_list is None else np.empty((0, self.step_len, n_y)),
                "index": index_list,
            }
        X = np.concatenate(X_list, axis=0)
        y = None if y_list is None else np.concatenate(y_list, axis=0)
        return {"x": X, "y": y, "index": index_list}

    # Compatibility helpers
    def get_index(self) -> pd.MultiIndex:
        """
        Align with TSDataSampler.get_index(): return MultiIndex in order <datetime, instrument>.
        Useful for external code that needs to know the index mapping.
        """
        return self.data.index

    @property
    def empty(self) -> bool:
        """Mimic pandas.DataFrame.empty: True if no indexable samples."""
        return len(self) == 0


class CSDatasetH(DatasetH):
    """
    DatasetH subclass that prepares cross-sectional samples via CSDataSampler.
    Keeps the one-date-per-item behavior and <datetime, instrument> primary ordering.
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


__all__ = [
    "CSDataSampler",
    "CSDatasetH",
]
