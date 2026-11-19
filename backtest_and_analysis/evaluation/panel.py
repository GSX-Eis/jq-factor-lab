"""Convert long factor observations to a datetime-by-asset panel."""

from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass(frozen=True)
class EvaluationPanel:
    """Factor, forward returns and a separate valid-pair mask per horizon."""

    datetimes: pd.DatetimeIndex
    assets: pd.Index
    factor: npt.NDArray[np.float64]
    returns: dict[Hashable, npt.NDArray[np.float64]]
    valid: dict[Hashable, npt.NDArray[np.bool_]]
    _index: pd.MultiIndex
    _columns: pd.Index

    @classmethod
    def from_long(
        cls,
        frame: pd.DataFrame,
        *,
        factor_column: Hashable = "factor",
        datetimes: pd.DatetimeIndex | None = None,
        assets: pd.Index | None = None,
    ) -> "EvaluationPanel":
        """Use one factor column and one column per horizon on a unique two-level index."""
        index = frame.index
        if not isinstance(index, pd.MultiIndex) or index.nlevels != 2 or not index.is_unique:
            raise ValueError("Expected a unique two-level (datetime, asset) index")

        times = index.get_level_values(0)
        symbols = index.get_level_values(1)
        if not isinstance(times, pd.DatetimeIndex) or times.hasnans or symbols.hasnans:
            raise ValueError("Expected datetime and asset keys without missing values")
        columns = frame.columns
        if (isinstance(columns, pd.MultiIndex) or not columns.is_unique or columns.hasnans
                or factor_column not in columns or len(columns) < 2):
            raise ValueError("Expected unique columns with a factor and at least one horizon")

        if datetimes is None:
            datetimes = times.unique().sort_values()
        if assets is None:
            assets = symbols.unique()
        if not isinstance(datetimes, pd.DatetimeIndex) or not isinstance(assets, pd.Index):
            raise TypeError("Axes must be a DatetimeIndex and an asset Index")
        if isinstance(assets, pd.MultiIndex) or any(
            not axis.is_unique or axis.hasnans for axis in (datetimes, assets)
        ):
            raise ValueError("Panel axes must have unique, non-missing keys")

        rows = datetimes.get_indexer(times)
        cols = assets.get_indexer(symbols)
        if np.any(rows < 0) or np.any(cols < 0):
            raise ValueError("Panel axes must contain every observed key")

        def scatter(column: Hashable) -> npt.NDArray[np.float64]:
            series = frame[column]
            if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_complex_dtype(series.dtype):
                raise ValueError(f"Expected real numeric values in column {column!r}")
            dtype = series.dtype.subtype if isinstance(series.dtype, pd.SparseDtype) else series.dtype
            dtype = np.dtype(getattr(dtype, "numpy_dtype", dtype))
            with np.errstate(over="ignore", invalid="ignore"):
                converted = series.to_numpy(dtype=np.float64, na_value=np.nan)
                if (dtype.kind in "iu" and dtype.itemsize >= 8) or dtype.itemsize > 8:
                    observed = series.notna().to_numpy()
                    original = series[observed].to_numpy(dtype=dtype)
                    as_float = converted[observed]
                    round_trip = as_float.astype(dtype)
                    # The rounded upper bound can be out of range even if the cast saturates.
                    out_of_range = dtype.kind in "iu" and np.any(
                        as_float >= float(int(np.iinfo(dtype).max) + 1)
                    )
                    if out_of_range or np.any(round_trip != original):
                        raise ValueError(f"Column {column!r} cannot be represented exactly as float64")
            values = np.full((len(datetimes), len(assets)), np.nan)
            values[rows, cols] = converted
            return values

        factor = scatter(factor_column)
        returns = {column: scatter(column) for column in columns if column != factor_column}
        finite_factor = np.isfinite(factor)
        valid = {horizon: finite_factor & np.isfinite(label) for horizon, label in returns.items()}
        return cls(datetimes, assets, factor, returns, valid, index, columns)

    def to_long(self) -> pd.DataFrame:
        """Gather the original observations in their original row and column order."""
        rows = self.datetimes.get_indexer(self._index.get_level_values(0))
        cols = self.assets.get_indexer(self._index.get_level_values(1))
        return pd.DataFrame(
            {
                column: self.returns.get(column, self.factor)[rows, cols]
                for column in self._columns
            },
            index=self._index,
            columns=self._columns,
        )
