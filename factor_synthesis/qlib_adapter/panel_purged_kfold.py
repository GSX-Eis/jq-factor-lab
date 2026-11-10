# -*- coding: utf-8 -*-
"""Panel CV using actual label exit dates, with no inferred weekday embargo."""
from collections.abc import Mapping
from numbers import Integral
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd


class LabelIsolationError(RuntimeError):
    """Label availability cannot be proved; the entire run must fail."""


def resolve_label_exit_dates(
    index: pd.MultiIndex,
    label_exit_dates,
    *,
    trading_calendar=None,
    label_period=None,
    calendar_source=None,
) -> pd.Series:
    """Expand producer date→date metadata onto the exact feature keys.

    Calendar fallback requires an identified, complete session sequence and an
    effective period already including execution lag. An incomplete explicit
    mapping is an error, never a request to fill its holes from the calendar.
    """
    if not isinstance(index, pd.MultiIndex) or index.nlevels != 2 or not index.is_unique:
        raise LabelIsolationError("label isolation requires unique two-level feature keys")
    try:
        entries = pd.DatetimeIndex(index.get_level_values(0)).normalize()
        if entries.hasnans or entries.tz is not None:
            raise ValueError("invalid entry dates")
        if label_exit_dates is not None:
            if not isinstance(label_exit_dates, (Mapping, pd.Series)):
                raise ValueError("expected producer date-to-date mapping")
            mapping = {}
            for key, value in label_exit_dates.items():
                day, exit_day = pd.Timestamp(key), pd.Timestamp(value)
                if pd.isna(day) or pd.isna(exit_day) or day.tz is not None or exit_day.tz is not None:
                    raise ValueError("invalid mapped date")
                day, exit_day = day.normalize(), exit_day.normalize()
                if day in mapping:
                    raise ValueError("duplicate normalized entry date")
                mapping[day] = exit_day
            if any(day not in mapping for day in entries):
                raise ValueError("missing label exit for required entry date")
            exits = pd.DatetimeIndex([mapping[day] for day in entries])
        else:
            if not isinstance(calendar_source, str) or not calendar_source.strip():
                raise ValueError("verified trading calendar source is required")
            if isinstance(label_period, bool) or not isinstance(label_period, Integral) or label_period <= 0:
                raise ValueError("positive effective label period including execution lag is required")
            if trading_calendar is None:
                raise ValueError("complete trading calendar is required")
            calendar = pd.DatetimeIndex(trading_calendar).normalize()
            if (calendar.hasnans or calendar.tz is not None or not calendar.is_unique
                    or not calendar.is_monotonic_increasing):
                raise ValueError("calendar sessions must be valid, unique and increasing")
            positions = calendar.get_indexer(entries)
            if (positions < 0).any() or (positions + label_period >= len(calendar)).any():
                raise ValueError("calendar does not cover entry and label exit sessions")
            exits = calendar.take(positions + label_period)
        if (exits <= entries).any():
            raise ValueError("label exit must be strictly later than entry")
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise LabelIsolationError(f"cannot resolve label exits: {exc}") from exc
    return pd.Series(exits, index=index, name="label_exit_date")


class PanelPurgedKFold:
    """Expanding date folds; keep only training labels exiting before validation.

    ``embargo_td`` remains accepted for existing configuration compatibility.
    It cannot substitute for actual exits, and does not shift validation dates.
    """

    def __init__(self, n_splits: int = 5, embargo_td: int = 5):
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if embargo_td < 0:
            raise ValueError(f"embargo_td must be >= 0, got {embargo_td}")
        self.n_splits = n_splits
        self.embargo_td = embargo_td

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        *,
        label_exit_dates=None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Validate every fold before yielding; the final fold includes the tail."""
        if groups is None or label_exit_dates is None:
            raise LabelIsolationError("PanelPurgedKFold requires date groups and aligned label exits")
        try:
            dates = pd.DatetimeIndex(groups).normalize()
            exits = pd.DatetimeIndex(label_exit_dates).normalize()
            if len(dates) != len(X) or len(exits) != len(X):
                raise ValueError("date/exit lengths must equal feature rows")
            if dates.hasnans or exits.hasnans or dates.tz is not None or exits.tz is not None:
                raise ValueError("invalid date/exit vector")
            if (exits <= dates).any():
                raise ValueError("label exit must be strictly later than entry")
        except (ValueError, TypeError) as exc:
            raise LabelIsolationError(f"invalid CV metadata: {exc}") from exc
        unique_dates = dates.unique().sort_values()
        q = len(unique_dates) // (self.n_splits + 1)
        if q < 1:
            raise LabelIsolationError(f"fold=1: need at least {self.n_splits + 1} dates")
        order = np.argsort(dates, kind="stable")
        folds = []
        for fold in range(self.n_splits):
            start = (fold + 1) * q
            end = len(unique_dates) if fold == self.n_splits - 1 else (fold + 2) * q
            boundary = unique_dates[start]
            train = order[((dates < boundary) & (exits < boundary))[order]]
            valid = order[dates.isin(unique_dates[start:end])[order]]
            if not len(train) or not len(valid):
                raise LabelIsolationError(
                    f"fold={fold + 1}: empty strict split at {boundary.date()} "
                    f"(train={len(train)}, valid={len(valid)})"
                )
            folds.append((train, valid))
        yield from folds

    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> int:
        """Return the configured number of folds; empty folds are never skipped."""
        return self.n_splits
