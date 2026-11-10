"""
Calendar skip rules: whole months and month-day windows.

``calendar_filters.skip_months`` historically accepted whole months
(``[1, 4, 12]``). This module generalises the same config key to also accept
month-day window strings so mechanism-anchored windows (e.g. forecast deadline
1/31, annual-report deadline 4/30) can replace whole-month flat periods:

    "skip_months": ["01-16:02-05", "04-16:04-30"]

Rules may be mixed: ``[12, "04-16:04-30"]``. Windows are inclusive on both
ends and may wrap the year boundary (``"12-20:01-10"``).

All consumers share one parser (``parse_skip_rules``) and one vectorised
predicate (``skip_mask``) so training / evaluation / backtest apply identical
calendars.
"""

from collections.abc import Iterable, Mapping
from datetime import date
from numbers import Integral
import re
from typing import Any, Optional, Set, Union

import numpy as np
import pandas as pd

# A rule is either a whole month (int 1-12) or a canonical window string
# "MM-DD:MM-DD" (zero-padded, inclusive both ends).
SkipRule = Union[int, str]
SkipRules = Set[SkipRule]

_WINDOW_RE = re.compile(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*:\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$")


def _parse_window(window: str) -> Optional[tuple]:
    """Return ((sm, sd), (em, ed)) for a window string, or None if invalid."""
    match = _WINDOW_RE.match(window)
    if not match:
        return None
    sm, sd, em, ed = (int(g) for g in match.groups())
    try:
        # A leap year allows recurring February 29 windows.
        date(2000, sm, sd)
        date(2000, em, ed)
    except ValueError:
        return None
    return (sm, sd), (em, ed)


def parse_skip_rules(raw: Optional[Iterable[Any]]) -> SkipRules:
    """Parse the ``skip_months`` config list into a normalised rule set.

    Accepts ints/int-strings (whole months, 1-12) and window strings
    "MM-DD:MM-DD". Invalid containers or entries raise; floats and booleans are rejected.
    """
    if raw is None:
        return set()
    if isinstance(raw, (str, bytes, Mapping)) or not isinstance(raw, Iterable):
        raise TypeError("skip_months must be an iterable of month integers or window strings")
    rules: SkipRules = set()
    for index, item in enumerate(raw):
        if isinstance(item, str):
            parsed = _parse_window(item)
            if parsed is not None:
                (sm, sd), (em, ed) = parsed
                rules.add(f"{sm:02d}-{sd:02d}:{em:02d}-{ed:02d}")
                continue
        if isinstance(item, bool) or not isinstance(item, (Integral, str)):
            raise ValueError(f"Invalid skip_months[{index}]: {item!r}")
        try:
            month = int(item)
        except ValueError as exc:
            raise ValueError(f"Invalid skip_months[{index}]: {item!r}") from exc
        if not 1 <= month <= 12:
            raise ValueError(f"Invalid skip_months[{index}]: {item!r}; month must be 1-12")
        rules.add(month)
    return rules


def skip_mask(dates: Any, rules: Optional[SkipRules]) -> np.ndarray:
    """Vectorised predicate: True where a date falls inside any skip rule."""
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    mask = np.zeros(len(idx), dtype=bool)
    if not rules or len(idx) == 0:
        return mask
    months = sorted(r for r in rules if isinstance(r, int))
    windows = sorted(r for r in rules if isinstance(r, str))
    if months:
        mask |= np.asarray(idx.month.isin(months))
    if windows:
        key = idx.month.values * 100 + idx.day.values
        for window in windows:
            parsed = _parse_window(window)
            if parsed is None:
                continue
            (sm, sd), (em, ed) = parsed
            lo, hi = sm * 100 + sd, em * 100 + ed
            if lo <= hi:
                mask |= (key >= lo) & (key <= hi)
            else:  # wraps the year boundary, e.g. 12-20:01-10
                mask |= (key >= lo) | (key <= hi)
    return mask


def date_in_skip(date: Any, rules: Optional[SkipRules]) -> bool:
    """Single-date convenience wrapper over ``skip_mask``."""
    if not rules:
        return False
    return bool(skip_mask([date], rules)[0])


def describe_skip_rules(rules: Optional[SkipRules]) -> str:
    """Stable human-readable description for logs/headers (mixed sets sort-safe)."""
    if not rules:
        return "none"
    months = sorted(r for r in rules if isinstance(r, int))
    windows = sorted(r for r in rules if isinstance(r, str))
    parts = [str(m) for m in months] + windows
    return "[" + ", ".join(parts) + "]"
