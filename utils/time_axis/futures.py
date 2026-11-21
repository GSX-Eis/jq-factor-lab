"""Explicit product calendars; an unknown root must never become a day-only exchange."""
import json
from pathlib import Path

DEFAULT_FUTURES_MAP = Path(__file__).resolve().parents[2] / 'config/futures_calendar_map.json'
RECOMMENDED_DOMESTIC_FUTURES_BUCKET_STRATEGY = 'segment_anchored'


def futures_calendar(product_root: str, *, path: str | Path = DEFAULT_FUTURES_MAP) -> dict[str, str]:
    mapping = json.loads(Path(path).read_text())
    if not isinstance(product_root, str) or product_root.upper() not in mapping:
        raise ValueError(f'Missing futures calendar for product_root {product_root!r}')
    entry = mapping[product_root.upper()]
    if (not isinstance(entry, dict) or any(not isinstance(entry.get(key), str) or not entry[key]
            for key in ('calendar_id', 'rth_reference_calendar'))):
        raise ValueError(f'Invalid futures calendar mapping for {product_root!r}')
    return dict(entry)


def recommended_bucket_strategy(product_root: str | None = None,
                                *, path: str | Path = DEFAULT_FUTURES_MAP) -> str:
    """Return the project-recommended explicit bucket_strategy for domestic futures."""
    if product_root is not None:
        futures_calendar(product_root, path=path)
    return RECOMMENDED_DOMESTIC_FUTURES_BUCKET_STRATEGY
