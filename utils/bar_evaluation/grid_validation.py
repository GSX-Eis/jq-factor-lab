"""Validate split grids against the published calendar or the observed Bar intervals."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from backtest_and_analysis.time_series.evaluator import _calendar_mapping_hash
from backtest_and_analysis.time_series.labels import _grid
from utils.time_axis import BarSchedule, Frequency

from .schema import BarEvaluationIntegrityError

if TYPE_CHECKING:
    from utils.session_calendar import SessionCalendar

Slots = tuple[pd.DatetimeIndex, pd.DatetimeIndex]


def _input_axis(data_identity: Mapping[str, Any], field: str) -> pd.Index:
    values = data_identity.get(field)
    if (not isinstance(values, list) or not values
            or any(not isinstance(value, str) or not value.strip() for value in values)):
        raise BarEvaluationIntegrityError(f"summary.json data_identity.{field} must be a nonempty list of strings")
    return pd.Index(values)


def resolve_bucket_strategy(
    summary: dict[str, Any], calendars: dict[str | None, SessionCalendar] | None,
    *, instruments: pd.Index, frequencies: pd.Index,
) -> str | None:
    identity = summary["calendar_identity"]
    config = summary.get("config") or {}
    continuous = config.get("continuous_futures")
    nested = continuous.get("bucket_strategy") if isinstance(continuous, Mapping) else None
    configured = [value for value in (config.get("bucket_strategy"), nested) if value is not None]
    strategy = identity.get("bucket_strategy", configured[0] if configured else None)
    if strategy is None and configured:
        if instruments.empty and frequencies.empty:
            instruments = _input_axis(summary["data_identity"], "instruments")
            frequencies = _input_axis(summary["data_identity"], "frequencies")
        verified = matching_calendars(summary) if calendars is None else calendars
        used = {verified.get(instrument, verified.get(None)) for instrument in instruments}
        if None in used:
            raise BarEvaluationIntegrityError("Cannot verify the default bucket_strategy without a matching Bar calendar")
        try:
            defaults = {BarSchedule(calendar, frequency, view=identity.get("view"), bucket_strategy=(
                "session" if Frequency.parse(frequency).kind == "session" else None
            )).bucket_strategy for calendar in used for frequency in frequencies}
        except ValueError as exc:
            raise BarEvaluationIntegrityError(f"Cannot verify the default bucket_strategy: {exc}") from exc
        if len(defaults) == 1:
            strategy = defaults.pop()
    if any(value != strategy for value in configured):
        raise BarEvaluationIntegrityError("summary.json bucket_strategy declarations disagree")
    if strategy is not None and (not isinstance(strategy, str) or not strategy.strip()):
        raise BarEvaluationIntegrityError("summary.json bucket_strategy must be a nonempty string or null")
    return strategy


def matching_calendars(summary: dict[str, Any]) -> dict[str | None, SessionCalendar]:
    identity = summary["calendar_identity"]
    config = summary.get("config") or {}
    slot_artifact = any(field in summary["data_identity"] for field in ("slot_id", "slot_config_hash"))
    from utils.session_calendar import VERIFIED_XSHG, SessionCalendar, _date
    from .normalization import _canonical_json

    configured = slot_artifact or "analysis_mode" in config or "bucket_strategy" in config
    config_options = {key: config[key] for key in ("start_date", "end_date") if key in config}
    slot_config = config.get("slot_config")
    slot_calendar = slot_config.get("calendar") if isinstance(slot_config, Mapping) else None
    slot_options = slot_calendar.get("options") if isinstance(slot_calendar, Mapping) else None
    frozen_calendar_id = None
    frozen_options = {}
    if isinstance(slot_options, Mapping):
        if "calendar_id" in slot_options:
            frozen_calendar_id = slot_options["calendar_id"]
            if not isinstance(frozen_calendar_id, str) or not frozen_calendar_id.strip():
                raise BarEvaluationIntegrityError("config.slot_config.calendar.options.calendar_id must be a nonempty string")
        frozen_options = {
            field: slot_options[key]
            for field, key in (("start_date", "start"), ("end_date", "end"),
                               ("rth_reference_calendar", "rth_reference_calendar"),
                               ("daily_anchor", "daily_anchor"), ("custom_sessions", "custom_sessions"))
            if key in slot_options
        }
    # An explicitly empty options object still declares a calendar to verify.
    has_frozen_calendar = isinstance(slot_options, Mapping)
    strict_schedule = slot_artifact or has_frozen_calendar
    if config.get("continuous_futures") and not strict_schedule:
        return {}
    calendars = {}
    by_spec = {}
    specs = identity.get("calendars") or {None: identity}
    if not isinstance(specs, dict) or any(not isinstance(spec, dict) for spec in specs.values()):
        raise BarEvaluationIntegrityError("calendar_identity.calendars must contain calendar objects")
    if identity.get("calendars") and (strict_schedule or any("schedule" in spec for spec in specs.values())):
        if (any(not isinstance(spec.get(name), str) for spec in specs.values()
                for name in ("calendar_id", "calendar_hash"))
                or _calendar_mapping_hash(specs) != identity.get("calendar_hash")):
            raise BarEvaluationIntegrityError("calendar_identity.calendars disagree with the published calendar hash")
    for instrument, spec in specs.items():
        name, digest = spec.get("calendar_id"), spec.get("calendar_hash")
        explicit_schedule = "schedule" in spec or has_frozen_calendar
        if name is None:
            name = frozen_calendar_id
        if digest is None and has_frozen_calendar and instrument is None:
            digest = (summary.get("result_identity") or {}).get("calendar_hash")
        if not isinstance(name, str) or not name.strip() or not isinstance(digest, str):
            if strict_schedule and explicit_schedule:
                raise BarEvaluationIntegrityError("A published Slot schedule requires calendar_id and calendar_hash")
            continue
        frozen_name = frozen_calendar_id if frozen_calendar_id is not None else name
        if "schedule" in spec:
            options = spec["schedule"]
            fields = {"start_date", "end_date", "rth_reference_calendar"}
            if (not isinstance(options, dict) or not fields <= options.keys()
                    or options.keys() - fields - {"daily_anchor", "custom_sessions"}
                    or any(not isinstance(options[key], str) or not options[key] for key in fields)):
                raise BarEvaluationIntegrityError("calendar_identity.schedule must contain dates and an RTH calendar")
            for field in options.keys() & frozen_options.keys():
                published, frozen = options[field], frozen_options[field]
                if field in ("start_date", "end_date"):
                    try:
                        published, frozen = _date(published), _date(frozen)
                    except (ValueError, TypeError, OverflowError) as exc:
                        raise BarEvaluationIntegrityError(f"Invalid calendar_identity.schedule.{field}: {exc}") from exc
                if published != frozen:
                    raise BarEvaluationIntegrityError(f"calendar_identity.schedule.{field} disagrees with frozen Slot options")
            options = frozen_options | options
        elif isinstance(slot_options, Mapping):
            options = frozen_options
        elif configured and (name == "XSHG" or len(config_options) == 2):
            options = config_options
        else:
            continue
        if "daily_anchor" in options and not isinstance(options["daily_anchor"], str):
            raise BarEvaluationIntegrityError("calendar_identity.schedule.daily_anchor must be a string")
        if "custom_sessions" in options and not isinstance(options["custom_sessions"], Mapping):
            raise BarEvaluationIntegrityError("calendar_identity.schedule.custom_sessions must be an object")
        key = frozen_name, digest, _canonical_json(options)
        if key not in by_spec:
            try:
                calendar = (SessionCalendar.from_pmc(options["start_date"], options["end_date"])
                            if frozen_name == VERIFIED_XSHG else SessionCalendar(frozen_name, **options))
            except (ValueError, RuntimeError, KeyError, TypeError):
                calendar = None
            by_spec[key] = calendar if calendar is not None and calendar.calendar_hash == digest else None
        if by_spec[key] is not None:
            if frozen_name != name and by_spec[key].calendar_id != name:
                raise BarEvaluationIntegrityError("calendar_identity.calendar_id disagrees with frozen Slot options")
            calendars[instrument] = by_spec[key]
        elif explicit_schedule and (strict_schedule or "custom_sessions" in options or "daily_anchor" in options):
            raise BarEvaluationIntegrityError(f"Cannot reconstruct calendar_id={name!r} with the published calendar_hash")
    return calendars


@dataclass(frozen=True)
class LabelGrid:
    slots: tuple[Slots, ...]
    scheduled: bool
    nominal_width: pd.Timedelta
    session_or_week: bool

    def check(self, *, time_series: bool, split: str, start: pd.Timestamp, end: pd.Timestamp,
              train_end: pd.Timestamp | None, oos_start: pd.Timestamp | None,
              grid_size: int, observed: pd.DatetimeIndex, field: str) -> pd.DatetimeIndex | None:
        if self.scheduled:
            expected = pd.DatetimeIndex([], tz="UTC")
            for starts, ends in self.slots:
                mask = ends <= train_end if split == "train" else ends >= oos_start if split == "oos" else slice(None)
                expected = expected.union((starts if time_series else ends)[mask])
            if len(expected) and (start, end, grid_size) == (expected[0], expected[-1], len(expected)):
                return expected
            raise BarEvaluationIntegrityError(
                f"{field}.grid_start_utc/grid_end_utc or grid_size disagrees with the Bar schedule"
            )

        # A historical export may lack enough metadata to reconstruct its calendar.
        # A missing Bar must fit in an observed gap and satisfy the split by its end clock.
        endpoints = {start, end}
        lower, upper = start.value, end.value
        if split == "train":
            upper = min(upper, train_end.value - (1 if time_series else 0))
        elif split == "oos" and not time_series:
            lower = max(lower, oos_start.value)

        width = self.nominal_width.value
        capacity = len(observed)
        for starts, ends in self.slots:
            axis = starts if time_series else ends
            endpoints = {t for t in endpoints if t not in axis}
            left, right = ends[:-1], starts[1:]
            gaps = (left < right) & (left <= end) & (right >= start)
            if split == "oos" and time_series:
                gaps &= right >= oos_start
            for a, b in zip(left[gaps], right[gaps], strict=True):
                first = max(lower, a.value + (0 if time_series else 1))
                last = min(upper, b.value - (1 if time_series else 0))
                if first > last:
                    continue
                endpoints = {t for t in endpoints if not first <= t.value <= last}
                # Round up for a partial Bar. A clipped range can include one Bar at either edge.
                room = min(b.value - a.value, last - first + width)
                capacity += (room + width - 1) // width
        if endpoints:
            raise BarEvaluationIntegrityError(f"{field}.grid_start_utc/grid_end_utc is outside possible Bar slots")
        if grid_size > capacity:
            raise BarEvaluationIntegrityError(f"{field}.grid_size exceeds the available Bar slots")
        return observed.sort_values() if grid_size == len(observed) else None


def _scheduled_slots(calendar: SessionCalendar, frequency: str, starts: pd.DatetimeIndex,
                     ends: pd.DatetimeIndex, view: str, strategy: str | None) -> Slots:
    parsed = Frequency.parse(frequency)
    if parsed.kind not in ("session", "week"):
        grid, availability, _ = _grid(calendar, frequency, starts[0], starts[-1], view, strategy)
        candidates = ((grid, availability),)
    else:
        schedule = BarSchedule(calendar, parsed, view=view,
                               bucket_strategy=strategy or ("session" if parsed.kind == "session" else None))
        # A weekly Bar opens in its first session and is emitted by its last session.
        # Include the entire boundary weeks and the next local date for overnight opens.
        first, last = (stamp.tz_convert(calendar.timezone).tz_localize(None).normalize()
                       for stamp in (starts[0], ends[-1]))
        padding = pd.Timedelta(days=7 if parsed.kind == "week" else 1)
        dates = calendar.sessions(max(first - padding, calendar.start), min(last + padding, calendar.end))
        opening, sessions, availability = [], [], []
        for day in dates:
            session = calendar.session_on(day.strftime("%Y-%m-%d"), view)
            slots = schedule.for_session(session) if session is not None else None
            if slots is not None and not slots.empty:
                opening.append(slots.start_utc.iloc[0])
                sessions.append(session)
                availability.append(slots.end_utc.iloc[0])
        scheduled_ends = pd.DatetimeIndex(availability, tz="UTC")
        anchors = map(calendar._session_anchor, sessions)
        candidates = ((pd.DatetimeIndex(times, tz="UTC"), scheduled_ends) for times in (opening, scheduled_ends, anchors))
    for grid, available in candidates:
        keep = (grid >= starts[0]) & (grid <= starts[-1])
        grid, available = grid[keep], available[keep]
        positions = grid.get_indexer(starts)
        if (positions >= 0).all() and available.take(positions).equals(ends):
            return grid, available
    raise BarEvaluationIntegrityError("per_bar_labels.csv clocks disagree with the published Bar schedule")


def build_label_grid(
    summary: dict[str, Any], labels: pd.DataFrame, calendars: dict[str | None, SessionCalendar],
    *, bucket_strategy: str | None,
) -> LabelGrid:
    config = summary.get("config") or {}
    view = summary["calendar_identity"].get("view")
    frequency = labels["frequency"].iloc[0]
    session_or_week = Frequency.parse(frequency).kind in ("session", "week")
    continuous = config.get("continuous_futures")
    view_config = continuous if isinstance(continuous, dict) else config
    if "view" in view_config and view != view_config["view"]:
        raise BarEvaluationIntegrityError("calendar_identity.view disagrees with the published Bar configuration")
    observed, scheduled = [], []
    for instrument, group in labels.groupby("instrument_id"):
        times = group.sort_values("signal_timestamp")
        starts, ends = pd.DatetimeIndex(times["signal_timestamp"]), pd.DatetimeIndex(times["signal_available_at"])
        calendar = calendars.get(instrument, calendars.get(None))
        allow_close_anchor = session_or_week and calendar is not None
        backward = ends < starts if allow_close_anchor else ends <= starts
        if backward.any() or (ends[:-1] > starts[1:]).any():
            raise BarEvaluationIntegrityError("per_bar_labels.csv contains overlapping Bar slots")
        observed.append((starts, ends))
        if calendar is None or (calendar.is_domestic_futures and bucket_strategy is None and not session_or_week):
            continue
        if view not in ("ETH", "RTH"):
            raise BarEvaluationIntegrityError("calendar_identity.view must be ETH or RTH")
        try:
            scheduled.append(_scheduled_slots(calendar, frequency, starts, ends, view, bucket_strategy))
        except BarEvaluationIntegrityError:
            raise
        except (ValueError, RuntimeError, TypeError) as exc:
            raise BarEvaluationIntegrityError(f"Cannot reconstruct the published Bar schedule: {exc}") from exc
    complete_schedule = len(scheduled) == len(observed)
    return LabelGrid(tuple(scheduled if complete_schedule else observed), complete_schedule,
                     Frequency.contract_duration(frequency), session_or_week)
