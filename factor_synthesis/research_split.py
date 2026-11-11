"""Fail-closed IS/OOS date resolution for synthesis experiment configs."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
from typing import Any, Mapping


def _window(splits: Mapping[str, Any], name: str) -> tuple[date, date, str]:
    raw = splits.get(name)
    if not isinstance(raw, Mapping):
        raise ValueError(f"research_protocol.splits.{name} must be an object")
    try:
        start = date.fromisoformat(str(raw["start_date"]))
        end = date.fromisoformat(str(raw["end_date"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"research_protocol.splits.{name} requires ISO start_date/end_date"
        ) from exc
    if start > end:
        raise ValueError(
            f"research_protocol.splits.{name} start_date must not exceed end_date"
        )
    return start, end, str(raw.get("role", name) or name)


def apply_research_split(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the research protocol and resolve its active output window.

    Legacy configs without ``research_protocol`` keep their existing behavior.
    Protocol-aware configs fail closed when the selected split and
    ``analysis_period`` disagree, so a full-window backtest cannot silently be
    labelled as IS or OOS.
    """

    protocol = config.get("research_protocol")
    if protocol is None:
        return config
    if not isinstance(protocol, Mapping):
        raise ValueError("research_protocol must be an object")
    if not str(protocol.get("campaign_id", "")).strip():
        raise ValueError("research_protocol.campaign_id is required")

    active_split = str(protocol.get("active_split", "")).strip()
    if active_split not in {"development", "locked_validation"}:
        raise ValueError(
            "research_protocol.active_split must be development or locked_validation"
        )
    splits = protocol.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("research_protocol.splits must be an object")

    development_start, development_end, _ = _window(splits, "development")
    validation_start, validation_end, _ = _window(splits, "locked_validation")
    if development_end >= validation_start:
        raise ValueError("development must end before locked_validation starts")

    future_blind = protocol.get("future_blind")
    if not isinstance(future_blind, Mapping):
        raise ValueError("research_protocol.future_blind must be an object")
    try:
        future_start = date.fromisoformat(str(future_blind["start_date"]))
        minimum_months = int(future_blind["minimum_months"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "future_blind requires ISO start_date and integer minimum_months"
        ) from exc
    if future_start <= validation_end:
        raise ValueError("future_blind must start after locked_validation ends")
    if minimum_months < 12:
        raise ValueError("future_blind.minimum_months must be at least 12")

    active_start, active_end, active_role = _window(splits, active_split)
    analysis_period = config.get("analysis_period")
    if not isinstance(analysis_period, Mapping):
        raise ValueError("analysis_period must be an object")
    actual = (
        str(analysis_period.get("start_date", "")),
        str(analysis_period.get("end_date", "")),
    )
    expected = (active_start.isoformat(), active_end.isoformat())
    if actual != expected:
        raise ValueError(
            "analysis_period must exactly match research_protocol.active_split: "
            f"actual={actual}, expected={expected}"
        )

    resolved = deepcopy(config)
    resolved["research_protocol"]["resolved_role"] = active_role
    return resolved


__all__ = ["apply_research_split"]
