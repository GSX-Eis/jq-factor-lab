from __future__ import annotations

from copy import deepcopy

import pytest

from factor_synthesis.research_split import apply_research_split


def _config(active_split: str = "development") -> dict:
    return {
        "analysis_period": {
            "start_date": "2018-01-01",
            "end_date": "2023-12-31",
        },
        "research_protocol": {
            "campaign_id": "wind_micro_2018_2025_v1",
            "active_split": active_split,
            "splits": {
                "development": {
                    "role": "development_is",
                    "start_date": "2018-01-01",
                    "end_date": "2023-12-31",
                },
                "locked_validation": {
                    "role": "research_oos",
                    "start_date": "2024-01-01",
                    "end_date": "2025-12-31",
                },
            },
            "future_blind": {
                "start_date": "2026-07-06",
                "minimum_months": 12,
            },
        },
    }


def test_development_split_resolves_exact_is_window() -> None:
    config = _config()

    resolved = apply_research_split(config)

    assert resolved["analysis_period"] == {
        "start_date": "2018-01-01",
        "end_date": "2023-12-31",
    }
    assert resolved["research_protocol"]["resolved_role"] == "development_is"


def test_locked_validation_resolves_exact_oos_window() -> None:
    config = _config("locked_validation")
    config["analysis_period"] = {
        "start_date": "2024-01-01",
        "end_date": "2025-12-31",
    }

    resolved = apply_research_split(config)

    assert resolved["analysis_period"]["start_date"] == "2024-01-01"
    assert resolved["analysis_period"]["end_date"] == "2025-12-31"
    assert resolved["research_protocol"]["resolved_role"] == "research_oos"


def test_split_rejects_analysis_window_drift() -> None:
    config = _config()
    config["analysis_period"]["end_date"] = "2025-12-31"

    with pytest.raises(ValueError, match="analysis_period must exactly match"):
        apply_research_split(config)


def test_split_rejects_overlap_and_future_blind_reuse() -> None:
    overlap = _config()
    overlap["research_protocol"]["splits"]["locked_validation"]["start_date"] = (
        "2023-01-01"
    )
    with pytest.raises(ValueError, match="must end before locked_validation starts"):
        apply_research_split(overlap)

    reused = _config()
    reused["research_protocol"]["future_blind"]["start_date"] = "2025-07-01"
    with pytest.raises(
        ValueError, match="future_blind must start after locked_validation"
    ):
        apply_research_split(reused)


def test_legacy_config_without_protocol_is_unchanged() -> None:
    config = {"analysis_period": {"start_date": "2022-01-01", "end_date": "2025-12-31"}}
    original = deepcopy(config)

    assert apply_research_split(config) == original
