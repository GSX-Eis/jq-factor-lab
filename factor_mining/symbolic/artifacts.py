"""Artifact writing helpers for symbolic factor mining."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_hall_of_fame(path: str | Path, candidates: Iterable[Any]) -> None:
    rows = []
    for rank, candidate in enumerate(candidates, start=1):
        rows.append(
            {
                "rank": rank,
                "expression": candidate.expression.to_string(),
                "score": candidate.score,
                "complexity": candidate.complexity,
                "train_rank_ic_mean": candidate.train_metrics.get("rank_ic_mean"),
                "valid_rank_ic_mean": candidate.valid_metrics.get("rank_ic_mean"),
                "valid_rank_ic_ir": candidate.valid_metrics.get("rank_ic_ir"),
                "valid_positive_ratio": candidate.valid_metrics.get("positive_ratio"),
            }
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(target, index=False)


def write_summary(path: str | Path, payload: Dict[str, Any]) -> None:
    write_json(path, payload)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class ProgressWriter:
    """Append-only JSONL progress sink for long symbolic mining runs."""

    def __init__(self, path: str | Path, *, run_id: str) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self.started_monotonic = time.perf_counter()
        self.sequence = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **payload: Any) -> None:
        self.sequence += 1
        record = {
            "seq": self.sequence,
            "event": event,
            "run_id": self.run_id,
            "elapsed_sec": round(time.perf_counter() - self.started_monotonic, 6),
            "wall_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(_json_safe(record), ensure_ascii=False, sort_keys=True)
                + "\n"
            )
