"""Cross-market standard signal and target-position export DeployTarget."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from factor_synthesis.strategy_bundle import StrategyBundle
from targets.contracts import (
    AlignmentReport,
    CompatibilityReport,
    DeployArtifactManifest,
    StrategyBundleSource,
)
from targets.signal_export.exporter import (
    TARGET_POSITIONS_JSONL_FILENAME,
    TARGET_POSITIONS_PARQUET_FILENAME,
    TRADE_ORDERS_JSONL_FILENAME,
    export_signal_artifacts,
)
from targets.signal_export.schema import (
    SIGNAL_EXPORT_TARGET_ID,
    check_bundle_compatibility,
)
from targets.signal_export.verifier import verify_signal_export_alignment


class SignalExportTarget:
    """Universal cross-market DeployTarget exporting target positions and differential trade orders."""

    @property
    def target_id(self) -> str:
        return SIGNAL_EXPORT_TARGET_ID

    def check_compatibility(
        self,
        bundle: StrategyBundle | StrategyBundleSource | str | Path,
        *,
        bundle_dir: str | Path | None = None,
    ) -> CompatibilityReport:
        return check_bundle_compatibility(bundle, bundle_dir=bundle_dir, target_id=self.target_id)

    def export(
        self,
        bundle: StrategyBundle | StrategyBundleSource | str | Path,
        output_dir: str | Path,
        *,
        target_positions: pd.DataFrame | Sequence[Mapping[str, Any]] | None = None,
        current_positions: Mapping[str, Any] | pd.DataFrame | Sequence[Mapping[str, Any]] | None = None,
        bundle_dir: str | Path | None = None,
        formats: Sequence[str] = ("parquet", "jsonl"),
        created_at_utc: str | None = None,
    ) -> DeployArtifactManifest:
        return export_signal_artifacts(
            bundle,
            output_dir,
            target_positions=target_positions,
            current_positions=current_positions,
            bundle_dir=bundle_dir,
            formats=formats,
            created_at_utc=created_at_utc,
            target_id=self.target_id,
        )

    def verify_alignment(
        self,
        bundle: StrategyBundle | StrategyBundleSource | str | Path,
        artifact: DeployArtifactManifest | str | Path,
        *,
        reference_positions: pd.DataFrame | Sequence[Mapping[str, Any]] | None = None,
        output_dir: str | Path | None = None,
        bundle_dir: str | Path | None = None,
    ) -> AlignmentReport:
        return verify_signal_export_alignment(
            bundle,
            artifact,
            reference_positions=reference_positions,
            output_dir=output_dir,
            bundle_dir=bundle_dir,
            target_id=self.target_id,
        )


__all__ = [
    "SIGNAL_EXPORT_TARGET_ID",
    "SignalExportTarget",
    "TARGET_POSITIONS_JSONL_FILENAME",
    "TARGET_POSITIONS_PARQUET_FILENAME",
    "TRADE_ORDERS_JSONL_FILENAME",
]
