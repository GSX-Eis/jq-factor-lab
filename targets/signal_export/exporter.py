"""Signal and target-position exporter for targets/signal_export."""
from __future__ import annotations

import fcntl
import json
import math
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from factor_synthesis.strategy_bundle import (
    MANIFEST_FILENAME,
    StrategyBundle,
    sha256_file,
)
from targets.contracts import (
    DEPLOY_MANIFEST_FILENAME,
    TARGET_ARTIFACT_INTEGRITY_FAILED,
    TARGET_BUNDLE_INVALID,
    TARGET_SIGNAL_SOURCE_MISSING,
    DeployArtifactEntry,
    DeployArtifactManifest,
    DeployTargetError,
    StrategyBundleSource,
    resolve_path_checked,
    resolve_verified_bundle_with_dir,
    utc_now_iso,
    validate_utc_iso_timestamp,
    verify_deploy_manifest,
    write_deploy_manifest,
)
from targets.signal_export.schema import (
    ORDER_OUTPUT_COLUMNS,
    POSITION_BASE_OUTPUT_COLUMNS,
    POSITION_OPTIONAL_COLUMNS,
    SIGNAL_EXPORT_TARGET_ID,
    check_bundle_compatibility,
    derive_order_action,
    derive_side,
    format_utc_timestamp,
    load_bundle_signal_positions,
    normalize_current_positions,
    normalize_formats,
    normalize_target_positions,
)

TARGET_POSITIONS_PARQUET_FILENAME = "target_positions.parquet"
TARGET_POSITIONS_JSONL_FILENAME = "target_positions.jsonl"
TRADE_ORDERS_JSONL_FILENAME = "trade_orders.jsonl"


def _validate_managed_destination(root: Path, rel_name: str) -> None:
    """Reject managed destination paths that escape output_dir or are non-regular file types."""
    candidate = root / rel_name
    try:
        st = os.lstat(candidate)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise DeployTargetError(
            f"Failed to inspect destination path {rel_name!r}: {exc}",
            code=TARGET_ARTIFACT_INTEGRITY_FAILED,
        ) from exc

    if stat.S_ISLNK(st.st_mode):
        resolved = resolve_path_checked(
            candidate,
            label=f"destination path {rel_name!r}",
            code=TARGET_ARTIFACT_INTEGRITY_FAILED,
        )
        if not resolved.is_relative_to(root):
            raise DeployTargetError(
                f"Destination path {rel_name!r} is a symlink escaping output directory",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
        try:
            target_exists = resolved.exists()
            target_is_file = resolved.is_file()
        except OSError as exc:
            raise DeployTargetError(
                f"Failed to inspect resolved destination path {rel_name!r}: {exc}",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            ) from exc
        if target_exists and not target_is_file:
            raise DeployTargetError(
                f"Destination path {rel_name!r} is a symlink to an unsupported file type",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
        return

    if stat.S_ISDIR(st.st_mode):
        raise DeployTargetError(
            f"Destination path {rel_name!r} is a directory; expected a regular file",
            code=TARGET_ARTIFACT_INTEGRITY_FAILED,
        )
    if not stat.S_ISREG(st.st_mode):
        raise DeployTargetError(
            f"Destination path {rel_name!r} has unsupported file type",
            code=TARGET_ARTIFACT_INTEGRITY_FAILED,
        )


def _collect_protected_bundle_paths(
    bundle: StrategyBundle,
    bundle_dir: Path,
) -> set[Path]:
    """Return resolved paths of the StrategyBundle manifest and all referenced files."""
    protected = {
        resolve_path_checked(
            bundle_dir / MANIFEST_FILENAME,
            label="StrategyBundle manifest",
            code=TARGET_ARTIFACT_INTEGRITY_FAILED,
        )
    }
    for ref in bundle.model_artifacts:
        protected.add(
            resolve_path_checked(
                bundle_dir / ref.path,
                label=f"StrategyBundle artifact {ref.path!r}",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
        )
    protected.add(
        resolve_path_checked(
            bundle_dir / bundle.backtest_returns.path,
            label="StrategyBundle backtest_returns",
            code=TARGET_ARTIFACT_INTEGRITY_FAILED,
        )
    )
    if bundle.backtest_nav is not None:
        protected.add(
            resolve_path_checked(
                bundle_dir / bundle.backtest_nav.path,
                label="StrategyBundle backtest_nav",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
        )
    protected.add(
        resolve_path_checked(
            bundle_dir / bundle.execution_result_ref,
            label="StrategyBundle execution_result_ref",
            code=TARGET_ARTIFACT_INTEGRITY_FAILED,
        )
    )
    return protected


def _ensure_no_bundle_output_overlap(
    bundle: StrategyBundle,
    bundle_dir: Path | None,
    output_root: Path,
    destination_names: Sequence[str],
) -> None:
    """Reject exports whose resolved output paths would overwrite or unlink verified bundle files."""
    if bundle_dir is None:
        return
    protected_paths = _collect_protected_bundle_paths(bundle, bundle_dir)
    for rel_name in destination_names:
        dest = output_root / rel_name
        resolved_dest = resolve_path_checked(
            dest,
            label=f"output destination {rel_name!r}",
            code=TARGET_ARTIFACT_INTEGRITY_FAILED,
        )
        if {dest, resolved_dest} & protected_paths:
            raise DeployTargetError(
                f"output_dir destination {dest} overlaps protected StrategyBundle path",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )


def _build_position_export_frame(
    normalized_positions: pd.DataFrame,
    bundle: StrategyBundle,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    present_optional = [
        col for col in POSITION_OPTIONAL_COLUMNS if col in normalized_positions.columns
    ]
    sides = [derive_side(float(w)) for w in normalized_positions["target_weight"].tolist()]
    export_df = pd.DataFrame(
        {
            "timestamp_utc": normalized_positions["timestamp_utc"],
            "session_id": normalized_positions["session_id"].astype(str),
            "instrument_id": normalized_positions["instrument_id"].astype(str),
            "target_weight": normalized_positions["target_weight"].astype(float),
            "side": sides,
            "slot_id": [bundle.slot_id] * len(normalized_positions),
            "strategy_id": [bundle.strategy_id] * len(normalized_positions),
            "frequency": [bundle.frequency] * len(normalized_positions),
            "bundle_hash": [bundle.bundle_hash] * len(normalized_positions),
        }
    )
    for col in present_optional:
        export_df[col] = normalized_positions[col]

    ordered_cols = [*POSITION_BASE_OUTPUT_COLUMNS, *present_optional]
    export_df = export_df.loc[:, ordered_cols]

    jsonl_records: list[dict[str, Any]] = []
    for row in export_df.to_dict(orient="records"):
        record: dict[str, Any] = {
            "timestamp_utc": format_utc_timestamp(pd.Timestamp(row["timestamp_utc"])),
            "session_id": str(row["session_id"]),
            "instrument_id": str(row["instrument_id"]),
            "target_weight": 0.0 if float(row["target_weight"]) == 0.0 else float(row["target_weight"]),
            "side": str(row["side"]),
            "slot_id": str(row["slot_id"]),
            "strategy_id": str(row["strategy_id"]),
            "frequency": str(row["frequency"]),
            "bundle_hash": str(row["bundle_hash"]),
        }
        for opt_col in present_optional:
            if opt_col == "lot_size":
                record[opt_col] = int(row[opt_col])
            else:
                val = float(row[opt_col])
                record[opt_col] = 0.0 if val == 0.0 else val
        jsonl_records.append(record)

    return export_df, jsonl_records


def _build_trade_orders(
    normalized_positions: pd.DataFrame,
    bundle: StrategyBundle,
    initial_positions: Mapping[str, float],
) -> tuple[list[dict[str, Any]], int]:
    state: dict[str, float] = dict(initial_positions)
    orders: list[dict[str, Any]] = []
    rebalance_count = 0

    for ts_utc, group in normalized_positions.groupby("timestamp_utc", sort=True):
        rebalance_count += 1
        ts_str = format_utc_timestamp(pd.Timestamp(ts_utc))
        session_id = str(group["session_id"].iloc[0])
        targets_at_ts = {
            str(inst): (0.0 if float(w) == 0.0 else float(w))
            for inst, w in zip(group["instrument_id"], group["target_weight"])
        }
        active_held = {inst for inst, prev_w in state.items() if prev_w != 0.0}
        ordered_instruments = sorted(set(targets_at_ts) | active_held)

        for inst in ordered_instruments:
            prev_w = 0.0 if state.get(inst, 0.0) == 0.0 else float(state.get(inst, 0.0))
            target_w = targets_at_ts.get(inst, 0.0)
            delta_w = target_w - prev_w
            if not math.isfinite(delta_w):
                raise DeployTargetError(
                    f"Order delta_weight for {inst!r} at {ts_str} exceeds finite float range",
                    code=TARGET_BUNDLE_INVALID,
                )
            delta_w = 0.0 if delta_w == 0.0 else delta_w
            action = derive_order_action(prev_w, target_w)
            orders.append(
                {
                    "timestamp_utc": ts_str,
                    "session_id": session_id,
                    "instrument_id": inst,
                    "prev_weight": prev_w,
                    "target_weight": target_w,
                    "delta_weight": delta_w,
                    "action": action,
                    "slot_id": bundle.slot_id,
                    "strategy_id": bundle.strategy_id,
                    "bundle_hash": bundle.bundle_hash,
                }
            )
            state[inst] = target_w

    return orders, rebalance_count


def _format_jsonl(records: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    lines = [
        json.dumps(
            {col: rec[col] for col in columns if col in rec},
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        for rec in records
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def export_signal_artifacts(
    bundle: StrategyBundle | StrategyBundleSource | str | Path,
    output_dir: str | Path,
    *,
    target_positions: pd.DataFrame | Sequence[Mapping[str, Any]] | None = None,
    current_positions: Mapping[str, Any] | pd.DataFrame | Sequence[Mapping[str, Any]] | None = None,
    bundle_dir: str | Path | None = None,
    formats: Sequence[str] = ("parquet", "jsonl"),
    created_at_utc: str | None = None,
    target_id: str = SIGNAL_EXPORT_TARGET_ID,
) -> DeployArtifactManifest:
    """Validate StrategyBundle + signal positions and export standardized artifacts transactionally.

    When ``target_positions`` is omitted, resolves a directory-bound signal artifact
    (``BundleArtifactRef`` with ``artifact_type`` in ``{"target_positions",
    "target_positions_parquet", "target_positions_jsonl"}`` inside
    ``StrategyBundle.model_artifacts``) from the verified ``bundle_dir``.
    """
    verified_bundle, verified_dir = resolve_verified_bundle_with_dir(
        bundle,
        bundle_dir=bundle_dir,
        require_verified_source=True,
    )
    compat = check_bundle_compatibility(
        verified_bundle,
        bundle_dir=verified_dir,
        target_id=target_id,
    )
    if not compat.compatible:
        first_issue = compat.issues[0]
        raise DeployTargetError(
            first_issue.message,
            code=first_issue.code,
        )

    timestamp_str = (
        validate_utc_iso_timestamp(created_at_utc, "created_at_utc")
        if created_at_utc is not None
        else utc_now_iso()
    )
    norm_formats = normalize_formats(formats)
    allow_short = bool(verified_bundle.execution_profile["allow_short"])

    signal_source_kind = "explicit_target_positions"
    signal_artifact_path: str | None = None
    raw_positions: pd.DataFrame | Sequence[Mapping[str, Any]] | None = target_positions
    if raw_positions is None:
        loaded_bundle_signal = load_bundle_signal_positions(verified_bundle, verified_dir)
        if loaded_bundle_signal is not None:
            raw_positions, signal_artifact_path = loaded_bundle_signal
            signal_source_kind = "bundle_artifact"

    norm_positions = normalize_target_positions(
        raw_positions,
        allow_short=allow_short,
        missing_error_code=TARGET_SIGNAL_SOURCE_MISSING,
    )
    initial_positions = normalize_current_positions(
        current_positions,
        allow_short=allow_short,
    )

    export_df, jsonl_position_records = _build_position_export_frame(
        norm_positions, verified_bundle
    )
    trade_orders, rebalance_count = _build_trade_orders(
        norm_positions, verified_bundle, initial_positions
    )

    root = resolve_path_checked(
        Path(output_dir).expanduser(),
        label="output_dir",
        code=TARGET_ARTIFACT_INTEGRITY_FAILED,
    )
    if root.exists() and not root.is_dir():
        raise DeployTargetError(
            f"output_dir exists and is not a directory: {root}",
            code=TARGET_ARTIFACT_INTEGRITY_FAILED,
        )

    managed_names = (
        TARGET_POSITIONS_PARQUET_FILENAME,
        TARGET_POSITIONS_JSONL_FILENAME,
        TRADE_ORDERS_JSONL_FILENAME,
        DEPLOY_MANIFEST_FILENAME,
    )
    lock_filename = f".{target_id}.lock"
    _ensure_no_bundle_output_overlap(
        verified_bundle,
        verified_dir,
        root,
        (*managed_names, lock_filename),
    )

    root.mkdir(parents=True, exist_ok=True)
    for rel_name in managed_names:
        _validate_managed_destination(root, rel_name)

    lock_path = root / lock_filename
    if lock_path.is_symlink():
        raise DeployTargetError(
            f"Destination path {lock_filename!r} must not be a symlink",
            code=TARGET_ARTIFACT_INTEGRITY_FAILED,
        )
    open_flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        lock_fd = os.open(lock_path, open_flags, 0o600)
    except OSError as exc:
        raise DeployTargetError(
            f"Failed to open export lock file {lock_path}: {exc}",
            code=TARGET_ARTIFACT_INTEGRITY_FAILED,
        ) from exc

    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise DeployTargetError(
                f"Concurrent export already in progress for output_dir: {root}",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            ) from exc

        for rel_name in managed_names:
            _validate_managed_destination(root, rel_name)

        stage_dir = Path(tempfile.mkdtemp(prefix=f".{target_id}_stage_", dir=root))
        backup_dir: Path | None = None
        preserve_backup = False
        try:
            artifact_entries: list[DeployArtifactEntry] = []
            if "parquet" in norm_formats:
                staged_parquet = stage_dir / TARGET_POSITIONS_PARQUET_FILENAME
                export_df.to_parquet(staged_parquet, index=False)
                artifact_entries.append(
                    DeployArtifactEntry(
                        path=TARGET_POSITIONS_PARQUET_FILENAME,
                        sha256=sha256_file(staged_parquet),
                        artifact_type="target_positions_parquet",
                        size_bytes=staged_parquet.stat().st_size,
                    )
                )

            if "jsonl" in norm_formats:
                staged_jsonl = stage_dir / TARGET_POSITIONS_JSONL_FILENAME
                pos_cols = [
                    *POSITION_BASE_OUTPUT_COLUMNS,
                    *(c for c in POSITION_OPTIONAL_COLUMNS if c in export_df.columns),
                ]
                staged_jsonl.write_text(
                    _format_jsonl(jsonl_position_records, pos_cols),
                    encoding="utf-8",
                )
                artifact_entries.append(
                    DeployArtifactEntry(
                        path=TARGET_POSITIONS_JSONL_FILENAME,
                        sha256=sha256_file(staged_jsonl),
                        artifact_type="target_positions_jsonl",
                        size_bytes=staged_jsonl.stat().st_size,
                    )
                )

            staged_orders = stage_dir / TRADE_ORDERS_JSONL_FILENAME
            staged_orders.write_text(
                _format_jsonl(trade_orders, ORDER_OUTPUT_COLUMNS),
                encoding="utf-8",
            )
            artifact_entries.append(
                DeployArtifactEntry(
                    path=TRADE_ORDERS_JSONL_FILENAME,
                    sha256=sha256_file(staged_orders),
                    artifact_type="trade_orders_jsonl",
                    size_bytes=staged_orders.stat().st_size,
                )
            )

            metadata: dict[str, Any] = {
                "formats": list(norm_formats),
                "row_count": int(len(export_df)),
                "order_count": int(len(trade_orders)),
                "rebalance_count": int(rebalance_count),
                "allow_short": allow_short,
                "initial_positions": initial_positions,
                "optional_columns": [
                    col for col in POSITION_OPTIONAL_COLUMNS if col in export_df.columns
                ],
                "signal_source": signal_source_kind,
            }
            if signal_artifact_path is not None:
                metadata["signal_artifact_path"] = signal_artifact_path

            manifest = DeployArtifactManifest(
                target_id=target_id,
                strategy_id=verified_bundle.strategy_id,
                bundle_hash=verified_bundle.bundle_hash,
                slot_id=verified_bundle.slot_id,
                frequency=verified_bundle.frequency,
                created_at_utc=timestamp_str,
                artifacts=tuple(artifact_entries),
                metadata=metadata,
            )
            write_deploy_manifest(manifest, stage_dir)

            publish_names = [entry.path for entry in manifest.artifacts] + [DEPLOY_MANIFEST_FILENAME]
            backup_dir = Path(tempfile.mkdtemp(prefix=f".{target_id}_backup_", dir=root))
            backed_up: list[str] = []
            for rel_name in managed_names:
                existing = root / rel_name
                if existing.is_file() or existing.is_symlink():
                    shutil.copy2(existing, backup_dir / rel_name, follow_symlinks=False)
                    backed_up.append(rel_name)

            modified_names: list[str] = []
            try:
                for rel_name in publish_names:
                    os.replace(stage_dir / rel_name, root / rel_name)
                    modified_names.append(rel_name)
                for stale_name in (TARGET_POSITIONS_PARQUET_FILENAME, TARGET_POSITIONS_JSONL_FILENAME):
                    if stale_name not in publish_names:
                        stale_path = root / stale_name
                        if stale_path.exists() or stale_path.is_symlink():
                            stale_path.unlink(missing_ok=True)
                            modified_names.append(stale_name)
                verified_manifest = verify_deploy_manifest(root)
            except Exception as exc:
                rollback_errors: list[str] = []
                for rel_name in dict.fromkeys(modified_names):
                    try:
                        if rel_name in backed_up:
                            backup_file = backup_dir / rel_name
                            if backup_file.exists() or backup_file.is_symlink():
                                os.replace(backup_file, root / rel_name)
                        else:
                            extra_path = root / rel_name
                            if extra_path.exists() or extra_path.is_symlink():
                                extra_path.unlink(missing_ok=True)
                    except Exception as rb_exc:
                        rollback_errors.append(f"{rel_name}: {rb_exc}")
                        if rel_name == DEPLOY_MANIFEST_FILENAME:
                            # Never leave an unverified failed manifest readable on disk when restoring the prior one fails.
                            try:
                                (root / DEPLOY_MANIFEST_FILENAME).unlink(missing_ok=True)
                            except Exception as unlink_exc:
                                rollback_errors.append(
                                    f"{DEPLOY_MANIFEST_FILENAME} withdraw: {unlink_exc}"
                                )
                if rollback_errors and any(backup_dir.iterdir()):
                    preserve_backup = True
                rb_summary = "; ".join(rollback_errors)
                if preserve_backup:
                    rb_summary = f"{rb_summary}; preserved backup at {backup_dir}"
                if isinstance(exc, DeployTargetError):
                    if rollback_errors:
                        raise DeployTargetError(
                            f"{exc}; rollback errors: {rb_summary}",
                            code=exc.code,
                        ) from exc
                    raise
                detail = f"Failed to publish deployment artifacts atomically: {exc}"
                if rollback_errors:
                    detail += f" (rollback errors: {rb_summary})"
                raise DeployTargetError(
                    detail,
                    code=TARGET_ARTIFACT_INTEGRITY_FAILED,
                ) from exc

            object.__setattr__(verified_manifest, "_base_dir", root)
            return verified_manifest
        finally:
            shutil.rmtree(stage_dir, ignore_errors=True)
            if backup_dir is not None and not preserve_backup:
                shutil.rmtree(backup_dir, ignore_errors=True)
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(lock_fd)


__all__ = [
    "TARGET_POSITIONS_JSONL_FILENAME",
    "TARGET_POSITIONS_PARQUET_FILENAME",
    "TRADE_ORDERS_JSONL_FILENAME",
    "export_signal_artifacts",
]
