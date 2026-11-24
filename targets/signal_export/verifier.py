"""Alignment and integrity verifier for targets/signal_export artifacts."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from factor_synthesis.strategy_bundle import StrategyBundle
from targets.contracts import (
    TARGET_ALIGNMENT_MISMATCH,
    TARGET_ARTIFACT_INTEGRITY_FAILED,
    TARGET_BUNDLE_INVALID,
    TARGET_EXECUTION_CONTRACT_UNSUPPORTED,
    TARGET_SIGNAL_SOURCE_MISSING,
    AlignmentCheckItem,
    AlignmentReport,
    DeployArtifactEntry,
    DeployArtifactManifest,
    DeployTargetError,
    StrategyBundleSource,
    resolve_verified_bundle_with_dir,
    verify_deploy_manifest,
)
from targets.signal_export.schema import (
    MAX_INT64,
    ORDER_OUTPUT_COLUMNS,
    POSITION_BASE_OUTPUT_COLUMNS,
    POSITION_OPTIONAL_COLUMNS,
    SIGNAL_EXPORT_TARGET_ID,
    SIGNAL_POSITION_ARTIFACT_TYPES,
    VALID_ACTIONS,
    VALID_SIDES,
    derive_order_action,
    derive_side,
    format_utc_timestamp,
    load_bundle_signal_positions,
    normalize_current_positions,
    normalize_formats,
    normalize_target_positions,
    parse_utc_timestamp,
    read_jsonl_records,
    read_parquet_records,
    require_clean_str,
    require_finite_float,
    validate_unique_session_per_timestamp,
)

_ALLOWED_ARTIFACT_TYPES = frozenset(
    {"target_positions_parquet", "target_positions_jsonl", "trade_orders_jsonl"}
)


def _validate_position_rows(
    rows: list[dict[str, Any]],
    *,
    bundle: StrategyBundle,
    source_name: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    if not rows:
        raise DeployTargetError(
            f"{source_name} must contain at least one position row",
            code=TARGET_ALIGNMENT_MISMATCH,
        )
    expected_base = set(POSITION_BASE_OUTPUT_COLUMNS)
    first_cols = set(rows[0].keys())
    missing_base = expected_base - first_cols
    if missing_base:
        raise DeployTargetError(
            f"{source_name} missing required fields: {sorted(missing_base)}",
            code=TARGET_ALIGNMENT_MISMATCH,
        )
    extra_cols = first_cols - expected_base - set(POSITION_OPTIONAL_COLUMNS)
    if extra_cols:
        raise DeployTargetError(
            f"{source_name} contains unexpected fields: {sorted(extra_cols)}",
            code=TARGET_ALIGNMENT_MISMATCH,
        )
    optional_cols = [col for col in POSITION_OPTIONAL_COLUMNS if col in first_cols]

    normalized_rows: list[dict[str, Any]] = []
    prev_sort_key: tuple[pd.Timestamp, str] | None = None
    seen_keys: set[tuple[pd.Timestamp, str]] = set()

    for idx, raw in enumerate(rows):
        if set(raw.keys()) != first_cols:
            raise DeployTargetError(
                f"{source_name} row {idx} schema does not match header",
                code=TARGET_ALIGNMENT_MISMATCH,
            )
        ts = parse_utc_timestamp(
            raw["timestamp_utc"],
            field_name=f"{source_name}[{idx}].timestamp_utc",
            code=TARGET_ALIGNMENT_MISMATCH,
        )
        session_id = require_clean_str(
            raw["session_id"],
            f"{source_name}[{idx}].session_id",
            code=TARGET_ALIGNMENT_MISMATCH,
        )
        instrument_id = require_clean_str(
            raw["instrument_id"],
            f"{source_name}[{idx}].instrument_id",
            code=TARGET_ALIGNMENT_MISMATCH,
        )
        target_weight = require_finite_float(
            raw["target_weight"],
            f"{source_name}[{idx}].target_weight",
            code=TARGET_ALIGNMENT_MISMATCH,
        )
        side = require_clean_str(
            raw["side"],
            f"{source_name}[{idx}].side",
            code=TARGET_ALIGNMENT_MISMATCH,
        )
        if side not in VALID_SIDES or side != derive_side(target_weight):
            raise DeployTargetError(
                f"{source_name}[{idx}] side {side!r} inconsistent with target_weight {target_weight}",
                code=TARGET_ALIGNMENT_MISMATCH,
            )
        for meta_key, expected_val in (
            ("slot_id", bundle.slot_id),
            ("strategy_id", bundle.strategy_id),
            ("frequency", bundle.frequency),
            ("bundle_hash", bundle.bundle_hash),
        ):
            actual_val = raw[meta_key]
            if actual_val != expected_val:
                raise DeployTargetError(
                    f"{source_name}[{idx}].{meta_key}={actual_val!r} does not match bundle {expected_val!r}",
                    code=TARGET_BUNDLE_INVALID,
                )

        sort_key = (ts, instrument_id)
        if sort_key in seen_keys:
            raise DeployTargetError(
                f"{source_name} contains duplicate key ({format_utc_timestamp(ts)}, {instrument_id!r})",
                code=TARGET_ALIGNMENT_MISMATCH,
            )
        if prev_sort_key is not None and sort_key < prev_sort_key:
            raise DeployTargetError(
                f"{source_name} is not sorted deterministically by (timestamp_utc, instrument_id)",
                code=TARGET_ALIGNMENT_MISMATCH,
            )
        seen_keys.add(sort_key)
        prev_sort_key = sort_key

        clean_row: dict[str, Any] = {
            "timestamp_utc": ts,
            "session_id": session_id,
            "instrument_id": instrument_id,
            "target_weight": target_weight,
            "side": side,
            "slot_id": bundle.slot_id,
            "strategy_id": bundle.strategy_id,
            "frequency": bundle.frequency,
            "bundle_hash": bundle.bundle_hash,
        }
        for opt_col in optional_cols:
            if opt_col == "lot_size":
                val = raw[opt_col]
                if hasattr(val, "item") and not isinstance(val, (bytes, bytearray)):
                    val = val.item()
                if type(val) is not int or val < 1 or val > MAX_INT64:
                    raise DeployTargetError(
                        f"{source_name}[{idx}].lot_size must be an integer in [1, {MAX_INT64}]",
                        code=TARGET_ALIGNMENT_MISMATCH,
                    )
                clean_row[opt_col] = val
            else:
                num = require_finite_float(
                    raw[opt_col],
                    f"{source_name}[{idx}].{opt_col}",
                    code=TARGET_ALIGNMENT_MISMATCH,
                )
                clean_row[opt_col] = num
        normalized_rows.append(clean_row)

    validate_unique_session_per_timestamp(
        [(row["timestamp_utc"], row["session_id"]) for row in normalized_rows],
        source_name=source_name,
        code=TARGET_ALIGNMENT_MISMATCH,
    )
    return normalized_rows, optional_cols


def _compare_parquet_and_jsonl(
    parquet_rows: list[dict[str, Any]],
    parquet_opt_cols: list[str],
    jsonl_rows: list[dict[str, Any]],
    jsonl_opt_cols: list[str],
) -> tuple[bool, float, str]:
    if parquet_opt_cols != jsonl_opt_cols:
        return (
            False,
            float("inf"),
            f"Optional columns differ: parquet={parquet_opt_cols}, jsonl={jsonl_opt_cols}",
        )
    if len(parquet_rows) != len(jsonl_rows):
        return (
            False,
            float("inf"),
            f"Row count mismatch: parquet={len(parquet_rows)}, jsonl={len(jsonl_rows)}",
        )
    max_diff = 0.0
    numeric_cols = ["target_weight", *[c for c in parquet_opt_cols if c != "lot_size"]]
    exact_cols = [
        "timestamp_utc",
        "session_id",
        "instrument_id",
        "side",
        "slot_id",
        "strategy_id",
        "frequency",
        "bundle_hash",
        *(["lot_size"] if "lot_size" in parquet_opt_cols else []),
    ]
    for idx, (p_row, j_row) in enumerate(zip(parquet_rows, jsonl_rows)):
        for col in exact_cols:
            if p_row[col] != j_row[col]:
                return (
                    False,
                    float("inf"),
                    f"Row {idx} field {col!r} mismatch: parquet={p_row[col]!r}, jsonl={j_row[col]!r}",
                )
        for col in numeric_cols:
            diff = abs(float(p_row[col]) - float(j_row[col]))
            if diff > max_diff:
                max_diff = diff
            if diff != 0.0:
                return (
                    False,
                    diff,
                    f"Row {idx} numeric field {col!r} differs by {diff}: "
                    f"parquet={p_row[col]!r}, jsonl={j_row[col]!r}",
                )
    return True, max_diff, ""


def _validate_trade_orders(
    order_rows: list[dict[str, Any]],
    position_rows: list[dict[str, Any]],
    *,
    bundle: StrategyBundle,
    initial_positions: Mapping[str, float],
    allow_short: bool,
) -> tuple[bool, str, str]:
    if not order_rows:
        return False, "trade_orders.jsonl must not be empty", TARGET_ALIGNMENT_MISMATCH
    expected_cols = set(ORDER_OUTPUT_COLUMNS)
    state: dict[str, float] = {
        str(k): (0.0 if float(v) == 0.0 else float(v))
        for k, v in initial_positions.items()
    }
    pos_by_ts: dict[pd.Timestamp, dict[str, tuple[str, float]]] = {}
    for row in position_rows:
        pos_by_ts.setdefault(row["timestamp_utc"], {})[row["instrument_id"]] = (
            row["session_id"],
            row["target_weight"],
        )

    parsed_orders_by_ts: dict[pd.Timestamp, dict[str, dict[str, Any]]] = {}
    for idx, raw in enumerate(order_rows):
        if not isinstance(raw, Mapping):
            return (
                False,
                f"trade_orders[{idx}] must be a mapping",
                TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
        if set(raw.keys()) != expected_cols:
            return (
                False,
                f"trade_orders[{idx}] schema mismatch: {sorted(raw.keys())}",
                TARGET_ALIGNMENT_MISMATCH,
            )
        try:
            ts = parse_utc_timestamp(
                raw["timestamp_utc"],
                field_name=f"trade_orders[{idx}].timestamp_utc",
                code=TARGET_ALIGNMENT_MISMATCH,
            )
            session_id = require_clean_str(
                raw["session_id"],
                f"trade_orders[{idx}].session_id",
                code=TARGET_ALIGNMENT_MISMATCH,
            )
            instrument_id = require_clean_str(
                raw["instrument_id"],
                f"trade_orders[{idx}].instrument_id",
                code=TARGET_ALIGNMENT_MISMATCH,
            )
            prev_w = require_finite_float(
                raw["prev_weight"],
                f"trade_orders[{idx}].prev_weight",
                code=TARGET_ALIGNMENT_MISMATCH,
            )
            target_w = require_finite_float(
                raw["target_weight"],
                f"trade_orders[{idx}].target_weight",
                code=TARGET_ALIGNMENT_MISMATCH,
            )
            delta_w = require_finite_float(
                raw["delta_weight"],
                f"trade_orders[{idx}].delta_weight",
                code=TARGET_ALIGNMENT_MISMATCH,
            )
            action = require_clean_str(
                raw["action"],
                f"trade_orders[{idx}].action",
                code=TARGET_ALIGNMENT_MISMATCH,
            )
        except DeployTargetError as exc:
            return False, str(exc), exc.code
        for meta_key, expected_val in (
            ("slot_id", bundle.slot_id),
            ("strategy_id", bundle.strategy_id),
            ("bundle_hash", bundle.bundle_hash),
        ):
            if raw[meta_key] != expected_val:
                return (
                    False,
                    f"trade_orders[{idx}].{meta_key} mismatch: {raw[meta_key]!r} != {expected_val!r}",
                    TARGET_BUNDLE_INVALID,
                )

        if not allow_short and (prev_w < 0.0 or target_w < 0.0 or action in ("SHORT", "COVER")):
            return (
                False,
                f"trade_orders[{idx}] violates allow_short=False: prev={prev_w}, target={target_w}, action={action}",
                TARGET_EXECUTION_CONTRACT_UNSUPPORTED,
            )

        expected_delta = target_w - prev_w
        if not math.isfinite(expected_delta):
            return (
                False,
                f"trade_orders[{idx}] target_weight - prev_weight exceeds finite float range",
                TARGET_ALIGNMENT_MISMATCH,
            )
        if expected_delta == 0.0:
            expected_delta = 0.0
        if delta_w != expected_delta:
            return (
                False,
                f"trade_orders[{idx}] delta_weight {delta_w!r} != target_weight - prev_weight ({expected_delta!r})",
                TARGET_ALIGNMENT_MISMATCH,
            )
        expected_action = derive_order_action(prev_w, target_w)
        if action not in VALID_ACTIONS or action != expected_action:
            return (
                False,
                f"trade_orders[{idx}] action {action!r} != expected {expected_action!r}",
                TARGET_ALIGNMENT_MISMATCH,
            )

        ts_bucket = parsed_orders_by_ts.setdefault(ts, {})
        if instrument_id in ts_bucket:
            return (
                False,
                f"trade_orders contains duplicate key ({format_utc_timestamp(ts)}, {instrument_id!r})",
                TARGET_ALIGNMENT_MISMATCH,
            )
        ts_bucket[instrument_id] = {
            "session_id": session_id,
            "prev_weight": prev_w,
            "target_weight": target_w,
        }

    if list(sorted(parsed_orders_by_ts)) != list(sorted(pos_by_ts)):
        return (
            False,
            "trade_orders rebalance timestamps do not match target_positions timestamps",
            TARGET_ALIGNMENT_MISMATCH,
        )

    for ts in sorted(pos_by_ts):
        targets_at_ts = pos_by_ts[ts]
        orders_at_ts = parsed_orders_by_ts[ts]
        expected_session = next(iter(targets_at_ts.values()))[0]
        active_held = {inst for inst, w in state.items() if w != 0.0}
        expected_instruments = set(targets_at_ts) | active_held
        if set(orders_at_ts) != expected_instruments:
            return (
                False,
                f"trade_orders instruments at {format_utc_timestamp(ts)} do not match expected rebalance set",
                TARGET_ALIGNMENT_MISMATCH,
            )
        for inst in sorted(expected_instruments):
            order = orders_at_ts[inst]
            if order["session_id"] != expected_session:
                return (
                    False,
                    f"trade_orders session_id mismatch for {inst!r} at {format_utc_timestamp(ts)}",
                    TARGET_ALIGNMENT_MISMATCH,
                )
            expected_prev = state.get(inst, 0.0)
            if order["prev_weight"] != expected_prev:
                return (
                    False,
                    f"trade_orders prev_weight continuity mismatch for {inst!r} at {format_utc_timestamp(ts)}: "
                    f"expected {expected_prev}, got {order['prev_weight']}",
                    TARGET_ALIGNMENT_MISMATCH,
                )
            expected_target = targets_at_ts[inst][1] if inst in targets_at_ts else 0.0
            if order["target_weight"] != expected_target:
                return (
                    False,
                    f"trade_orders target_weight mismatch for {inst!r} at {format_utc_timestamp(ts)}: "
                    f"expected {expected_target}, got {order['target_weight']}",
                    TARGET_ALIGNMENT_MISMATCH,
                )
            state[inst] = expected_target

    return True, "", ""


def _fail_report(
    *,
    target_id: str,
    bundle_hash: str,
    checks: list[AlignmentCheckItem],
    name: str,
    reason: str,
    code: str,
    max_abs_diff: float | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> AlignmentReport:
    checks.append(AlignmentCheckItem(name=name, passed=False, reason=reason, code=code))
    return AlignmentReport(
        target_id=target_id,
        passed=False,
        bundle_hash=bundle_hash,
        checks=tuple(checks),
        metrics=dict(metrics or {}),
        max_abs_diff=max_abs_diff,
        error_code=code,
    )


def verify_signal_export_alignment(
    bundle: StrategyBundle | StrategyBundleSource | str | Path,
    artifact: DeployArtifactManifest | str | Path,
    *,
    reference_positions: pd.DataFrame | Sequence[Mapping[str, Any]] | None = None,
    output_dir: str | Path | None = None,
    bundle_dir: str | Path | None = None,
    target_id: str = SIGNAL_EXPORT_TARGET_ID,
) -> AlignmentReport:
    """Verify exported manifest integrity, bundle binding, Parquet/JSONL parity, order closure, and reference alignment."""
    checks: list[AlignmentCheckItem] = []
    bundle_hash_for_report = ""
    bundle_hint = bundle.bundle if isinstance(bundle, StrategyBundleSource) else bundle
    if isinstance(bundle_hint, StrategyBundle) and isinstance(getattr(bundle_hint, "bundle_hash", None), str):
        if len(bundle_hint.bundle_hash) == 64:
            bundle_hash_for_report = bundle_hint.bundle_hash

    try:
        verified_bundle, verified_bundle_dir = resolve_verified_bundle_with_dir(
            bundle,
            bundle_dir=bundle_dir,
            require_verified_source=True,
        )
        bundle_hash_for_report = verified_bundle.bundle_hash
    except DeployTargetError as exc:
        return _fail_report(
            target_id=target_id,
            bundle_hash=bundle_hash_for_report,
            checks=checks,
            name="bundle_valid",
            reason=str(exc),
            code=exc.code,
        )

    try:
        manifest = verify_deploy_manifest(artifact, output_dir=output_dir)
    except DeployTargetError as exc:
        return _fail_report(
            target_id=target_id,
            bundle_hash=bundle_hash_for_report,
            checks=checks,
            name="manifest_integrity",
            reason=str(exc),
            code=exc.code,
        )
    checks.append(
        AlignmentCheckItem(
            name="manifest_integrity",
            passed=True,
            reason="manifest_hash, path containment, and artifact SHA-256 checksums verified",
        )
    )

    root: Path = getattr(manifest, "_base_dir")
    binding_mismatches: list[str] = []
    if manifest.target_id != target_id:
        binding_mismatches.append(f"target_id {manifest.target_id!r} != {target_id!r}")
    if manifest.bundle_hash != verified_bundle.bundle_hash:
        binding_mismatches.append(
            f"bundle_hash {manifest.bundle_hash} != {verified_bundle.bundle_hash}"
        )
    if manifest.slot_id != verified_bundle.slot_id:
        binding_mismatches.append(
            f"slot_id {manifest.slot_id!r} != {verified_bundle.slot_id!r}"
        )
    if manifest.frequency != verified_bundle.frequency:
        binding_mismatches.append(
            f"frequency {manifest.frequency!r} != {verified_bundle.frequency!r}"
        )
    if manifest.strategy_id != verified_bundle.strategy_id:
        binding_mismatches.append(
            f"strategy_id {manifest.strategy_id!r} != {verified_bundle.strategy_id!r}"
        )
    if binding_mismatches:
        return _fail_report(
            target_id=target_id,
            bundle_hash=verified_bundle.bundle_hash,
            checks=checks,
            name="bundle_binding",
            reason="; ".join(binding_mismatches),
            code=TARGET_BUNDLE_INVALID,
        )
    checks.append(
        AlignmentCheckItem(
            name="bundle_binding",
            passed=True,
            reason="bundle_hash, slot_id, frequency, and strategy_id match StrategyBundle",
        )
    )

    by_type: dict[str, DeployArtifactEntry] = {}
    for entry in manifest.artifacts:
        if entry.artifact_type not in _ALLOWED_ARTIFACT_TYPES:
            return _fail_report(
                target_id=target_id,
                bundle_hash=verified_bundle.bundle_hash,
                checks=checks,
                name="position_artifacts_present",
                reason=f"Unsupported artifact_type in manifest: {entry.artifact_type!r}",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
        if entry.artifact_type in by_type:
            return _fail_report(
                target_id=target_id,
                bundle_hash=verified_bundle.bundle_hash,
                checks=checks,
                name="position_artifacts_present",
                reason=f"Duplicate artifact_type in manifest: {entry.artifact_type!r}",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
        by_type[entry.artifact_type] = entry

    parquet_entry = by_type.get("target_positions_parquet")
    jsonl_entry = by_type.get("target_positions_jsonl")
    orders_entry = by_type.get("trade_orders_jsonl")

    if "formats" not in manifest.metadata:
        return _fail_report(
            target_id=target_id,
            bundle_hash=verified_bundle.bundle_hash,
            checks=checks,
            name="position_artifacts_present",
            reason="Manifest metadata missing required 'formats' declaration",
            code=TARGET_ARTIFACT_INTEGRITY_FAILED,
        )
    try:
        declared_formats = normalize_formats(manifest.metadata["formats"])
    except DeployTargetError as exc:
        return _fail_report(
            target_id=target_id,
            bundle_hash=verified_bundle.bundle_hash,
            checks=checks,
            name="position_artifacts_present",
            reason=f"Invalid manifest metadata.formats: {exc}",
            code=TARGET_ARTIFACT_INTEGRITY_FAILED,
        )

    declared_pos_types = {f"target_positions_{fmt}" for fmt in declared_formats}
    actual_pos_types = {
        atype
        for atype in ("target_positions_parquet", "target_positions_jsonl")
        if atype in by_type
    }
    if declared_pos_types != actual_pos_types:
        return _fail_report(
            target_id=target_id,
            bundle_hash=verified_bundle.bundle_hash,
            checks=checks,
            name="position_artifacts_present",
            reason=(
                f"Manifest metadata.formats {list(declared_formats)!r} requires position artifacts "
                f"{sorted(declared_pos_types)}, found {sorted(actual_pos_types)}"
            ),
            code=TARGET_ARTIFACT_INTEGRITY_FAILED,
        )

    if parquet_entry is None and jsonl_entry is None:
        return _fail_report(
            target_id=target_id,
            bundle_hash=verified_bundle.bundle_hash,
            checks=checks,
            name="position_artifacts_present",
            reason="Manifest contains neither target_positions_parquet nor target_positions_jsonl",
            code=TARGET_ARTIFACT_INTEGRITY_FAILED,
        )

    parquet_rows: list[dict[str, Any]] | None = None
    parquet_opt_cols: list[str] = []
    jsonl_rows: list[dict[str, Any]] | None = None
    jsonl_opt_cols: list[str] = []

    try:
        if parquet_entry is not None:
            raw_parquet_rows = read_parquet_records(
                root / parquet_entry.path, label=parquet_entry.path
            )
            parquet_rows, parquet_opt_cols = _validate_position_rows(
                raw_parquet_rows,
                bundle=verified_bundle,
                source_name=parquet_entry.path,
            )
        if jsonl_entry is not None:
            raw_jsonl_rows = read_jsonl_records(
                root / jsonl_entry.path, label=jsonl_entry.path
            )
            jsonl_rows, jsonl_opt_cols = _validate_position_rows(
                raw_jsonl_rows,
                bundle=verified_bundle,
                source_name=jsonl_entry.path,
            )
    except DeployTargetError as exc:
        return _fail_report(
            target_id=target_id,
            bundle_hash=verified_bundle.bundle_hash,
            checks=checks,
            name="position_schema",
            reason=str(exc),
            code=exc.code,
        )

    checks.append(
        AlignmentCheckItem(
            name="position_schema",
            passed=True,
            reason="Exported target position rows satisfy UTC, sort, key uniqueness, session uniqueness, and metadata contracts",
        )
    )

    canonical_rows = parquet_rows if parquet_rows is not None else jsonl_rows
    canonical_opt_cols = parquet_opt_cols if parquet_rows is not None else jsonl_opt_cols
    assert canonical_rows is not None

    max_abs_diff = 0.0
    if parquet_rows is not None and jsonl_rows is not None:
        eq_ok, format_diff, eq_reason = _compare_parquet_and_jsonl(
            parquet_rows, parquet_opt_cols, jsonl_rows, jsonl_opt_cols
        )
        max_abs_diff = format_diff if math.isfinite(format_diff) else None  # type: ignore[assignment]
        if not eq_ok:
            return _fail_report(
                target_id=target_id,
                bundle_hash=verified_bundle.bundle_hash,
                checks=checks,
                name="parquet_jsonl_parity",
                reason=eq_reason,
                code=TARGET_ALIGNMENT_MISMATCH,
                max_abs_diff=max_abs_diff,
            )
        checks.append(
            AlignmentCheckItem(
                name="parquet_jsonl_parity",
                passed=True,
                reason="Parquet and JSONL target positions match bit-for-bit (max_abs_diff == 0.0)",
            )
        )

    allow_short = bool(verified_bundle.execution_profile["allow_short"])
    if "allow_short" in manifest.metadata and manifest.metadata["allow_short"] is not allow_short:
        return _fail_report(
            target_id=target_id,
            bundle_hash=verified_bundle.bundle_hash,
            checks=checks,
            name="allow_short_constraint",
            reason=(
                f"Manifest metadata.allow_short ({manifest.metadata['allow_short']!r}) "
                f"disagrees with bundle allow_short ({allow_short!r})"
            ),
            code=TARGET_EXECUTION_CONTRACT_UNSUPPORTED,
            max_abs_diff=max_abs_diff,
        )
    if not allow_short:
        neg_rows = [row for row in canonical_rows if row["target_weight"] < 0.0 or row["side"] == "SHORT"]
        if neg_rows:
            sample = neg_rows[0]
            return _fail_report(
                target_id=target_id,
                bundle_hash=verified_bundle.bundle_hash,
                checks=checks,
                name="allow_short_constraint",
                reason=(
                    f"Negative weight {sample['target_weight']} for {sample['instrument_id']!r} "
                    "violates execution_profile.allow_short=False"
                ),
                code=TARGET_EXECUTION_CONTRACT_UNSUPPORTED,
                max_abs_diff=max_abs_diff,
            )
    checks.append(
        AlignmentCheckItem(
            name="allow_short_constraint",
            passed=True,
            reason=f"All positions respect allow_short={allow_short}",
        )
    )

    if orders_entry is None:
        return _fail_report(
            target_id=target_id,
            bundle_hash=verified_bundle.bundle_hash,
            checks=checks,
            name="trade_orders_closure",
            reason="Manifest missing trade_orders_jsonl artifact",
            code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            max_abs_diff=max_abs_diff,
        )

    try:
        raw_orders = read_jsonl_records(root / orders_entry.path, label=orders_entry.path)
    except DeployTargetError as exc:
        return _fail_report(
            target_id=target_id,
            bundle_hash=verified_bundle.bundle_hash,
            checks=checks,
            name="trade_orders_closure",
            reason=str(exc),
            code=exc.code,
            max_abs_diff=max_abs_diff,
        )

    if "initial_positions" in manifest.metadata:
        initial_positions_raw = manifest.metadata["initial_positions"]
        if not isinstance(initial_positions_raw, Mapping):
            return _fail_report(
                target_id=target_id,
                bundle_hash=verified_bundle.bundle_hash,
                checks=checks,
                name="trade_orders_closure",
                reason="Manifest metadata.initial_positions must be a mapping",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
                max_abs_diff=max_abs_diff,
            )
    else:
        initial_positions_raw = {}

    try:
        initial_positions = normalize_current_positions(
            initial_positions_raw,
            allow_short=allow_short,
        )
    except DeployTargetError as exc:
        err_code = (
            TARGET_ARTIFACT_INTEGRITY_FAILED
            if exc.code == TARGET_BUNDLE_INVALID
            else exc.code
        )
        return _fail_report(
            target_id=target_id,
            bundle_hash=verified_bundle.bundle_hash,
            checks=checks,
            name="trade_orders_closure",
            reason=f"Invalid manifest metadata.initial_positions: {exc}",
            code=err_code,
            max_abs_diff=max_abs_diff,
        )

    orders_ok, orders_reason, orders_code = _validate_trade_orders(
        raw_orders,
        canonical_rows,
        bundle=verified_bundle,
        initial_positions=initial_positions,
        allow_short=allow_short,
    )
    if not orders_ok:
        return _fail_report(
            target_id=target_id,
            bundle_hash=verified_bundle.bundle_hash,
            checks=checks,
            name="trade_orders_closure",
            reason=orders_reason,
            code=orders_code,
            max_abs_diff=max_abs_diff,
        )
    rebalance_timestamps = {row["timestamp_utc"] for row in canonical_rows}
    meta_checks = (
        ("row_count", len(canonical_rows)),
        ("order_count", len(raw_orders)),
        ("rebalance_count", len(rebalance_timestamps)),
    )
    for meta_key, actual_count in meta_checks:
        if meta_key in manifest.metadata:
            meta_val = manifest.metadata[meta_key]
            if type(meta_val) is not int or meta_val != actual_count:
                return _fail_report(
                    target_id=target_id,
                    bundle_hash=verified_bundle.bundle_hash,
                    checks=checks,
                    name="trade_orders_closure",
                    reason=f"Manifest metadata.{meta_key} ({meta_val!r}) != actual ({actual_count})",
                    code=TARGET_ALIGNMENT_MISMATCH,
                    max_abs_diff=max_abs_diff,
                )
    if "optional_columns" in manifest.metadata:
        opt_cols_meta = manifest.metadata["optional_columns"]
        if (
            isinstance(opt_cols_meta, (str, bytes))
            or not isinstance(opt_cols_meta, Sequence)
            or list(opt_cols_meta) != canonical_opt_cols
        ):
            return _fail_report(
                target_id=target_id,
                bundle_hash=verified_bundle.bundle_hash,
                checks=checks,
                name="trade_orders_closure",
                reason=(
                    f"Manifest metadata.optional_columns ({opt_cols_meta!r}) "
                    f"!= actual ({canonical_opt_cols!r})"
                ),
                code=TARGET_ALIGNMENT_MISMATCH,
                max_abs_diff=max_abs_diff,
            )

    checks.append(
        AlignmentCheckItem(
            name="trade_orders_closure",
            passed=True,
            reason="All trade orders satisfy delta_weight == target_weight - prev_weight and rebalance state continuity",
        )
    )

    has_signal_source = "signal_source" in manifest.metadata
    declared_signal_source = manifest.metadata.get("signal_source")
    declared_signal_path = manifest.metadata.get("signal_artifact_path")

    if has_signal_source and declared_signal_source not in (
        "explicit_target_positions",
        "bundle_artifact",
    ):
        return _fail_report(
            target_id=target_id,
            bundle_hash=verified_bundle.bundle_hash,
            checks=checks,
            name="reference_positions_alignment",
            reason=f"Unsupported manifest metadata.signal_source: {declared_signal_source!r}",
            code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            max_abs_diff=max_abs_diff,
        )

    if declared_signal_source == "explicit_target_positions":
        if declared_signal_path is not None:
            return _fail_report(
                target_id=target_id,
                bundle_hash=verified_bundle.bundle_hash,
                checks=checks,
                name="reference_positions_alignment",
                reason=(
                    "Manifest declares signal_source='explicit_target_positions' "
                    "but also sets signal_artifact_path"
                ),
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
                max_abs_diff=max_abs_diff,
            )
    elif not has_signal_source and declared_signal_path is not None:
        return _fail_report(
            target_id=target_id,
            bundle_hash=verified_bundle.bundle_hash,
            checks=checks,
            name="reference_positions_alignment",
            reason="Manifest sets signal_artifact_path without declaring signal_source='bundle_artifact'",
            code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            max_abs_diff=max_abs_diff,
        )

    effective_reference = reference_positions
    if declared_signal_source == "bundle_artifact":
        signal_refs = [
            ref
            for ref in verified_bundle.model_artifacts
            if ref.artifact_type in SIGNAL_POSITION_ARTIFACT_TYPES
        ]
        if not signal_refs:
            return _fail_report(
                target_id=target_id,
                bundle_hash=verified_bundle.bundle_hash,
                checks=checks,
                name="reference_positions_alignment",
                reason="Manifest declares signal_source='bundle_artifact' but bundle binds no signal artifact",
                code=TARGET_SIGNAL_SOURCE_MISSING,
                max_abs_diff=max_abs_diff,
            )
        if len(signal_refs) > 1:
            return _fail_report(
                target_id=target_id,
                bundle_hash=verified_bundle.bundle_hash,
                checks=checks,
                name="reference_positions_alignment",
                reason=(
                    "Manifest declares signal_source='bundle_artifact' but bundle binds "
                    f"multiple ambiguous target_positions artifacts: {[ref.path for ref in signal_refs]}"
                ),
                code=TARGET_BUNDLE_INVALID,
                max_abs_diff=max_abs_diff,
            )
        bound_signal_path = signal_refs[0].path
        if declared_signal_path is not None and declared_signal_path != bound_signal_path:
            return _fail_report(
                target_id=target_id,
                bundle_hash=verified_bundle.bundle_hash,
                checks=checks,
                name="reference_positions_alignment",
                reason=(
                    f"Manifest signal_artifact_path ({declared_signal_path!r}) != "
                    f"bundle signal artifact ({bound_signal_path!r})"
                ),
                code=TARGET_BUNDLE_INVALID,
                max_abs_diff=max_abs_diff,
            )
        if effective_reference is None:
            try:
                loaded_bundle_signal = load_bundle_signal_positions(
                    verified_bundle, verified_bundle_dir
                )
            except DeployTargetError as exc:
                return _fail_report(
                    target_id=target_id,
                    bundle_hash=verified_bundle.bundle_hash,
                    checks=checks,
                    name="reference_positions_alignment",
                    reason=str(exc),
                    code=exc.code,
                    max_abs_diff=max_abs_diff,
                )
            assert loaded_bundle_signal is not None
            effective_reference = loaded_bundle_signal[0]
    elif declared_signal_source is None and effective_reference is None:
        try:
            loaded_bundle_signal = load_bundle_signal_positions(
                verified_bundle, verified_bundle_dir
            )
        except DeployTargetError as exc:
            return _fail_report(
                target_id=target_id,
                bundle_hash=verified_bundle.bundle_hash,
                checks=checks,
                name="reference_positions_alignment",
                reason=str(exc),
                code=exc.code,
                max_abs_diff=max_abs_diff,
            )
        if loaded_bundle_signal is not None:
            effective_reference = loaded_bundle_signal[0]

    ref_max_diff: float | None = None
    if effective_reference is not None:
        try:
            ref_df = normalize_target_positions(
                effective_reference,
                allow_short=allow_short,
                missing_error_code=TARGET_ALIGNMENT_MISMATCH,
            )
        except DeployTargetError as exc:
            err_code = TARGET_ALIGNMENT_MISMATCH if exc.code == TARGET_BUNDLE_INVALID else exc.code
            return _fail_report(
                target_id=target_id,
                bundle_hash=verified_bundle.bundle_hash,
                checks=checks,
                name="reference_positions_alignment",
                reason=str(exc),
                code=err_code,
                max_abs_diff=max_abs_diff,
            )

        actual_map = {
            (row["timestamp_utc"], row["instrument_id"]): (row["session_id"], row["target_weight"])
            for row in canonical_rows
        }
        ref_map = {
            (
                pd.Timestamp(row["timestamp_utc"]).tz_convert("UTC"),
                str(row["instrument_id"]),
            ): (str(row["session_id"]), float(row["target_weight"]))
            for row in ref_df.to_dict(orient="records")
        }
        if set(actual_map) != set(ref_map):
            missing_in_actual = len(set(ref_map) - set(actual_map))
            extra_in_actual = len(set(actual_map) - set(ref_map))
            return _fail_report(
                target_id=target_id,
                bundle_hash=verified_bundle.bundle_hash,
                checks=checks,
                name="reference_positions_alignment",
                reason=(
                    f"Reference key mismatch: missing_in_exported={missing_in_actual}, "
                    f"extra_in_exported={extra_in_actual}"
                ),
                code=TARGET_ALIGNMENT_MISMATCH,
            )

        worst_diff = 0.0
        for key in sorted(actual_map):
            act_sess, act_w = actual_map[key]
            ref_sess, ref_w = ref_map[key]
            if act_sess != ref_sess:
                return _fail_report(
                    target_id=target_id,
                    bundle_hash=verified_bundle.bundle_hash,
                    checks=checks,
                    name="reference_positions_alignment",
                    reason=(
                        f"session_id mismatch at ({format_utc_timestamp(key[0])}, {key[1]!r}): "
                        f"exported={act_sess!r}, reference={ref_sess!r}"
                    ),
                    code=TARGET_ALIGNMENT_MISMATCH,
                )
            diff = abs(act_w - ref_w)
            if not math.isfinite(diff) or (math.isfinite(worst_diff) and diff > worst_diff):
                worst_diff = diff
        if math.isfinite(worst_diff):
            ref_max_diff = worst_diff
            max_abs_diff = max(max_abs_diff, worst_diff) if max_abs_diff is not None else worst_diff
        else:
            ref_max_diff = None
            max_abs_diff = None
        if not math.isfinite(worst_diff) or worst_diff > 1e-12:
            fail_metrics: dict[str, Any] = {
                "row_count": len(canonical_rows),
                "order_count": len(raw_orders),
            }
            if ref_max_diff is not None:
                fail_metrics["reference_max_abs_diff"] = ref_max_diff
            else:
                fail_metrics["reference_max_abs_diff_overflow"] = True
            diff_label = f"{worst_diff:.3e}" if math.isfinite(worst_diff) else "inf (float overflow)"
            return _fail_report(
                target_id=target_id,
                bundle_hash=verified_bundle.bundle_hash,
                checks=checks,
                name="reference_positions_alignment",
                reason=f"Reference weight max_abs_diff={diff_label} exceeds atol=1e-12",
                code=TARGET_ALIGNMENT_MISMATCH,
                max_abs_diff=max_abs_diff,
                metrics=fail_metrics,
            )
        checks.append(
            AlignmentCheckItem(
                name="reference_positions_alignment",
                passed=True,
                reason=f"Reference weights aligned within rtol=0, atol=1e-12 (max_abs_diff={worst_diff:.3e})",
            )
        )

    metrics: dict[str, Any] = {
        "row_count": len(canonical_rows),
        "order_count": len(raw_orders),
        "parquet_checked": parquet_rows is not None,
        "jsonl_checked": jsonl_rows is not None,
    }
    if ref_max_diff is not None:
        metrics["reference_max_abs_diff"] = ref_max_diff

    return AlignmentReport(
        target_id=target_id,
        passed=True,
        bundle_hash=verified_bundle.bundle_hash,
        checks=tuple(checks),
        metrics=metrics,
        max_abs_diff=max_abs_diff,
    )


__all__ = ["verify_signal_export_alignment"]
