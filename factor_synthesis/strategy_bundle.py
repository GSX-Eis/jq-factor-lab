"""Immutable versioned StrategyBundle contract for Stage 6 deployment targets."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from utils.market_db.identity import market_db_identity
from utils.slots.contracts import ExecutionProfile
from utils.time_axis import Frequency

STRATEGY_BUNDLE_SCHEMA_VERSION = "strategy_bundle.v1"
MANIFEST_FILENAME = "manifest.json"
_EXECUTION_RESULT_SCHEMA_VERSION = "execution_result.v1"
_HEX_CHARS = frozenset("0123456789abcdef")

_REQUIRED_PERFORMANCE_KEYS = (
    "annual_return",
    "sharpe_ratio",
    "max_drawdown",
    "annual_turnover",
)
_OPTIONAL_FLOAT_PERFORMANCE_KEYS = (
    "total_return",
    "calmar_ratio",
    "sortino_ratio",
)
_REPLAYABLE_MODEL_ARTIFACT_TYPES = frozenset(
    {"model_bundle", "model_weights", "lgb_model", "model_index"}
)
_FORBIDDEN_MODEL_ARTIFACT_TYPES = frozenset(
    {"synthesis_spec", "diagnostics", "size_conditional_diagnostics"}
)


class StrategyBundleError(ValueError):
    """Raised when a StrategyBundle fails schema, path, or integrity verification."""


def _reject_json_constant(token: str) -> Any:
    raise StrategyBundleError(
        f"Non-standard JSON constant {token!r} is not permitted in StrategyBundle"
    )


def _require_nonempty_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise StrategyBundleError(
            f"{field} must be a nonempty string without surrounding whitespace"
        )
    return value


def _validate_sha256_hex(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in _HEX_CHARS for c in value):
        raise StrategyBundleError(
            f"{field} must be a 64-character lowercase SHA-256 hex digest"
        )
    return value


def _validate_rel_posix_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise StrategyBundleError(f"{field} must be a nonempty relative POSIX path")
    if "\x00" in value or "\\" in value:
        raise StrategyBundleError(f"{field} contains forbidden path characters: {value!r}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise StrategyBundleError(f"{field} must not escape bundle root: {value!r}")
    normalized = pure.as_posix()
    if normalized in ("", ".") or normalized != value:
        raise StrategyBundleError(f"{field} must be canonical relative POSIX path: {value!r}")
    return normalized


def _normalize_json_value(
    value: Any,
    *,
    field: str,
    freeze: bool,
    ancestors: frozenset[int] = frozenset(),
) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise StrategyBundleError(f"{field} contains non-finite float value")
        return value
    if hasattr(value, "item") and not isinstance(value, (bytes, bytearray)):
        return _normalize_json_value(
            value.item(), field=field, freeze=freeze, ancestors=ancestors
        )
    if id(value) in ancestors:
        raise StrategyBundleError(f"{field} contains cyclic structure")
    ancestors = ancestors | {id(value)}
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise StrategyBundleError(f"{field} mapping keys must be strings")
            out[key] = _normalize_json_value(
                item, field=f"{field}.{key}", freeze=freeze, ancestors=ancestors
            )
        return MappingProxyType(out) if freeze else out
    if isinstance(value, (list, tuple)):
        items = [
            _normalize_json_value(
                item, field=f"{field}[{idx}]", freeze=freeze, ancestors=ancestors
            )
            for idx, item in enumerate(value)
        ]
        return tuple(items) if freeze else items
    raise StrategyBundleError(
        f"{field} contains unsupported JSON type {type(value).__name__}"
    )


def _unfreeze_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _unfreeze_json_value(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_unfreeze_json_value(item) for item in value]
    return value


def sha256_file(path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_bundle_hash(payload_without_hash: Mapping[str, Any]) -> str:
    """Compute canonical SHA-256 digest of a StrategyBundle payload without bundle_hash."""
    clean = {k: v for k, v in payload_without_hash.items() if k != "bundle_hash"}
    canonical = json.dumps(
        _unfreeze_json_value(clean),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _resolve_contained_file(root_dir: Path, rel_path: str, field: str) -> Path:
    clean_rel = _validate_rel_posix_path(rel_path, field)
    root_resolved = root_dir.resolve()
    candidate = root_dir / clean_rel
    if candidate.is_symlink():
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root_resolved):
            raise StrategyBundleError(
                f"{field} symlink escapes bundle directory: {clean_rel!r}"
            )
    else:
        resolved = candidate.resolve()
    if not resolved.is_relative_to(root_resolved) or resolved == root_resolved:
        raise StrategyBundleError(
            f"{field} resolves outside bundle directory: {clean_rel!r}"
        )
    if not resolved.is_file():
        raise StrategyBundleError(f"{field} missing referenced file: {clean_rel!r}")
    return resolved


@dataclass(frozen=True)
class StrategyFactorSpec:
    """Immutable input factor specification and compute dependency metadata."""

    factor_id: str
    factor_identity_hash: str
    frequency: str
    preprocessing: str | Mapping[str, Any] = "none"
    aux_features_required: tuple[str, ...] = ()
    source_module: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "factor_id", _require_nonempty_str(self.factor_id, "factor_id")
        )
        object.__setattr__(
            self,
            "factor_identity_hash",
            _validate_sha256_hex(self.factor_identity_hash, "factor_identity_hash"),
        )
        parsed = Frequency.parse(
            _require_nonempty_str(self.frequency, "factor.frequency")
        )
        if not parsed.is_valid or parsed.canonical is None:
            raise StrategyBundleError(f"Invalid factor frequency: {self.frequency!r}")
        object.__setattr__(self, "frequency", parsed.canonical)
        if isinstance(self.preprocessing, str):
            object.__setattr__(
                self,
                "preprocessing",
                _require_nonempty_str(self.preprocessing, "factor.preprocessing"),
            )
        elif isinstance(self.preprocessing, Mapping):
            object.__setattr__(
                self,
                "preprocessing",
                _normalize_json_value(
                    self.preprocessing, field="factor.preprocessing", freeze=True
                ),
            )
        else:
            raise StrategyBundleError(
                "factor.preprocessing must be a string or mapping"
            )
        if isinstance(self.aux_features_required, str) or not isinstance(
            self.aux_features_required, Sequence
        ):
            raise StrategyBundleError(
                "factor.aux_features_required must be a sequence of strings"
            )
        aux = tuple(
            _require_nonempty_str(item, "factor.aux_features_required[]")
            for item in self.aux_features_required
        )
        object.__setattr__(self, "aux_features_required", aux)
        if self.source_module is not None:
            object.__setattr__(
                self,
                "source_module",
                _require_nonempty_str(self.source_module, "factor.source_module"),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "factor_identity_hash": self.factor_identity_hash,
            "frequency": self.frequency,
            "preprocessing": _unfreeze_json_value(self.preprocessing),
            "aux_features_required": list(self.aux_features_required),
            "source_module": self.source_module,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> StrategyFactorSpec:
        if not isinstance(raw, Mapping):
            raise StrategyBundleError("factor entry must be a mapping")
        expected = {
            "factor_id",
            "factor_identity_hash",
            "frequency",
            "preprocessing",
            "aux_features_required",
            "source_module",
        }
        missing = {"factor_id", "factor_identity_hash", "frequency"} - set(raw)
        if missing:
            raise StrategyBundleError(
                f"factor entry missing required fields: {sorted(missing)}"
            )
        extra = set(raw) - expected
        if extra:
            raise StrategyBundleError(
                f"factor entry contains unknown fields: {sorted(extra)}"
            )
        return cls(
            factor_id=raw["factor_id"],
            factor_identity_hash=raw["factor_identity_hash"],
            frequency=raw["frequency"],
            preprocessing=raw.get("preprocessing", "none"),
            aux_features_required=tuple(raw.get("aux_features_required") or ()),
            source_module=raw.get("source_module"),
        )


@dataclass(frozen=True)
class BundleArtifactRef:
    """Content-addressed relative file entry inside a StrategyBundle directory."""

    path: str
    sha256: str
    artifact_type: str
    size_bytes: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "path", _validate_rel_posix_path(self.path, "artifact.path")
        )
        object.__setattr__(
            self, "sha256", _validate_sha256_hex(self.sha256, "artifact.sha256")
        )
        object.__setattr__(
            self,
            "artifact_type",
            _require_nonempty_str(self.artifact_type, "artifact.artifact_type"),
        )
        if self.size_bytes is not None:
            if type(self.size_bytes) is not int or self.size_bytes < 0:
                raise StrategyBundleError(
                    "artifact.size_bytes must be a non-negative integer"
                )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path": self.path,
            "sha256": self.sha256,
            "artifact_type": self.artifact_type,
        }
        if self.size_bytes is not None:
            payload["size_bytes"] = self.size_bytes
        return payload

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> BundleArtifactRef:
        if not isinstance(raw, Mapping):
            raise StrategyBundleError("artifact entry must be a mapping")
        missing = {"path", "sha256", "artifact_type"} - set(raw)
        if missing:
            raise StrategyBundleError(
                f"artifact entry missing required fields: {sorted(missing)}"
            )
        extra = set(raw) - {"path", "sha256", "artifact_type", "size_bytes"}
        if extra:
            raise StrategyBundleError(
                f"artifact entry contains unknown fields: {sorted(extra)}"
            )
        return cls(
            path=raw["path"],
            sha256=raw["sha256"],
            artifact_type=raw["artifact_type"],
            size_bytes=raw.get("size_bytes"),
        )


@dataclass(frozen=True)
class StrategyBundle:
    """Versioned immutable contract produced by Stage 5 and consumed by Stage 6."""

    slot_id: str
    slot_config_hash: str
    calendar_hash: str
    slot_data_fingerprint: str | Mapping[str, Any]
    frequency: str
    factors: tuple[StrategyFactorSpec, ...]
    synthesis_method: str
    model_artifacts: tuple[BundleArtifactRef, ...]
    preprocessing_params: Mapping[str, Any]
    random_seed: int
    rebalance_config: Mapping[str, Any]
    execution_profile: Mapping[str, Any]
    universe_filter: Mapping[str, Any]
    allow_short: bool
    benchmark_instrument: str | None
    performance_summary: Mapping[str, Any]
    backtest_returns: BundleArtifactRef
    execution_result_ref: str
    execution_result_sha256: str
    strategy_id: str = "strategy"
    backtest_nav: BundleArtifactRef | None = None
    schema_version: str = STRATEGY_BUNDLE_SCHEMA_VERSION
    bundle_hash: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != STRATEGY_BUNDLE_SCHEMA_VERSION:
            raise StrategyBundleError(
                f"Unsupported StrategyBundle schema_version: {self.schema_version!r}"
            )
        object.__setattr__(
            self, "strategy_id", _require_nonempty_str(self.strategy_id, "strategy_id")
        )
        object.__setattr__(
            self, "slot_id", _require_nonempty_str(self.slot_id, "slot_id")
        )
        object.__setattr__(
            self,
            "slot_config_hash",
            _validate_sha256_hex(self.slot_config_hash, "slot_config_hash"),
        )
        object.__setattr__(
            self,
            "calendar_hash",
            _validate_sha256_hex(self.calendar_hash, "calendar_hash"),
        )
        if isinstance(self.slot_data_fingerprint, str):
            object.__setattr__(
                self,
                "slot_data_fingerprint",
                _require_nonempty_str(
                    self.slot_data_fingerprint, "slot_data_fingerprint"
                ),
            )
        elif isinstance(self.slot_data_fingerprint, Mapping) and self.slot_data_fingerprint:
            object.__setattr__(
                self,
                "slot_data_fingerprint",
                _normalize_json_value(
                    self.slot_data_fingerprint,
                    field="slot_data_fingerprint",
                    freeze=True,
                ),
            )
        else:
            raise StrategyBundleError(
                "slot_data_fingerprint must be a nonempty string or mapping"
            )

        parsed_freq = Frequency.parse(
            _require_nonempty_str(self.frequency, "frequency")
        )
        if not parsed_freq.is_valid or parsed_freq.canonical is None:
            raise StrategyBundleError(f"Invalid frequency: {self.frequency!r}")
        object.__setattr__(self, "frequency", parsed_freq.canonical)

        if not isinstance(self.factors, Sequence) or isinstance(self.factors, (str, bytes)) or not self.factors:
            raise StrategyBundleError("factors must be a nonempty sequence")
        norm_factors = tuple(
            item if isinstance(item, StrategyFactorSpec) else StrategyFactorSpec.from_mapping(item)
            for item in self.factors
        )
        if len({item.factor_id for item in norm_factors}) != len(norm_factors):
            raise StrategyBundleError("factors must contain distinct factor_id values")
        object.__setattr__(self, "factors", norm_factors)

        object.__setattr__(
            self,
            "synthesis_method",
            _require_nonempty_str(self.synthesis_method, "synthesis_method"),
        )

        if (
            not isinstance(self.model_artifacts, Sequence)
            or isinstance(self.model_artifacts, (str, bytes))
            or not self.model_artifacts
        ):
            raise StrategyBundleError("model_artifacts must be a nonempty sequence")
        norm_models = tuple(
            item if isinstance(item, BundleArtifactRef) else BundleArtifactRef.from_mapping(item)
            for item in self.model_artifacts
        )
        for item in norm_models:
            if item.artifact_type in _FORBIDDEN_MODEL_ARTIFACT_TYPES:
                raise StrategyBundleError(
                    f"Unsupported non-replayable model artifact_type: {item.artifact_type!r}"
                )
        if not any(item.artifact_type in _REPLAYABLE_MODEL_ARTIFACT_TYPES for item in norm_models):
            raise StrategyBundleError(
                "model_artifacts must include at least one concrete replayable model or weights artifact"
            )
        object.__setattr__(self, "model_artifacts", norm_models)

        for map_field in (
            "preprocessing_params",
            "rebalance_config",
            "universe_filter",
        ):
            raw_map = getattr(self, map_field)
            if not isinstance(raw_map, Mapping):
                raise StrategyBundleError(f"{map_field} must be a mapping")
            object.__setattr__(
                self,
                map_field,
                _normalize_json_value(raw_map, field=map_field, freeze=True),
            )

        if type(self.random_seed) is not int:
            raise StrategyBundleError("random_seed must be an integer")

        if not isinstance(self.execution_profile, Mapping):
            raise StrategyBundleError("execution_profile must be a mapping")
        profile_plain = _normalize_json_value(
            self.execution_profile, field="execution_profile", freeze=False
        )
        try:
            validated_profile = ExecutionProfile(**profile_plain)
        except (TypeError, ValueError) as exc:
            raise StrategyBundleError(f"Invalid execution_profile: {exc}") from exc
        object.__setattr__(
            self,
            "execution_profile",
            _normalize_json_value(
                asdict(validated_profile), field="execution_profile", freeze=True
            ),
        )

        if type(self.allow_short) is not bool:
            raise StrategyBundleError("allow_short must be a boolean")
        if self.allow_short != validated_profile.allow_short:
            raise StrategyBundleError(
                f"allow_short ({self.allow_short!r}) disagrees with execution_profile.allow_short ({validated_profile.allow_short!r})"
            )
        if self.benchmark_instrument is not None:
            object.__setattr__(
                self,
                "benchmark_instrument",
                _require_nonempty_str(self.benchmark_instrument, "benchmark_instrument"),
            )
        if self.benchmark_instrument != validated_profile.benchmark_instrument:
            raise StrategyBundleError(
                f"benchmark_instrument ({self.benchmark_instrument!r}) disagrees with execution_profile.benchmark_instrument ({validated_profile.benchmark_instrument!r})"
            )

        if not isinstance(self.performance_summary, Mapping):
            raise StrategyBundleError("performance_summary must be a mapping")
        perf_frozen = _normalize_json_value(
            self.performance_summary, field="performance_summary", freeze=True
        )
        missing_perf = [k for k in _REQUIRED_PERFORMANCE_KEYS if k not in perf_frozen]
        if missing_perf:
            raise StrategyBundleError(
                f"performance_summary missing required metrics: {missing_perf}"
            )
        for key in _REQUIRED_PERFORMANCE_KEYS:
            val = perf_frozen[key]
            if type(val) not in (int, float) or not math.isfinite(float(val)):
                raise StrategyBundleError(
                    f"performance_summary[{key!r}] must be a finite number"
                )
        for key in _OPTIONAL_FLOAT_PERFORMANCE_KEYS:
            if key in perf_frozen and perf_frozen[key] is not None:
                val = perf_frozen[key]
                if type(val) not in (int, float) or not math.isfinite(float(val)):
                    raise StrategyBundleError(
                        f"performance_summary[{key!r}] must be a finite number or null"
                    )
        if "trade_count" in perf_frozen and perf_frozen["trade_count"] is not None:
            tc = perf_frozen["trade_count"]
            if type(tc) is not int or tc < 0:
                raise StrategyBundleError(
                    "performance_summary['trade_count'] must be a non-negative integer"
                )
        object.__setattr__(self, "performance_summary", perf_frozen)

        backtest_ret = (
            self.backtest_returns
            if isinstance(self.backtest_returns, BundleArtifactRef)
            else BundleArtifactRef.from_mapping(self.backtest_returns)
        )
        object.__setattr__(self, "backtest_returns", backtest_ret)

        if self.backtest_nav is None:
            raise StrategyBundleError("backtest_nav is required")
        backtest_nav = (
            self.backtest_nav
            if isinstance(self.backtest_nav, BundleArtifactRef)
            else BundleArtifactRef.from_mapping(self.backtest_nav)
        )
        object.__setattr__(self, "backtest_nav", backtest_nav)

        object.__setattr__(
            self,
            "execution_result_ref",
            _validate_rel_posix_path(
                self.execution_result_ref, "execution_result_ref"
            ),
        )
        object.__setattr__(
            self,
            "execution_result_sha256",
            _validate_sha256_hex(
                self.execution_result_sha256, "execution_result_sha256"
            ),
        )

        # Ensure all relative paths inside the bundle are distinct.
        rel_paths = [item.path for item in norm_models] + [backtest_ret.path]
        if self.backtest_nav is not None:
            rel_paths.append(self.backtest_nav.path)
        rel_paths.append(self.execution_result_ref)
        if len(set(rel_paths)) != len(rel_paths):
            raise StrategyBundleError(
                "StrategyBundle contains duplicate artifact paths"
            )

        expected_hash = compute_bundle_hash(self._manifest_payload_without_hash())
        if not self.bundle_hash:
            object.__setattr__(self, "bundle_hash", expected_hash)
        else:
            supplied = _validate_sha256_hex(self.bundle_hash, "bundle_hash")
            if supplied != expected_hash:
                raise StrategyBundleError(
                    f"bundle_hash mismatch: expected {expected_hash}, got {supplied}"
                )
            object.__setattr__(self, "bundle_hash", supplied)

    @property
    def metadata(self) -> dict[str, str]:
        return market_db_identity(self.slot_data_fingerprint)

    def _manifest_payload_without_hash(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "strategy_id": self.strategy_id,
            "slot_id": self.slot_id,
            "slot_config_hash": self.slot_config_hash,
            "calendar_hash": self.calendar_hash,
            "slot_data_fingerprint": _unfreeze_json_value(self.slot_data_fingerprint),
            "frequency": self.frequency,
            "factors": [item.to_dict() for item in self.factors],
            "synthesis_method": self.synthesis_method,
            "model_artifacts": [item.to_dict() for item in self.model_artifacts],
            "preprocessing_params": _unfreeze_json_value(self.preprocessing_params),
            "random_seed": self.random_seed,
            "rebalance_config": _unfreeze_json_value(self.rebalance_config),
            "execution_profile": _unfreeze_json_value(self.execution_profile),
            "universe_filter": _unfreeze_json_value(self.universe_filter),
            "allow_short": self.allow_short,
            "benchmark_instrument": self.benchmark_instrument,
            "performance_summary": _unfreeze_json_value(self.performance_summary),
            "backtest_returns": self.backtest_returns.to_dict(),
            "backtest_nav": self.backtest_nav.to_dict() if self.backtest_nav is not None else None,
            "execution_result_ref": self.execution_result_ref,
            "execution_result_sha256": self.execution_result_sha256,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    def to_dict(self) -> dict[str, Any]:
        payload = self._manifest_payload_without_hash()
        payload["bundle_hash"] = self.bundle_hash
        return payload

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> StrategyBundle:
        if not isinstance(raw, Mapping):
            raise StrategyBundleError("StrategyBundle manifest must be a JSON object")
        expected = {
            "schema_version",
            "strategy_id",
            "slot_id",
            "slot_config_hash",
            "calendar_hash",
            "slot_data_fingerprint",
            "frequency",
            "factors",
            "synthesis_method",
            "model_artifacts",
            "preprocessing_params",
            "random_seed",
            "rebalance_config",
            "execution_profile",
            "universe_filter",
            "allow_short",
            "benchmark_instrument",
            "performance_summary",
            "backtest_returns",
            "backtest_nav",
            "execution_result_ref",
            "execution_result_sha256",
            "bundle_hash",
        }
        missing = expected - set(raw)
        if missing:
            raise StrategyBundleError(
                f"StrategyBundle manifest missing required fields: {sorted(missing)}"
            )
        extra = set(raw) - expected - {"metadata"}
        if extra:
            raise StrategyBundleError(
                f"StrategyBundle manifest contains unknown fields: {sorted(extra)}"
            )
        if raw["schema_version"] != STRATEGY_BUNDLE_SCHEMA_VERSION:
            raise StrategyBundleError(
                f"Unsupported StrategyBundle schema_version: {raw['schema_version']!r}"
            )
        if raw.get("metadata", {}) != market_db_identity(raw["slot_data_fingerprint"]):
            raise StrategyBundleError("metadata must match the slot_data_fingerprint market database")
        _validate_sha256_hex(raw["bundle_hash"], "bundle_hash")
        return cls(
            schema_version=raw["schema_version"],
            strategy_id=raw["strategy_id"],
            slot_id=raw["slot_id"],
            slot_config_hash=raw["slot_config_hash"],
            calendar_hash=raw["calendar_hash"],
            slot_data_fingerprint=raw["slot_data_fingerprint"],
            frequency=raw["frequency"],
            factors=tuple(
                StrategyFactorSpec.from_mapping(item) for item in raw["factors"]
            ),
            synthesis_method=raw["synthesis_method"],
            model_artifacts=tuple(
                BundleArtifactRef.from_mapping(item) for item in raw["model_artifacts"]
            ),
            preprocessing_params=raw["preprocessing_params"],
            random_seed=raw["random_seed"],
            rebalance_config=raw["rebalance_config"],
            execution_profile=raw["execution_profile"],
            universe_filter=raw["universe_filter"],
            allow_short=raw["allow_short"],
            benchmark_instrument=raw["benchmark_instrument"],
            performance_summary=raw["performance_summary"],
            backtest_returns=BundleArtifactRef.from_mapping(raw["backtest_returns"]),
            backtest_nav=BundleArtifactRef.from_mapping(raw["backtest_nav"]),
            execution_result_ref=raw["execution_result_ref"],
            execution_result_sha256=raw["execution_result_sha256"],
            bundle_hash=raw["bundle_hash"],
        )


def _artifact_checks(bundle: StrategyBundle) -> list[tuple[str, str, int | None, str]]:
    checks: list[tuple[str, str, int | None, str]] = [
        (item.path, item.sha256, item.size_bytes, f"model_artifacts[{idx}]")
        for idx, item in enumerate(bundle.model_artifacts)
    ]
    checks.append(
        (
            bundle.backtest_returns.path,
            bundle.backtest_returns.sha256,
            bundle.backtest_returns.size_bytes,
            "backtest_returns",
        )
    )
    if bundle.backtest_nav is not None:
        checks.append(
            (
                bundle.backtest_nav.path,
                bundle.backtest_nav.sha256,
                bundle.backtest_nav.size_bytes,
                "backtest_nav",
            )
        )
    checks.append(
        (
            bundle.execution_result_ref,
            bundle.execution_result_sha256,
            None,
            "execution_result_ref",
        )
    )
    return checks


def write_strategy_bundle(
    bundle: StrategyBundle,
    output_dir: str | Path,
    *,
    source_files: Mapping[str, str | Path] | None = None,
) -> Path:
    """Materialize an immutable StrategyBundle directory and verify it fail-closed."""
    if not isinstance(bundle, StrategyBundle):
        raise StrategyBundleError("bundle must be a StrategyBundle instance")
    root = Path(output_dir).expanduser().resolve()
    manifest_path = root / MANIFEST_FILENAME
    if manifest_path.exists():
        raise StrategyBundleError(
            f"StrategyBundle manifest already exists and is immutable: {manifest_path}"
        )
    root.mkdir(parents=True, exist_ok=True)
    if source_files is not None:
        for rel_path, src in source_files.items():
            clean_rel = _validate_rel_posix_path(rel_path, "source_files key")
            src_path = Path(src).expanduser().resolve()
            if not src_path.is_file():
                raise StrategyBundleError(
                    f"source_files[{clean_rel!r}] does not exist: {src_path}"
                )
            dest = root / clean_rel
            if not dest.resolve().is_relative_to(root):
                raise StrategyBundleError(
                    f"source_files[{clean_rel!r}] escapes output_dir"
                )
            dest.parent.mkdir(parents=True, exist_ok=True)
            if src_path != dest.resolve():
                shutil.copy2(src_path, dest)

    for rel_path, expected_sha, expected_size, field in _artifact_checks(bundle):
        resolved = _resolve_contained_file(root, rel_path, field)
        actual_sha = sha256_file(resolved)
        if actual_sha != expected_sha:
            raise StrategyBundleError(
                f"{field} sha256 mismatch for {rel_path!r}: "
                f"expected {expected_sha}, got {actual_sha}"
            )
        if expected_size is not None and resolved.stat().st_size != expected_size:
            raise StrategyBundleError(
                f"{field} size mismatch for {rel_path!r}"
            )

    temporary = root / f".{MANIFEST_FILENAME}.tmp"
    try:
        temporary.write_text(
            json.dumps(
                bundle.to_dict(),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(manifest_path)
    finally:
        temporary.unlink(missing_ok=True)

    verify_strategy_bundle(root)
    return root


_METHODS_REQUIRING_MODEL_BUNDLE = frozenset(
    {"weighted_sum", "lasso", "lightgbm", "size_conditional_elasticnet"}
)


def _is_finite_number(value: Any) -> bool:
    return (
        type(value) in (int, float)
        or (hasattr(value, "item") and not isinstance(value, (bytes, bytearray, bool)) and type(value.item()) in (int, float))
    ) and math.isfinite(float(value))


def _has_finite_numeric_weights(weights: Any, *, require_nonzero: bool = False) -> bool:
    if not isinstance(weights, Mapping) or not weights:
        return False
    has_nonzero = False
    for key, val in weights.items():
        if not isinstance(key, str) or not key.strip():
            return False
        if type(val) is bool or not _is_finite_number(val):
            return False
        if abs(float(val)) > 0.0:
            has_nonzero = True
    return has_nonzero if require_nonzero else True


def _has_nonempty_str_sequence(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and bool(value)
        and all(isinstance(item, str) and bool(item.strip()) for item in value)
    )


def _validate_parent_baseline_ref(
    pb: Any,
    *,
    idx: int,
    model_idx: int,
    model_dir: Path,
    root_resolved: Path,
    resolved_files: Mapping[str, Path],
) -> None:
    if not isinstance(pb, Mapping) or not pb:
        raise StrategyBundleError(
            f"model_artifacts[{idx}] index.models[{model_idx}] has invalid parent_baseline"
        )
    pb_rel = pb.get("path")
    pb_sha = pb.get("sha256")
    clean_rel = _validate_rel_posix_path(
        pb_rel, f"model_artifacts[{idx}].index.models[{model_idx}].parent_baseline.path"
    )
    expected_sha = _validate_sha256_hex(
        pb_sha, f"model_artifacts[{idx}].index.models[{model_idx}].parent_baseline.sha256"
    )
    candidate = model_dir / clean_rel
    if candidate.is_symlink():
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root_resolved):
            raise StrategyBundleError(
                f"model_artifacts[{idx}] parent_baseline symlink escapes bundle directory: {clean_rel!r}"
            )
    else:
        resolved = candidate.resolve()
    if not resolved.is_relative_to(root_resolved) or resolved == root_resolved:
        raise StrategyBundleError(
            f"model_artifacts[{idx}] parent_baseline resolves outside bundle directory: {clean_rel!r}"
        )
    if not resolved.is_file():
        raise StrategyBundleError(
            f"model_artifacts[{idx}] missing parent_baseline file: {clean_rel!r}"
        )
    rel_to_root = resolved.relative_to(root_resolved).as_posix()
    if rel_to_root not in resolved_files:
        raise StrategyBundleError(
            f"model_artifacts[{idx}] parent_baseline {rel_to_root!r} is not registered in bundle model_artifacts"
        )
    actual_sha = sha256_file(resolved)
    if actual_sha != expected_sha:
        raise StrategyBundleError(
            f"model_artifacts[{idx}] parent_baseline sha256 mismatch for {clean_rel!r}: "
            f"expected {expected_sha}, got {actual_sha}"
        )


def _validate_lightgbm_model_str(
    content: Any,
    *,
    context: str,
    expected_num_features: int | None = None,
) -> None:
    if not isinstance(content, str) or not content.strip():
        raise StrategyBundleError(f"{context} missing model content")
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise StrategyBundleError(
            f"{context} cannot be validated without lightgbm installed"
        ) from exc
    try:
        booster = lgb.Booster(model_str=content)
    except Exception as exc:
        raise StrategyBundleError(
            f"{context} failed to load LightGBM Booster: {exc}"
        ) from exc
    if booster.num_trees() <= 0 or booster.num_feature() <= 0:
        raise StrategyBundleError(
            f"{context} contains empty LightGBM Booster "
            f"(trees={booster.num_trees()}, features={booster.num_feature()})"
        )
    if expected_num_features is not None and booster.num_feature() != expected_num_features:
        raise StrategyBundleError(
            f"{context} feature count mismatch: booster expects {booster.num_feature()}, "
            f"metadata lists {expected_num_features}"
        )


def _validate_model_artifacts(
    bundle: StrategyBundle,
    resolved_files: Mapping[str, Path],
    root_resolved: Path,
) -> None:
    if bundle.synthesis_method in _METHODS_REQUIRING_MODEL_BUNDLE and not any(
        item.artifact_type == "model_bundle" for item in bundle.model_artifacts
    ):
        raise StrategyBundleError(
            f"synthesis_method {bundle.synthesis_method!r} requires at least one model_bundle artifact"
        )

    for idx, item in enumerate(bundle.model_artifacts):
        model_path = resolved_files[item.path]
        if model_path.stat().st_size <= 0:
            raise StrategyBundleError(
                f"model_artifacts[{idx}] is empty: {item.path!r}"
            )
        if item.artifact_type == "lgb_model":
            try:
                content = model_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise StrategyBundleError(
                    f"model_artifacts[{idx}] unreadable lgb_model {item.path!r}: {exc}"
                ) from exc
            _validate_lightgbm_model_str(
                content,
                context=f"model_artifacts[{idx}] lgb_model {item.path!r}",
            )
        elif item.artifact_type == "model_weights" and model_path.suffix.lower() == ".csv":
            with model_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                rows = [row for row in reader if row]
            if len(rows) < 2:
                raise StrategyBundleError(
                    f"model_artifacts[{idx}] model_weights CSV must contain header and weights"
                )
        elif item.artifact_type == "model_index":
            try:
                index_payload = json.loads(
                    model_path.read_text(encoding="utf-8"),
                    parse_constant=_reject_json_constant,
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise StrategyBundleError(
                    f"model_artifacts[{idx}] invalid model_index JSON: {exc}"
                ) from exc
            if not isinstance(index_payload, Mapping) or not isinstance(
                index_payload.get("models"), list
            ) or not index_payload["models"]:
                raise StrategyBundleError(
                    f"model_artifacts[{idx}] model_index must contain non-empty models list"
                )
            if index_payload.get("requires_parent_baseline", False):
                expected_count = index_payload.get("expected_model_count")
                if type(expected_count) is not int or expected_count != len(index_payload["models"]):
                    raise StrategyBundleError(
                        f"model_artifacts[{idx}] incomplete parent bundle: "
                        f"expected {expected_count!r} models, found {len(index_payload['models'])}"
                    )
        elif item.artifact_type == "model_bundle":
            try:
                model_payload = json.loads(
                    model_path.read_text(encoding="utf-8"),
                    parse_constant=_reject_json_constant,
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise StrategyBundleError(
                    f"model_artifacts[{idx}] invalid JSON: {exc}"
                ) from exc
            if not isinstance(model_payload, Mapping) or not model_payload:
                raise StrategyBundleError(
                    f"model_artifacts[{idx}] model_bundle must be a nonempty JSON object"
                )
            idx_block = model_payload.get("index")
            if not isinstance(idx_block, Mapping):
                raise StrategyBundleError(
                    f"model_artifacts[{idx}] model_bundle missing required index block"
                )
            models_list = idx_block.get("models")
            if (
                not isinstance(models_list, list)
                or not models_list
                or any(not isinstance(m, Mapping) for m in models_list)
            ):
                raise StrategyBundleError(
                    f"model_artifacts[{idx}] model_bundle has empty or invalid index.models"
                )
            declared_type = model_payload.get("model_type") or (
                model_payload.get("metadata", {}).get("method")
                if isinstance(model_payload.get("metadata"), Mapping)
                else None
            )
            if declared_type is not None and str(declared_type) != bundle.synthesis_method:
                raise StrategyBundleError(
                    f"model_artifacts[{idx}] model_bundle type {declared_type!r} disagrees with synthesis_method {bundle.synthesis_method!r}"
                )
            method = bundle.synthesis_method
            requires_parent = bool(idx_block.get("requires_parent_baseline", False))
            models_dict = model_payload.get("models")
            if method == "lightgbm" and (not isinstance(models_dict, Mapping) or not models_dict):
                raise StrategyBundleError(
                    f"model_artifacts[{idx}] lightgbm model_bundle has empty models map"
                )

            for m_idx, model_entry in enumerate(models_list):
                if method == "weighted_sum":
                    if not _has_finite_numeric_weights(
                        model_entry.get("weights"), require_nonzero=True
                    ) or not _has_nonempty_str_sequence(model_entry.get("feature_names")):
                        raise StrategyBundleError(
                            f"model_artifacts[{idx}] weighted_sum model[{m_idx}] missing valid weights or feature_names"
                        )
                elif method == "lasso":
                    pb = model_entry.get("parent_baseline")
                    if requires_parent and (not isinstance(pb, Mapping) or not pb):
                        raise StrategyBundleError(
                            f"model_artifacts[{idx}] lasso model[{m_idx}] missing required parent_baseline"
                        )
                    if pb is not None and pb != {}:
                        _validate_parent_baseline_ref(
                            pb,
                            idx=idx,
                            model_idx=m_idx,
                            model_dir=model_path.parent,
                            root_resolved=root_resolved,
                            resolved_files=resolved_files,
                        )
                    has_weights = _has_finite_numeric_weights(
                        model_entry.get("weights")
                    ) and _has_nonempty_str_sequence(model_entry.get("feature_names"))
                    is_blend_only = bool(model_entry.get("blend_only")) and (
                        _has_finite_numeric_weights(model_entry.get("blend_weights"))
                        or (isinstance(pb, Mapping) and bool(pb))
                    )
                    if not (has_weights or is_blend_only):
                        raise StrategyBundleError(
                            f"model_artifacts[{idx}] lasso model[{m_idx}] missing replayable weights or blend baseline"
                        )
                elif method == "lightgbm":
                    mfile = model_entry.get("model_file")
                    if (
                        not isinstance(mfile, str)
                        or not mfile.strip()
                        or not _has_nonempty_str_sequence(model_entry.get("feature_names"))
                    ):
                        raise StrategyBundleError(
                            f"model_artifacts[{idx}] lightgbm model[{m_idx}] missing model_file or feature_names"
                        )
                    content = models_dict.get(mfile) if isinstance(models_dict, Mapping) else None
                    _validate_lightgbm_model_str(
                        content,
                        context=f"model_artifacts[{idx}] lightgbm model[{m_idx}] ({mfile!r})",
                        expected_num_features=len(model_entry["feature_names"]),
                    )
                elif method == "size_conditional_elasticnet":
                    coeffs = model_entry.get("coefficients")
                    global_coeffs = (
                        coeffs.get("global_coefficients")
                        if isinstance(coeffs, Mapping)
                        else None
                    )
                    if (
                        not _has_finite_numeric_weights(global_coeffs)
                        or not _has_nonempty_str_sequence(model_entry.get("feature_names"))
                        or not isinstance(model_entry.get("selected_policy"), Mapping)
                        or not model_entry["selected_policy"]
                        or type(model_entry.get("intercept")) is bool
                        or not _is_finite_number(model_entry.get("intercept"))
                    ):
                        raise StrategyBundleError(
                            f"model_artifacts[{idx}] size_conditional_elasticnet model[{m_idx}] missing replayable coefficients or policy"
                        )
                else:
                    has_weights = _has_finite_numeric_weights(model_entry.get("weights"))
                    has_coeffs = isinstance(model_entry.get("coefficients"), Mapping) and bool(
                        model_entry["coefficients"]
                    )
                    mfile = model_entry.get("model_file")
                    has_lgb_ref = (
                        isinstance(mfile, str)
                        and bool(mfile.strip())
                        and isinstance(models_dict, Mapping)
                        and isinstance(models_dict.get(mfile), str)
                        and bool(models_dict[mfile].strip())
                    )
                    if has_lgb_ref:
                        _validate_lightgbm_model_str(
                            models_dict[mfile],
                            context=f"model_artifacts[{idx}] model[{m_idx}] ({mfile!r})",
                        )
                    if not (has_weights or has_coeffs or has_lgb_ref):
                        raise StrategyBundleError(
                            f"model_artifacts[{idx}] model[{m_idx}] missing replayable weights or model content"
                        )

            if requires_parent:
                expected_count = idx_block.get("expected_model_count")
                if type(expected_count) is not int or expected_count != len(models_list):
                    raise StrategyBundleError(
                        f"model_artifacts[{idx}] incomplete parent bundle: "
                        f"expected {expected_count!r} models, found {len(models_list)}"
                    )


def verify_strategy_bundle(bundle_dir: str | Path) -> StrategyBundle:
    """Verify manifest schema, file containment, SHA-256 checksums, and bundle_hash."""
    root = Path(bundle_dir).expanduser()
    if not root.is_dir():
        raise StrategyBundleError(f"StrategyBundle directory does not exist: {root}")
    root_resolved = root.resolve()
    manifest_path = root_resolved / MANIFEST_FILENAME
    if manifest_path.is_symlink() and not manifest_path.resolve().is_relative_to(
        root_resolved
    ):
        raise StrategyBundleError("StrategyBundle manifest symlink escapes bundle root")
    if not manifest_path.is_file():
        raise StrategyBundleError(
            f"StrategyBundle manifest missing: {manifest_path}"
        )
    try:
        raw_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise StrategyBundleError(
            f"Failed to parse StrategyBundle manifest: {exc}"
        ) from exc

    bundle = StrategyBundle.from_mapping(raw_manifest)
    resolved_files: dict[str, Path] = {}
    for rel_path, expected_sha, expected_size, field in _artifact_checks(bundle):
        resolved = _resolve_contained_file(root_resolved, rel_path, field)
        actual_sha = sha256_file(resolved)
        if actual_sha != expected_sha:
            raise StrategyBundleError(
                f"{field} sha256 mismatch for {rel_path!r}: "
                f"expected {expected_sha}, got {actual_sha}"
            )
        if expected_size is not None and resolved.stat().st_size != expected_size:
            raise StrategyBundleError(f"{field} size mismatch for {rel_path!r}")
        resolved_files[rel_path] = resolved

    _validate_model_artifacts(bundle, resolved_files, root_resolved)

    ret_rows = _read_validated_series_csv(
        resolved_files[bundle.backtest_returns.path],
        value_column="net_return",
        field="backtest_returns",
    )
    nav_rows = _read_validated_series_csv(
        resolved_files[bundle.backtest_nav.path],
        value_column="nav",
        field="backtest_nav",
    )
    if [row[0] for row in ret_rows] != [row[0] for row in nav_rows]:
        raise StrategyBundleError(
            "backtest_nav dates do not match backtest_returns dates"
        )

    exec_path = resolved_files[bundle.execution_result_ref]
    try:
        exec_payload = json.loads(
            exec_path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise StrategyBundleError(
            f"Failed to parse execution_result_ref JSON: {exc}"
        ) from exc
    from tools.agent_harness.run_result import validate_execution_result_payload

    try:
        validate_execution_result_payload(
            exec_payload, require_succeeded=True, require_identity=True
        )
    except ValueError as exc:
        raise StrategyBundleError(f"Invalid execution_result_ref: {exc}") from exc

    return bundle


def _read_validated_series_csv(
    path: Path, *, value_column: str, field: str
) -> list[tuple[str, float]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "date" not in reader.fieldnames or value_column not in reader.fieldnames:
                raise StrategyBundleError(
                    f"{field} CSV must contain 'date' and {value_column!r} columns"
                )
            rows: list[tuple[str, float]] = []
            for row in reader:
                date_val = str(row.get("date") or "").strip()
                raw_num = row.get(value_column)
                if not date_val or raw_num is None or str(raw_num).strip() == "":
                    raise StrategyBundleError(
                        f"{field} CSV contains blank date or {value_column}"
                    )
                try:
                    num = float(raw_num)
                except ValueError as exc:
                    raise StrategyBundleError(
                        f"{field} CSV contains non-numeric {value_column}: {raw_num!r}"
                    ) from exc
                if not math.isfinite(num):
                    raise StrategyBundleError(
                        f"{field} CSV contains non-finite {value_column}"
                    )
                rows.append((date_val, num))
    except OSError as exc:
        raise StrategyBundleError(f"Failed to read {field} CSV: {exc}") from exc
    if not rows:
        raise StrategyBundleError(f"{field} CSV must contain at least one row")
    return rows


def load_strategy_bundle(bundle_dir: str | Path) -> StrategyBundle:
    """Verify and load an immutable StrategyBundle from disk."""
    return verify_strategy_bundle(bundle_dir)


__all__ = [
    "BundleArtifactRef",
    "MANIFEST_FILENAME",
    "STRATEGY_BUNDLE_SCHEMA_VERSION",
    "StrategyBundle",
    "StrategyBundleError",
    "StrategyFactorSpec",
    "compute_bundle_hash",
    "load_strategy_bundle",
    "sha256_file",
    "verify_strategy_bundle",
    "write_strategy_bundle",
]
