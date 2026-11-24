"""Output-layer deployment target contracts, manifest verification, and safe bundle loading."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable

import pandas as pd

from factor_synthesis.strategy_bundle import (
    StrategyBundle,
    StrategyBundleError,
    compute_bundle_hash,
    sha256_file,
    verify_strategy_bundle,
)
from utils.time_axis import Frequency

DEPLOY_ARTIFACT_MANIFEST_SCHEMA_VERSION = "deploy_artifact_manifest.v1"
DEPLOY_MANIFEST_FILENAME = "deploy_manifest.json"

TARGET_BUNDLE_INVALID = "TARGET_BUNDLE_INVALID"
TARGET_SLOT_INCOMPATIBLE = "TARGET_SLOT_INCOMPATIBLE"
TARGET_AUX_FEATURE_UNSUPPORTED = "TARGET_AUX_FEATURE_UNSUPPORTED"
TARGET_EXECUTION_CONTRACT_UNSUPPORTED = "TARGET_EXECUTION_CONTRACT_UNSUPPORTED"
TARGET_SIGNAL_SOURCE_MISSING = "TARGET_SIGNAL_SOURCE_MISSING"
TARGET_ARTIFACT_INTEGRITY_FAILED = "TARGET_ARTIFACT_INTEGRITY_FAILED"
TARGET_ALIGNMENT_MISMATCH = "TARGET_ALIGNMENT_MISMATCH"

_HEX_CHARS = frozenset("0123456789abcdef")


class DeployTargetError(ValueError):
    """Raised when a StrategyBundle or deployment artifact violates DeployTarget contracts."""

    def __init__(self, message: str, *, code: str = TARGET_BUNDLE_INVALID) -> None:
        if not isinstance(code, str) or not code.strip() or code != code.strip():
            raise ValueError("DeployTargetError code must be a nonempty string")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("DeployTargetError message must be a nonempty string")
        self.code = code
        formatted = message if message.startswith(f"[{code}]") else f"[{code}] {message}"
        super().__init__(formatted)


def _reject_json_constant(token: str) -> Any:
    raise DeployTargetError(
        f"Non-standard JSON constant {token!r} is not permitted in deployment artifacts",
        code=TARGET_ARTIFACT_INTEGRITY_FAILED,
    )


def _require_nonempty_str(
    value: Any,
    field_name: str,
    *,
    code: str = TARGET_BUNDLE_INVALID,
) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise DeployTargetError(
            f"{field_name} must be a nonempty string without surrounding whitespace",
            code=code,
        )
    return value


def _validate_sha256_hex(
    value: Any,
    field_name: str,
    *,
    code: str = TARGET_ARTIFACT_INTEGRITY_FAILED,
) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in _HEX_CHARS for c in value):
        raise DeployTargetError(
            f"{field_name} must be a 64-character lowercase SHA-256 hex digest",
            code=code,
        )
    return value


def _validate_rel_posix_path(
    value: Any,
    field_name: str,
    *,
    code: str = TARGET_ARTIFACT_INTEGRITY_FAILED,
) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise DeployTargetError(
            f"{field_name} must be a nonempty relative POSIX path",
            code=code,
        )
    if "\x00" in value or "\\" in value:
        raise DeployTargetError(
            f"{field_name} contains forbidden path characters: {value!r}",
            code=code,
        )
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise DeployTargetError(
            f"{field_name} must not escape output directory: {value!r}",
            code=code,
        )
    normalized = pure.as_posix()
    if normalized in ("", ".") or normalized != value:
        raise DeployTargetError(
            f"{field_name} must be a canonical relative POSIX path: {value!r}",
            code=code,
        )
    return normalized


def _normalize_json_value(
    value: Any,
    *,
    field_name: str,
    freeze: bool,
    code: str = TARGET_BUNDLE_INVALID,
    ancestors: frozenset[int] = frozenset(),
) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise DeployTargetError(
                f"{field_name} contains non-finite float value",
                code=code,
            )
        return 0.0 if value == 0.0 else value
    if hasattr(value, "item") and not isinstance(value, (bytes, bytearray)):
        return _normalize_json_value(
            value.item(),
            field_name=field_name,
            freeze=freeze,
            code=code,
            ancestors=ancestors,
        )
    if id(value) in ancestors:
        raise DeployTargetError(
            f"{field_name} contains cyclic structure",
            code=code,
        )
    ancestors = ancestors | {id(value)}
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise DeployTargetError(
                    f"{field_name} mapping keys must be strings",
                    code=code,
                )
            out[key] = _normalize_json_value(
                item,
                field_name=f"{field_name}.{key}",
                freeze=freeze,
                code=code,
                ancestors=ancestors,
            )
        return MappingProxyType(out) if freeze else out
    if isinstance(value, (list, tuple)):
        items = [
            _normalize_json_value(
                item,
                field_name=f"{field_name}[{idx}]",
                freeze=freeze,
                code=code,
                ancestors=ancestors,
            )
            for idx, item in enumerate(value)
        ]
        return tuple(items) if freeze else items
    raise DeployTargetError(
        f"{field_name} contains unsupported JSON type {type(value).__name__}",
        code=code,
    )


def _unfreeze_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _unfreeze_json_value(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_unfreeze_json_value(item) for item in value]
    return value


def _validate_utc_iso_timestamp(value: Any, field_name: str) -> str:
    raw = _require_nonempty_str(value, field_name, code=TARGET_ARTIFACT_INTEGRITY_FAILED)
    try:
        ts = pd.Timestamp(raw)
    except (ValueError, TypeError) as exc:
        raise DeployTargetError(
            f"{field_name} must be a valid UTC ISO-8601 timestamp: {raw!r}",
            code=TARGET_ARTIFACT_INTEGRITY_FAILED,
        ) from exc
    if pd.isna(ts) or ts.tzinfo is None:
        raise DeployTargetError(
            f"{field_name} must include an explicit UTC timezone: {raw!r}",
            code=TARGET_ARTIFACT_INTEGRITY_FAILED,
        )
    utc_offset = ts.utcoffset()
    if utc_offset is None or utc_offset.total_seconds() != 0:
        raise DeployTargetError(
            f"{field_name} must be in UTC (offset 0), got {raw!r}",
            code=TARGET_ARTIFACT_INTEGRITY_FAILED,
        )
    return raw


def resolve_path_checked(
    path: Path,
    *,
    label: str,
    code: str = TARGET_ARTIFACT_INTEGRITY_FAILED,
) -> Path:
    """Resolve a filesystem path, converting OSError and symlink-loop RuntimeError to DeployTargetError."""
    try:
        return path.resolve()
    except (OSError, RuntimeError) as exc:
        raise DeployTargetError(
            f"Failed to resolve {label} ({path}): {exc}",
            code=code,
        ) from exc


def _resolve_contained_file(
    root_dir: Path,
    rel_path: str,
    field_name: str,
) -> Path:
    clean_rel = _validate_rel_posix_path(
        rel_path, field_name, code=TARGET_ARTIFACT_INTEGRITY_FAILED
    )
    root_resolved = resolve_path_checked(
        root_dir, label="output directory", code=TARGET_ARTIFACT_INTEGRITY_FAILED
    )
    candidate = root_dir / clean_rel
    if candidate.is_symlink():
        resolved = resolve_path_checked(
            candidate, label=field_name, code=TARGET_ARTIFACT_INTEGRITY_FAILED
        )
        if not resolved.is_relative_to(root_resolved):
            raise DeployTargetError(
                f"{field_name} symlink escapes output directory: {clean_rel!r}",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
    else:
        resolved = resolve_path_checked(
            candidate, label=field_name, code=TARGET_ARTIFACT_INTEGRITY_FAILED
        )
    if not resolved.is_relative_to(root_resolved) or resolved == root_resolved:
        raise DeployTargetError(
            f"{field_name} resolves outside output directory: {clean_rel!r}",
            code=TARGET_ARTIFACT_INTEGRITY_FAILED,
        )
    if not resolved.is_file():
        raise DeployTargetError(
            f"{field_name} missing referenced file: {clean_rel!r}",
            code=TARGET_ARTIFACT_INTEGRITY_FAILED,
        )
    return resolved


def compute_deploy_manifest_hash(payload_without_hash: Mapping[str, Any]) -> str:
    """Compute canonical SHA-256 digest of a DeployArtifactManifest payload without manifest_hash."""
    clean = {k: v for k, v in payload_without_hash.items() if k != "manifest_hash"}
    canonical = json.dumps(
        _unfreeze_json_value(clean),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CompatibilityIssue:
    """Immutable compatibility diagnostic raised during DeployTarget preflight."""

    code: str
    message: str
    field: str | None = None
    severity: Literal["error", "warning"] = "error"

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _require_nonempty_str(self.code, "issue.code"))
        object.__setattr__(
            self, "message", _require_nonempty_str(self.message, "issue.message")
        )
        if self.field is not None:
            object.__setattr__(
                self, "field", _require_nonempty_str(self.field, "issue.field")
            )
        if self.severity not in ("error", "warning"):
            raise DeployTargetError(
                f"issue.severity must be 'error' or 'warning', got {self.severity!r}",
                code=TARGET_BUNDLE_INVALID,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "field": self.field,
            "severity": self.severity,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> CompatibilityIssue:
        if not isinstance(raw, Mapping):
            raise DeployTargetError("CompatibilityIssue entry must be a mapping")
        missing = {"code", "message"} - set(raw)
        if missing:
            raise DeployTargetError(
                f"CompatibilityIssue missing required fields: {sorted(missing)}"
            )
        extra = set(raw) - {"code", "message", "field", "severity"}
        if extra:
            raise DeployTargetError(
                f"CompatibilityIssue contains unknown fields: {sorted(extra)}"
            )
        return cls(
            code=raw["code"],
            message=raw["message"],
            field=raw.get("field"),
            severity=raw.get("severity", "error"),
        )


@dataclass(frozen=True)
class CompatibilityReport:
    """Immutable preflight compatibility report for a StrategyBundle against a DeployTarget."""

    target_id: str
    compatible: bool
    bundle_hash: str
    slot_id: str
    issues: tuple[CompatibilityIssue, ...] = ()
    warnings: tuple[CompatibilityIssue | str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "target_id", _require_nonempty_str(self.target_id, "target_id")
        )
        if type(self.compatible) is not bool:
            raise DeployTargetError("compatible must be a boolean")
        if self.bundle_hash == "" and not self.compatible:
            object.__setattr__(self, "bundle_hash", "")
        else:
            object.__setattr__(
                self,
                "bundle_hash",
                _validate_sha256_hex(
                    self.bundle_hash, "bundle_hash", code=TARGET_BUNDLE_INVALID
                ),
            )
        if self.slot_id == "" and not self.compatible:
            object.__setattr__(self, "slot_id", "")
        else:
            object.__setattr__(
                self, "slot_id", _require_nonempty_str(self.slot_id, "slot_id")
            )

        if isinstance(self.issues, (str, bytes)) or not isinstance(self.issues, Sequence):
            raise DeployTargetError("issues must be a sequence of CompatibilityIssue")
        norm_issues = tuple(
            item
            if isinstance(item, CompatibilityIssue)
            else CompatibilityIssue.from_mapping(item)
            for item in self.issues
        )
        object.__setattr__(self, "issues", norm_issues)

        if isinstance(self.warnings, (str, bytes)) or not isinstance(
            self.warnings, Sequence
        ):
            raise DeployTargetError("warnings must be a sequence")
        norm_warnings: list[CompatibilityIssue | str] = []
        for idx, item in enumerate(self.warnings):
            if isinstance(item, str):
                norm_warnings.append(
                    _require_nonempty_str(item, f"warnings[{idx}]")
                )
            elif isinstance(item, CompatibilityIssue):
                if item.severity != "warning":
                    raise DeployTargetError(
                        f"warnings[{idx}] CompatibilityIssue must have severity='warning'"
                    )
                norm_warnings.append(item)
            elif isinstance(item, Mapping):
                parsed = CompatibilityIssue.from_mapping(item)
                if parsed.severity != "warning":
                    raise DeployTargetError(
                        f"warnings[{idx}] CompatibilityIssue must have severity='warning'"
                    )
                norm_warnings.append(parsed)
            else:
                raise DeployTargetError(
                    f"warnings[{idx}] must be a string or CompatibilityIssue"
                )
        object.__setattr__(self, "warnings", tuple(norm_warnings))

        if self.compatible and any(item.severity == "error" for item in norm_issues):
            raise DeployTargetError(
                "compatible=True cannot carry error-severity issues"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "compatible": self.compatible,
            "bundle_hash": self.bundle_hash,
            "slot_id": self.slot_id,
            "issues": [item.to_dict() for item in self.issues],
            "warnings": [
                item.to_dict() if isinstance(item, CompatibilityIssue) else item
                for item in self.warnings
            ],
        }


@dataclass(frozen=True)
class DeployArtifactEntry:
    """Content-addressed relative artifact entry inside a deployment output directory."""

    path: str
    sha256: str
    artifact_type: str
    size_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "path",
            _validate_rel_posix_path(
                self.path, "artifact.path", code=TARGET_ARTIFACT_INTEGRITY_FAILED
            ),
        )
        object.__setattr__(
            self,
            "sha256",
            _validate_sha256_hex(
                self.sha256, "artifact.sha256", code=TARGET_ARTIFACT_INTEGRITY_FAILED
            ),
        )
        object.__setattr__(
            self,
            "artifact_type",
            _require_nonempty_str(
                self.artifact_type,
                "artifact.artifact_type",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            ),
        )
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise DeployTargetError(
                "artifact.size_bytes must be a non-negative integer",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "artifact_type": self.artifact_type,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> DeployArtifactEntry:
        if not isinstance(raw, Mapping):
            raise DeployTargetError(
                "artifact entry must be a mapping",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
        expected = {"path", "sha256", "artifact_type", "size_bytes"}
        missing = expected - set(raw)
        if missing:
            raise DeployTargetError(
                f"artifact entry missing required fields: {sorted(missing)}",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
        extra = set(raw) - expected
        if extra:
            raise DeployTargetError(
                f"artifact entry contains unknown fields: {sorted(extra)}",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
        return cls(
            path=raw["path"],
            sha256=raw["sha256"],
            artifact_type=raw["artifact_type"],
            size_bytes=raw["size_bytes"],
        )


@dataclass(frozen=True)
class DeployArtifactManifest:
    """Immutable versioned manifest binding exported target files to a StrategyBundle."""

    target_id: str
    strategy_id: str
    bundle_hash: str
    slot_id: str
    frequency: str
    created_at_utc: str
    artifacts: tuple[DeployArtifactEntry, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = DEPLOY_ARTIFACT_MANIFEST_SCHEMA_VERSION
    manifest_hash: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != DEPLOY_ARTIFACT_MANIFEST_SCHEMA_VERSION:
            raise DeployTargetError(
                f"Unsupported DeployArtifactManifest schema_version: {self.schema_version!r}",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
        for field_name in ("target_id", "strategy_id", "slot_id"):
            object.__setattr__(
                self,
                field_name,
                _require_nonempty_str(
                    getattr(self, field_name),
                    field_name,
                    code=TARGET_ARTIFACT_INTEGRITY_FAILED,
                ),
            )
        object.__setattr__(
            self,
            "bundle_hash",
            _validate_sha256_hex(
                self.bundle_hash, "bundle_hash", code=TARGET_ARTIFACT_INTEGRITY_FAILED
            ),
        )
        parsed_freq = Frequency.parse(
            _require_nonempty_str(
                self.frequency, "frequency", code=TARGET_ARTIFACT_INTEGRITY_FAILED
            )
        )
        if not parsed_freq.is_valid or parsed_freq.canonical is None:
            raise DeployTargetError(
                f"Invalid frequency: {self.frequency!r}",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
        object.__setattr__(self, "frequency", parsed_freq.canonical)
        object.__setattr__(
            self,
            "created_at_utc",
            _validate_utc_iso_timestamp(self.created_at_utc, "created_at_utc"),
        )

        if isinstance(self.artifacts, (str, bytes)) or not isinstance(
            self.artifacts, Sequence
        ):
            raise DeployTargetError(
                "artifacts must be a sequence of DeployArtifactEntry",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
        norm_artifacts = tuple(
            item
            if isinstance(item, DeployArtifactEntry)
            else DeployArtifactEntry.from_mapping(item)
            for item in self.artifacts
        )
        seen_paths: set[str] = set()
        for entry in norm_artifacts:
            if entry.path == DEPLOY_MANIFEST_FILENAME:
                raise DeployTargetError(
                    f"artifacts must not list {DEPLOY_MANIFEST_FILENAME!r}",
                    code=TARGET_ARTIFACT_INTEGRITY_FAILED,
                )
            if entry.path in seen_paths:
                raise DeployTargetError(
                    f"DeployArtifactManifest contains duplicate artifact path: {entry.path!r}",
                    code=TARGET_ARTIFACT_INTEGRITY_FAILED,
                )
            seen_paths.add(entry.path)
        object.__setattr__(self, "artifacts", norm_artifacts)

        if not isinstance(self.metadata, Mapping):
            raise DeployTargetError(
                "metadata must be a mapping",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
        object.__setattr__(
            self,
            "metadata",
            _normalize_json_value(
                self.metadata,
                field_name="metadata",
                freeze=True,
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            ),
        )

        expected_hash = compute_deploy_manifest_hash(self._payload_without_hash())
        if not self.manifest_hash:
            object.__setattr__(self, "manifest_hash", expected_hash)
        else:
            supplied = _validate_sha256_hex(
                self.manifest_hash,
                "manifest_hash",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
            if supplied != expected_hash:
                raise DeployTargetError(
                    f"manifest_hash mismatch: expected {expected_hash}, got {supplied}",
                    code=TARGET_ARTIFACT_INTEGRITY_FAILED,
                )
            object.__setattr__(self, "manifest_hash", supplied)

    def _payload_without_hash(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_id": self.target_id,
            "strategy_id": self.strategy_id,
            "bundle_hash": self.bundle_hash,
            "slot_id": self.slot_id,
            "frequency": self.frequency,
            "created_at_utc": self.created_at_utc,
            "artifacts": [entry.to_dict() for entry in self.artifacts],
            "metadata": _unfreeze_json_value(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_without_hash()
        payload["manifest_hash"] = self.manifest_hash
        return payload

    def to_json(self) -> str:
        return (
            json.dumps(
                self.to_dict(),
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> DeployArtifactManifest:
        if not isinstance(raw, Mapping):
            raise DeployTargetError(
                "DeployArtifactManifest must be a JSON object",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
        expected = {
            "schema_version",
            "target_id",
            "strategy_id",
            "bundle_hash",
            "slot_id",
            "frequency",
            "created_at_utc",
            "artifacts",
            "metadata",
            "manifest_hash",
        }
        missing = expected - set(raw)
        if missing:
            raise DeployTargetError(
                f"DeployArtifactManifest missing required fields: {sorted(missing)}",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
        extra = set(raw) - expected
        if extra:
            raise DeployTargetError(
                f"DeployArtifactManifest contains unknown fields: {sorted(extra)}",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
        if raw["schema_version"] != DEPLOY_ARTIFACT_MANIFEST_SCHEMA_VERSION:
            raise DeployTargetError(
                f"Unsupported DeployArtifactManifest schema_version: {raw['schema_version']!r}",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
        _validate_sha256_hex(
            raw["manifest_hash"],
            "manifest_hash",
            code=TARGET_ARTIFACT_INTEGRITY_FAILED,
        )
        artifacts_raw = raw["artifacts"]
        if isinstance(artifacts_raw, (str, bytes)) or not isinstance(
            artifacts_raw, Sequence
        ):
            raise DeployTargetError(
                "DeployArtifactManifest artifacts must be a sequence",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
        return cls(
            schema_version=raw["schema_version"],
            target_id=raw["target_id"],
            strategy_id=raw["strategy_id"],
            bundle_hash=raw["bundle_hash"],
            slot_id=raw["slot_id"],
            frequency=raw["frequency"],
            created_at_utc=raw["created_at_utc"],
            artifacts=tuple(
                DeployArtifactEntry.from_mapping(item) for item in artifacts_raw
            ),
            metadata=raw["metadata"],
            manifest_hash=raw["manifest_hash"],
        )

    @classmethod
    def from_json(cls, content: str) -> DeployArtifactManifest:
        if not isinstance(content, str) or not content.strip():
            raise DeployTargetError(
                "DeployArtifactManifest JSON must be a nonempty string",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
        try:
            payload = json.loads(content, parse_constant=_reject_json_constant)
        except json.JSONDecodeError as exc:
            raise DeployTargetError(
                f"Failed to parse DeployArtifactManifest JSON: {exc}",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            ) from exc
        return cls.from_mapping(payload)


def write_deploy_manifest(
    manifest: DeployArtifactManifest,
    output_dir: str | Path,
) -> Path:
    """Verify artifact files on disk and atomically write deploy_manifest.json."""
    if not isinstance(manifest, DeployArtifactManifest):
        raise DeployTargetError(
            "manifest must be a DeployArtifactManifest instance",
            code=TARGET_ARTIFACT_INTEGRITY_FAILED,
        )
    root = resolve_path_checked(
        Path(output_dir).expanduser(),
        label="output_dir",
        code=TARGET_ARTIFACT_INTEGRITY_FAILED,
    )
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / DEPLOY_MANIFEST_FILENAME
    if manifest_path.is_symlink():
        resolved_manifest = resolve_path_checked(
            manifest_path,
            label=DEPLOY_MANIFEST_FILENAME,
            code=TARGET_ARTIFACT_INTEGRITY_FAILED,
        )
        if not resolved_manifest.is_relative_to(root):
            raise DeployTargetError(
                "deploy_manifest.json symlink escapes output directory",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )

    for entry in manifest.artifacts:
        resolved = _resolve_contained_file(root, entry.path, f"artifacts[{entry.path!r}]")
        actual_sha = sha256_file(resolved)
        if actual_sha != entry.sha256:
            raise DeployTargetError(
                f"artifacts[{entry.path!r}] sha256 mismatch: expected {entry.sha256}, got {actual_sha}",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
        actual_size = resolved.stat().st_size
        if actual_size != entry.size_bytes:
            raise DeployTargetError(
                f"artifacts[{entry.path!r}] size mismatch: expected {entry.size_bytes}, got {actual_size}",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{DEPLOY_MANIFEST_FILENAME}.",
        suffix=".tmp",
        dir=root,
    )
    temporary = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(manifest.to_json())
        temporary.replace(manifest_path)
    finally:
        temporary.unlink(missing_ok=True)

    verify_deploy_manifest(manifest_path)
    return manifest_path


def verify_deploy_manifest(
    artifact: DeployArtifactManifest | str | Path,
    *,
    output_dir: str | Path | None = None,
) -> DeployArtifactManifest:
    """Fail-closed verification of DeployArtifactManifest schema, hash, containment, and SHA-256s."""
    if isinstance(artifact, (str, Path)):
        candidate = Path(artifact).expanduser()
        if not candidate.exists():
            raise DeployTargetError(
                f"Deploy artifact path does not exist: {candidate}",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
        if candidate.is_dir():
            root = resolve_path_checked(
                candidate,
                label="artifact output directory",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
            manifest_path = root / DEPLOY_MANIFEST_FILENAME
            if manifest_path.is_symlink():
                resolved_manifest = resolve_path_checked(
                    manifest_path,
                    label=DEPLOY_MANIFEST_FILENAME,
                    code=TARGET_ARTIFACT_INTEGRITY_FAILED,
                )
                if not resolved_manifest.is_relative_to(root):
                    raise DeployTargetError(
                        "deploy_manifest.json symlink escapes output directory",
                        code=TARGET_ARTIFACT_INTEGRITY_FAILED,
                    )
        else:
            manifest_path = candidate
            root = resolve_path_checked(
                manifest_path.parent,
                label="artifact parent directory",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
            if manifest_path.is_symlink():
                resolved_manifest = resolve_path_checked(
                    manifest_path,
                    label=DEPLOY_MANIFEST_FILENAME,
                    code=TARGET_ARTIFACT_INTEGRITY_FAILED,
                )
                if not resolved_manifest.is_relative_to(root):
                    raise DeployTargetError(
                        "deploy_manifest.json symlink escapes output directory",
                        code=TARGET_ARTIFACT_INTEGRITY_FAILED,
                    )
        if output_dir is not None:
            expected_root = resolve_path_checked(
                Path(output_dir).expanduser(),
                label="output_dir",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
            if expected_root != root:
                raise DeployTargetError(
                    "output_dir does not match manifest parent directory",
                    code=TARGET_ARTIFACT_INTEGRITY_FAILED,
                )
        if not manifest_path.is_file():
            raise DeployTargetError(
                f"Deploy manifest missing: {manifest_path}",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
        try:
            content = manifest_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise DeployTargetError(
                f"Failed to read deploy manifest: {exc}",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            ) from exc
        manifest = DeployArtifactManifest.from_json(content)
    elif isinstance(artifact, DeployArtifactManifest):
        expected_hash = compute_deploy_manifest_hash(artifact._payload_without_hash())
        if artifact.manifest_hash != expected_hash:
            raise DeployTargetError(
                f"manifest_hash mismatch: expected {expected_hash}, got {artifact.manifest_hash}",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
        base_dir = output_dir if output_dir is not None else getattr(artifact, "_base_dir", None)
        if base_dir is None:
            raise DeployTargetError(
                "output_dir is required to verify DeployArtifactManifest artifacts on disk",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
        root = resolve_path_checked(
            Path(base_dir).expanduser(),
            label="output_dir",
            code=TARGET_ARTIFACT_INTEGRITY_FAILED,
        )
        if not root.is_dir():
            raise DeployTargetError(
                f"Deploy output directory does not exist: {root}",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
        manifest_path = root / DEPLOY_MANIFEST_FILENAME
        if manifest_path.is_symlink():
            resolved_manifest = resolve_path_checked(
                manifest_path,
                label=DEPLOY_MANIFEST_FILENAME,
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
            if not resolved_manifest.is_relative_to(root):
                raise DeployTargetError(
                    "deploy_manifest.json symlink escapes output directory",
                    code=TARGET_ARTIFACT_INTEGRITY_FAILED,
                )
        if not manifest_path.is_file():
            raise DeployTargetError(
                f"Deploy manifest missing on disk: {manifest_path}",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
        try:
            disk_manifest = DeployArtifactManifest.from_json(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError) as exc:
            raise DeployTargetError(
                f"Failed to read deploy manifest on disk: {exc}",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            ) from exc
        if disk_manifest.manifest_hash != artifact.manifest_hash:
            raise DeployTargetError(
                "On-disk deploy_manifest.json does not match supplied DeployArtifactManifest",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
        manifest = artifact
    else:
        raise DeployTargetError(
            "artifact must be a DeployArtifactManifest or filesystem path",
            code=TARGET_ARTIFACT_INTEGRITY_FAILED,
        )

    for entry in manifest.artifacts:
        resolved = _resolve_contained_file(root, entry.path, f"artifacts[{entry.path!r}]")
        actual_sha = sha256_file(resolved)
        if actual_sha != entry.sha256:
            raise DeployTargetError(
                f"artifacts[{entry.path!r}] sha256 mismatch: expected {entry.sha256}, got {actual_sha}",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
        actual_size = resolved.stat().st_size
        if actual_size != entry.size_bytes:
            raise DeployTargetError(
                f"artifacts[{entry.path!r}] size mismatch: expected {entry.size_bytes}, got {actual_size}",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )

    object.__setattr__(manifest, "_base_dir", root)
    return manifest


@dataclass(frozen=True)
class AlignmentCheckItem:
    """Single immutable alignment check outcome."""

    name: str
    passed: bool
    reason: str = ""
    code: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_nonempty_str(self.name, "check.name"))
        if type(self.passed) is not bool:
            raise DeployTargetError("check.passed must be a boolean")
        if not isinstance(self.reason, str):
            raise DeployTargetError("check.reason must be a string")
        if not isinstance(self.code, str):
            raise DeployTargetError("check.code must be a string")
        if self.code and self.code != self.code.strip():
            raise DeployTargetError("check.code must not contain surrounding whitespace")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "reason": self.reason,
            "code": self.code,
        }


@dataclass(frozen=True)
class AlignmentReport:
    """Immutable alignment verification report returned by DeployTarget.verify_alignment."""

    target_id: str
    passed: bool
    bundle_hash: str
    checks: tuple[AlignmentCheckItem, ...] = ()
    warnings: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    max_abs_diff: float | None = None
    error_code: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "target_id", _require_nonempty_str(self.target_id, "target_id")
        )
        if type(self.passed) is not bool:
            raise DeployTargetError("passed must be a boolean")
        if self.bundle_hash == "" and not self.passed:
            object.__setattr__(self, "bundle_hash", "")
        else:
            object.__setattr__(
                self,
                "bundle_hash",
                _validate_sha256_hex(self.bundle_hash, "bundle_hash", code=TARGET_BUNDLE_INVALID),
            )
        if isinstance(self.checks, (str, bytes)) or not isinstance(self.checks, Sequence):
            raise DeployTargetError("checks must be a sequence of AlignmentCheckItem")
        norm_checks = tuple(
            item
            if isinstance(item, AlignmentCheckItem)
            else AlignmentCheckItem(**item)
            for item in self.checks
        )
        object.__setattr__(self, "checks", norm_checks)

        if isinstance(self.warnings, (str, bytes)) or not isinstance(
            self.warnings, Sequence
        ):
            raise DeployTargetError("warnings must be a sequence of strings")
        norm_warnings = tuple(
            _require_nonempty_str(item, f"warnings[{idx}]")
            for idx, item in enumerate(self.warnings)
        )
        object.__setattr__(self, "warnings", norm_warnings)

        if not isinstance(self.metrics, Mapping):
            raise DeployTargetError("metrics must be a mapping")
        object.__setattr__(
            self,
            "metrics",
            _normalize_json_value(
                self.metrics, field_name="metrics", freeze=True, code=TARGET_BUNDLE_INVALID
            ),
        )

        if self.max_abs_diff is not None:
            val = self.max_abs_diff
            if type(val) is bool or not isinstance(val, (int, float)):
                raise DeployTargetError(
                    "max_abs_diff must be None or a finite non-negative float"
                )
            try:
                num = float(val)
            except OverflowError as exc:
                raise DeployTargetError(
                    "max_abs_diff must be None or a finite non-negative float"
                ) from exc
            if not math.isfinite(num) or num < 0.0:
                raise DeployTargetError(
                    "max_abs_diff must be None or a finite non-negative float"
                )
            object.__setattr__(self, "max_abs_diff", 0.0 if num == 0.0 else num)

        if not isinstance(self.error_code, str):
            raise DeployTargetError("error_code must be a string")
        if self.error_code and self.error_code != self.error_code.strip():
            raise DeployTargetError("error_code must not have surrounding whitespace")
        if self.passed:
            if self.error_code:
                raise DeployTargetError("passed=True cannot carry a non-empty error_code")
            if any(not item.passed for item in norm_checks):
                raise DeployTargetError("passed=True cannot contain failing check items")

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "passed": self.passed,
            "bundle_hash": self.bundle_hash,
            "checks": [item.to_dict() for item in self.checks],
            "warnings": list(self.warnings),
            "metrics": _unfreeze_json_value(self.metrics),
            "max_abs_diff": self.max_abs_diff,
            "error_code": self.error_code,
        }


def _verify_bundle_directory(bundle_dir: Path) -> StrategyBundle:
    try:
        return verify_strategy_bundle(bundle_dir)
    # Path.resolve() can raise RuntimeError for symlink loops in Python 3.12.
    except (StrategyBundleError, OSError, RuntimeError) as exc:
        raise DeployTargetError(
            f"StrategyBundle verification failed: {exc}",
            code=TARGET_BUNDLE_INVALID,
        ) from exc


@dataclass(frozen=True)
class StrategyBundleSource:
    """Explicit, stateless locator pairing a StrategyBundle directory with an optional StrategyBundle."""

    bundle_dir: Path
    bundle: StrategyBundle | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.bundle_dir, (str, Path)):
            raise DeployTargetError(
                "StrategyBundleSource.bundle_dir must be a filesystem path",
                code=TARGET_BUNDLE_INVALID,
            )
        object.__setattr__(
            self,
            "bundle_dir",
            resolve_path_checked(
                Path(self.bundle_dir).expanduser(),
                label="StrategyBundleSource.bundle_dir",
                code=TARGET_BUNDLE_INVALID,
            ),
        )
        if self.bundle is not None and not isinstance(self.bundle, StrategyBundle):
            raise DeployTargetError(
                "StrategyBundleSource.bundle must be a StrategyBundle or None",
                code=TARGET_BUNDLE_INVALID,
            )

    @classmethod
    def from_dir(cls, bundle_dir: str | Path) -> StrategyBundleSource:
        """Verify bundle_dir via S5 gate and construct an explicit StrategyBundleSource."""
        resolved = resolve_path_checked(
            Path(bundle_dir).expanduser(),
            label="bundle_dir",
            code=TARGET_BUNDLE_INVALID,
        )
        return cls(bundle_dir=resolved, bundle=_verify_bundle_directory(resolved))


def resolve_verified_bundle_with_dir(
    bundle: StrategyBundle | StrategyBundleSource | str | Path | None = None,
    *,
    bundle_dir: str | Path | None = None,
    require_verified_source: bool = True,
) -> tuple[StrategyBundle, Path | None]:
    """Load and verify a StrategyBundle fail-closed, returning (bundle, verified_bundle_dir).

    When ``require_verified_source=True`` (default for export/alignment), a ``bundle_dir``
    or an explicit ``StrategyBundleSource`` must supply the directory, and reading always
    re-runs the S5 ``verify_strategy_bundle()`` gate without implicit global state.
    """
    effective_dir = bundle_dir
    effective_bundle: StrategyBundle | None = None
    if isinstance(bundle, StrategyBundleSource):
        if effective_dir is not None:
            resolved_effective_dir = resolve_path_checked(
                Path(effective_dir).expanduser(),
                label="bundle_dir",
                code=TARGET_BUNDLE_INVALID,
            )
            if resolved_effective_dir != bundle.bundle_dir:
                raise DeployTargetError(
                    "Conflicting StrategyBundleSource.bundle_dir and bundle_dir provided",
                    code=TARGET_BUNDLE_INVALID,
                )
        effective_dir = bundle.bundle_dir
        effective_bundle = bundle.bundle
    elif isinstance(bundle, (str, Path)):
        if effective_dir is not None:
            resolved_effective_dir = resolve_path_checked(
                Path(effective_dir).expanduser(),
                label="bundle_dir",
                code=TARGET_BUNDLE_INVALID,
            )
            resolved_bundle_path = resolve_path_checked(
                Path(bundle).expanduser(),
                label="bundle path",
                code=TARGET_BUNDLE_INVALID,
            )
            if resolved_effective_dir != resolved_bundle_path:
                raise DeployTargetError(
                    "Conflicting bundle path and bundle_dir provided",
                    code=TARGET_BUNDLE_INVALID,
                )
        effective_dir = bundle
    elif bundle is not None:
        if not isinstance(bundle, StrategyBundle):
            raise DeployTargetError(
                "bundle must be a StrategyBundle, StrategyBundleSource, or bundle_dir path",
                code=TARGET_BUNDLE_INVALID,
            )
        effective_bundle = bundle

    if effective_dir is not None:
        resolved_dir = resolve_path_checked(
            Path(effective_dir).expanduser(),
            label="bundle_dir",
            code=TARGET_BUNDLE_INVALID,
        )
        verified = _verify_bundle_directory(resolved_dir)
        if effective_bundle is not None:
            if effective_bundle.bundle_hash != verified.bundle_hash or effective_bundle.to_dict() != verified.to_dict():
                raise DeployTargetError(
                    "Supplied StrategyBundle does not match verified bundle_dir manifest",
                    code=TARGET_BUNDLE_INVALID,
                )
        return verified, resolved_dir

    if effective_bundle is None:
        raise DeployTargetError(
            "Either bundle or bundle_dir must be provided",
            code=TARGET_BUNDLE_INVALID,
        )

    if require_verified_source:
        raise DeployTargetError(
            "Unverified in-memory StrategyBundle cannot bypass verify_strategy_bundle gate; "
            "supply bundle_dir or an explicit StrategyBundleSource",
            code=TARGET_BUNDLE_INVALID,
        )

    try:
        recomputed = compute_bundle_hash(effective_bundle._manifest_payload_without_hash())
    except Exception as exc:
        raise DeployTargetError(
            f"Invalid StrategyBundle payload: {exc}",
            code=TARGET_BUNDLE_INVALID,
        ) from exc
    if recomputed != effective_bundle.bundle_hash:
        raise DeployTargetError(
            "StrategyBundle bundle_hash verification failed",
            code=TARGET_BUNDLE_INVALID,
        )
    return effective_bundle, None


def resolve_verified_bundle(
    bundle: StrategyBundle | StrategyBundleSource | str | Path | None = None,
    *,
    bundle_dir: str | Path | None = None,
    require_verified_source: bool = True,
) -> StrategyBundle:
    """Load and verify a StrategyBundle fail-closed."""
    verified, _ = resolve_verified_bundle_with_dir(
        bundle,
        bundle_dir=bundle_dir,
        require_verified_source=require_verified_source,
    )
    return verified


def utc_now_iso() -> str:
    """Return current UTC timestamp formatted in canonical ISO-8601 Z notation."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@runtime_checkable
class DeployTarget(Protocol):
    """Protocol implemented by all Stage 6 deployment output targets."""

    @property
    def target_id(self) -> str: ...

    def check_compatibility(
        self, bundle: StrategyBundle | StrategyBundleSource | str | Path, **kwargs: Any
    ) -> CompatibilityReport: ...

    def export(
        self,
        bundle: StrategyBundle | StrategyBundleSource | str | Path,
        output_dir: str | Path,
        **kwargs: Any,
    ) -> DeployArtifactManifest: ...

    def verify_alignment(
        self,
        bundle: StrategyBundle | StrategyBundleSource | str | Path,
        artifact: DeployArtifactManifest | str | Path,
        **kwargs: Any,
    ) -> AlignmentReport: ...


def validate_utc_iso_timestamp(value: Any, field_name: str) -> str:
    return _validate_utc_iso_timestamp(value, field_name)


def resolve_contained_file(root_dir: Path, rel_path: str, field_name: str) -> Path:
    return _resolve_contained_file(root_dir, rel_path, field_name)


__all__ = [
    "AlignmentCheckItem",
    "AlignmentReport",
    "CompatibilityIssue",
    "CompatibilityReport",
    "DEPLOY_ARTIFACT_MANIFEST_SCHEMA_VERSION",
    "DEPLOY_MANIFEST_FILENAME",
    "DeployArtifactEntry",
    "DeployArtifactManifest",
    "DeployTarget",
    "DeployTargetError",
    "StrategyBundleSource",
    "TARGET_ALIGNMENT_MISMATCH",
    "TARGET_ARTIFACT_INTEGRITY_FAILED",
    "TARGET_AUX_FEATURE_UNSUPPORTED",
    "TARGET_BUNDLE_INVALID",
    "TARGET_EXECUTION_CONTRACT_UNSUPPORTED",
    "TARGET_SIGNAL_SOURCE_MISSING",
    "TARGET_SLOT_INCOMPATIBLE",
    "compute_deploy_manifest_hash",
    "resolve_contained_file",
    "resolve_path_checked",
    "resolve_verified_bundle",
    "resolve_verified_bundle_with_dir",
    "utc_now_iso",
    "validate_utc_iso_timestamp",
    "verify_deploy_manifest",
    "write_deploy_manifest",
]
