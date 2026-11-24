"""Explicit registry of Stage 6 deployment output targets."""
from __future__ import annotations

from .contracts import DeployTarget


def _validate_target_instance(target: object) -> DeployTarget:
    if not isinstance(target, DeployTarget):
        raise TypeError("register expects a DeployTarget")
    target_id = getattr(target, "target_id", None)
    if not isinstance(target_id, str) or not target_id.strip() or target_id != target_id.strip():
        raise TypeError("DeployTarget.target_id must be a nonempty string without surrounding whitespace")
    for method in ("check_compatibility", "export", "verify_alignment"):
        if not callable(getattr(target, method, None)):
            raise TypeError(f"DeployTarget must define callable {method}")
    return target


class TargetRegistry:
    """Explicit, side-effect-free registry mapping target_id to DeployTarget instances."""

    def __init__(self) -> None:
        self._targets: dict[str, DeployTarget] = {}

    def register(self, target: DeployTarget) -> None:
        validated = _validate_target_instance(target)
        if validated.target_id in self._targets:
            raise ValueError(f"DeployTarget already registered: {validated.target_id!r}")
        self._targets[validated.target_id] = validated

    def get(self, target_id: str) -> DeployTarget:
        try:
            return self._targets[target_id]
        except KeyError:
            raise KeyError(f"Unknown DeployTarget: {target_id!r}") from None

    def list_targets(self) -> tuple[str, ...]:
        return tuple(sorted(self._targets))

    def __contains__(self, target_id: object) -> bool:
        return isinstance(target_id, str) and target_id in self._targets

    def __len__(self) -> int:
        return len(self._targets)


def build_default_target_registry() -> TargetRegistry:
    """Build a fresh TargetRegistry populated with the default built-in targets."""
    from .jq_cloud import JqCloudTarget
    from .signal_export import SignalExportTarget

    registry = TargetRegistry()
    registry.register(SignalExportTarget())
    registry.register(JqCloudTarget())
    return registry


__all__ = ["TargetRegistry", "build_default_target_registry"]
