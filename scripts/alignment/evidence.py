"""Content-bound alignment evidence; no independent run registry."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_hashes(paths) -> dict[str, str]:
    return {str(p.resolve()): digest(p) for p in sorted(set(paths)) if p.is_file()}


def intact(records: dict[str, str]) -> bool:
    return bool(records) and all(
        Path(p).is_file() and digest(Path(p)) == h for p, h in records.items()
    )


def intact_inputs(manifest: dict) -> bool:
    """Reject changed or removed source/data inputs before publishing a verdict."""
    try:
        if not intact(manifest["sources"]):
            return False
        versions = dict(manifest.get("data_versions", {}))
        strategy = manifest.get("strategy_data_version")
        if strategy:
            versions[strategy["path"]] = strategy
        for raw_path, expected in versions.items():
            actual = Path(raw_path).stat()
            if (actual.st_size, actual.st_mtime_ns) != (
                expected["size"],
                expected["mtime_ns"],
            ):
                return False
        return True
    except (OSError, KeyError, TypeError):
        return False


def input_manifest(repo: Path, factor: Path, factor_class: str, extra_paths=()) -> dict:
    paths = {factor, *map(Path, extra_paths)}
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise ValueError("alignment_source_missing: " + ",".join(missing))
    paths.update(repo.glob("requirements*.txt"))
    paths.update(
        p
        for p in (repo / "pyproject.toml", repo / "uv.lock", repo / "environment.yml")
        if p.is_file()
    )
    # ponytail: conservative shared-source invalidation; use exact imports if hashing becomes material.
    for directory in (
        "utils",
        "scripts/build",
        "scripts/alignment",
        "jq_strategies",
        "factor_synthesis/core",
        "backtest_and_analysis",
        "factor_synthesis/vectorbt_adapter",
        "tools/agent_harness",
        "scripts/wind_micro",
    ):
        paths.update((repo / directory).glob("*.py"))
    paths.update(
        repo / p
        for p in (
            "scripts/build_jq_strategy.py",
            "config/factor_requirements.json",
            "config/cloud_api_mapping.json",
            "config/small_mkt_config.json",
        )
    )
    paths.update((repo / ".claude/skills/quantation-alignment/scripts").glob("*.py"))
    paths.update((repo / ".claude/skills/quantation-alignment/scripts").glob("*.sh"))
    pending = [path for path in paths if path.suffix == ".py" and path.is_file()]
    seen = set()
    while pending:
        current = pending.pop().resolve()
        if current in seen:
            continue
        seen.add(current)
        paths.add(current)
        for node in ast.walk(ast.parse(current.read_bytes())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = node.module or ""
                modules = [prefix] + [prefix + "." + a.name for a in node.names]
                if node.level:
                    root = current.parent
                    for _ in range(node.level - 1):
                        root = root.parent
                else:
                    root = repo
            else:
                continue
            if isinstance(node, ast.Import):
                root = repo
            for module in modules:
                candidate = root.joinpath(*module.strip(".").split(".")).with_suffix(
                    ".py"
                )
                if not candidate.is_file():
                    candidate = (
                        root.joinpath(*module.strip(".").split(".")) / "__init__.py"
                    )
                if candidate.is_file():
                    pending.append(candidate)
    config_path = repo / "config/small_mkt_config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    data = {}
    for section in ("database",):
        value = config.get(section, {}).get("duckdb_path")
        if value:
            path = Path(value)
            if not path.is_absolute():
                path = repo / path
            if path.is_file():
                stat = path.stat()
                data[str(path.resolve())] = {
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
    payload = {
        "schema_version": 2,
        "factor_class": factor_class,
        "sources": file_hashes(paths),
        "analysis_period": config.get("analysis_period", {}),
        "pool_config": config.get(
            "dynamic_pools",
            config.get("stock_pool", config.get("dynamic_stock_pool", {})),
        ),
        "data_versions": data,
    }
    payload["fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    return payload


def layer_results(stages: list[dict]) -> dict:
    layers = {
        name: {"status": "NOT_VERIFIED"}
        for name in ("inputs", "factor", "signal", "execution")
    }
    version_failure = next(
        (s for s in stages if s["name"] == "input_version" and not s["ok"]), None
    )
    if version_failure:
        layers["inputs"] = {"status": "FAIL", "reason": version_failure["reason"]}
    comparison = next((s for s in stages if s["name"] == "compare"), None)
    if comparison:
        layers["factor"] = {
            "status": "PASS" if comparison["ok"] else "FAIL",
            "reason": comparison.get("reason", ""),
            "metrics": comparison.get("metrics", {}),
        }
    return layers


def archive_previous(output: Path, fingerprint: str) -> None:
    """Move superseded working data once; never duplicate large retry directories."""
    from datetime import datetime, timezone

    manifest = output / "alignment_manifest.json"
    if not manifest.is_file():
        return
    try:
        old = json.loads(manifest.read_text()).get("fingerprint")
    except (ValueError, OSError):
        old = None
    if old == fingerprint:
        return
    destination = (
        output / "history" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    )
    destination.mkdir(parents=True)
    retained = {}
    # Move the manifest last: an interrupted archival still exposes its old
    # revision, and a retry cannot mistake mixed files for the new revision.
    for source in sorted(output.iterdir(), key=lambda path: path.name == "alignment_manifest.json"):
        if source.name == "history" or source.name.startswith("."):
            continue
        target = destination / source.name
        source.replace(target)
        retained[str(source)] = str(target)
    (destination / "retained_paths.json").write_text(
        json.dumps(retained, indent=2) + "\n"
    )
