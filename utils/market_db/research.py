"""Resolve the explicit research market selection against a local read-only database."""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import replace
import os
from pathlib import Path

import duckdb

from utils.slots import MarketSlot, SlotConfig, load_slot_config

from .bindings import open_market_db, resolve_market_db_path
from .identity import bind_market_db_fingerprint, market_db_identity
from .profile import is_market_db_id
from .storage import market_db_sidecar_path, read_market_db_profile

_NATIVE_DOMAINS = {"jq_ashare_local": "cn_equity", "cn_futures": "cn_futures", "global_futures": "global_futures"}


def resolve_research_slot(
    declaration: object, slot_id: str | None, base_dir: Path,
) -> tuple[SlotConfig, Path]:
    if isinstance(declaration, str) and declaration != "jq_ashare_local":
        path = (base_dir / Path(declaration).expanduser()).resolve()
        config, base_dir = load_slot_config(path), path.parent
    elif isinstance(declaration, Mapping):
        config = SlotConfig.from_mapping(declaration)
    elif declaration in (None, "jq_ashare_local"):
        from slots.jq_ashare_local import daily_slot_config

        config = daily_slot_config()
    else:
        raise ValueError("Supply a SlotConfig YAML/JSON or a SlotConfig object")
    if slot_id and slot_id != config.slot_id:
        raise ValueError("slot_id disagrees with the SlotConfig")
    return config, base_dir


def resolve_research_market_db(db_id: str) -> Path:
    if not is_market_db_id(db_id):
        raise ValueError("--market-db requires a db_id from QUANT_DATA_BINDINGS.market_dbs")
    return resolve_market_db_path(db_id)


def same_database(left: Path, right: Path) -> bool:
    return left == right or (left.exists() and right.exists() and left.samefile(right))


@contextmanager
def open_research_slot(
    db_id: str, config: SlotConfig, base_dir: Path, *, declared_db: str | None = None,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> Iterator[tuple[MarketSlot, Path]]:
    from slots import build_slot_from_config

    path = resolve_research_market_db(db_id)
    if declared_db is not None and not same_database(path, (base_dir / Path(declared_db).expanduser()).resolve()):
        raise ValueError(f"Declared database conflicts with --market-db {db_id!r}")

    with ExitStack() as stack:
        if connection is None:
            connection, profile, _ = stack.enter_context(open_market_db(path))
        else:
            catalog_path = connection.execute(
                "SELECT path FROM duckdb_databases() WHERE database_name = current_database()"
            ).fetchone()[0]
            if catalog_path is None or not same_database(path, Path(catalog_path).resolve()):
                raise ValueError(f"Connection conflicts with --market-db {db_id!r}")
            profile, _ = read_market_db_profile(connection)
        if profile.db_id != db_id:
            raise ValueError(f"Binding {db_id!r} points to database {profile.db_id!r}")
        provider = config.bar_feed.provider
        native_domain = _NATIVE_DOMAINS.get(provider)
        if provider not in profile.allowed_slot_providers or (native_domain and native_domain != profile.market_domain):
            raise ValueError(f"Slot provider {provider!r} conflicts with --market-db {db_id!r}")
        domain = getattr(config, "market_domain", None)
        if domain is not None and domain != profile.market_domain:
            raise ValueError(f"Slot market_domain {domain!r} conflicts with --market-db {db_id!r}")

        runtime = config
        if not native_domain:
            if provider != "duckdb_table":
                raise ValueError("--market-db requires DuckDB as the Slot bar source")
            for name in ("bar_feed", "universe", "aux_features", "microstructure"):
                component = getattr(config, name)
                if component.provider != "duckdb_table":
                    continue
                raw_path = component.options.get("path")
                if raw_path is not None and (
                    not isinstance(raw_path, str) or not raw_path.strip()
                    or not same_database(path, (base_dir / Path(raw_path).expanduser()).resolve())
                ):
                    raise ValueError(f"Slot {name} path conflicts with --market-db {db_id!r}")
                options = {**component.options, "path": str(path)}
                runtime = replace(runtime, **{name: replace(component, options=options)})
        slot = build_slot_from_config(runtime, connection=connection if native_domain else None, base_dir=base_dir)
        if not native_domain:
            if slot.calendar.calendar_id != profile.calendar_id or slot.calendar.timezone != profile.timezone:
                raise ValueError(f"Slot calendar or timezone conflicts with --market-db {db_id!r}")
            stack.close()
        identity = market_db_identity(profile)
        yield replace(slot, slot_config_hash=config.slot_config_hash, slot_data_fingerprint=bind_market_db_fingerprint(
            slot.slot_data_fingerprint, identity
        )), path


def synthesis_market_db_paths(config: Mapping) -> tuple[str | None, str | None]:
    from utils.data_store import binding_contract

    if binding_contract(config.get("data_bindings")):
        raise ValueError("--market-db requires Scheme B inputs and conflicts with frozen DuckDB factor bindings")
    database = config.get("database", {})
    price = database.get("price_db", {}).get("duckdb_path")
    return price or config.get("slot_db"), database.get("local_db", {}).get("duckdb_path")


def check_synthesis_market_outputs(
    source: Path,
    config: dict,
    execution_dir: Path,
    return_series_dir: str | Path,
    *,
    files: Sequence[str | Path] = (),
    directories: Sequence[Path] = (),
    cwd: Path | None = None,
) -> None:
    """Check synthesis output files and namespaces before any producer can write."""
    from utils.output_paths import bind_output_defaults

    cwd = cwd or Path.cwd()
    protected = (source, market_db_sidecar_path(source))
    protected_paths = {*protected, *(item.resolve() for item in protected)}

    def check(raw_path: str | Path, *, tree: bool = False) -> Path:
        path = cwd / Path(raw_path).expanduser()
        target = path.resolve()
        for item in protected_paths:
            if (same_database(item, target) or item in target.parents
                    or (tree and target in item.parents)):
                raise ValueError(f"Output {path} overlaps the selected market database or its profile ({item})")
        return target

    for path in (*files, *(execution_dir / name for name in (
        "output_bindings.json", "resolved_output_paths.json", "benchmark_reports.json",
        "algorithm_model_evidence.json", "strategy_bundle_manifest.json",
    ))):
        if path:
            check(path)

    _, result_db = synthesis_market_db_paths(config)
    if result_db:
        check(result_db)
    resolved = bind_output_defaults(config, directory=execution_dir)
    settings, cache = resolved["output_settings"], resolved["cache_settings"]
    trees = [
        *directories, return_series_dir, cache.get("cache_dir"), cache.get("parquet_dir"),
        *(settings.get(key) for key in ("weights_output_dir", "model_output_dir", "diagnostics_output_dir")),
        *(execution_dir / name for name in ("strategy_bundle", "strategy_bundles", "diagnostics", "mlruns")),
        *execution_dir.glob(".strategy_bundle_staging_*"),
    ]
    if settings.get("save_alignment_data"):
        trees.append(settings.get("alignment_output_dir", "outputs/synthesis/alignment"))
    benchmark = config.get("performance_evaluation", {}).get("benchmark_suite", {})
    if benchmark.get("enabled") and benchmark.get("cache_fixed_benchmarks", True):
        trees.append(benchmark.get("fixed_cache_dir", "outputs/synthesis/benchmark_cache"))
    if config.get("use_precomputed_returns"):
        trees.append("./cache")
    for method in config.get("synthesis_methods", {}).values():
        prepared = method.get("size_conditioning", {}).get("prepared_cache", {})
        if prepared.get("enabled", True):
            trees.append(prepared.get("cache_root"))

    # Existing symlinks and hard links can escape an otherwise disjoint output directory.
    linked_input = any(item.exists() and item.stat().st_nlink > 1 for item in protected_paths)
    visited: set[Path] = set()
    for directory in filter(None, trees):
        root = check(directory, tree=True)
        for parent, subdirs, names in os.walk(root, followlinks=True):
            target = check(parent, tree=True)
            if target in visited:
                subdirs.clear()
                continue
            visited.add(target)
            for name in names:
                child = Path(parent) / name
                if linked_input or child.is_symlink():
                    check(child)
