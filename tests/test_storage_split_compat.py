"""Compatibility checks for the split Bar storage packages."""
import ast
import json
import subprocess
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from graphlib import TopologicalSorter
from importlib import import_module
from importlib.util import resolve_name
from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq
import pytest

from test_bar_evaluation_store import _make_ts_bars, _run_cli
from test_market_factor_values import _make_sample_batch
from utils.duckdb_results_writer import DuckDBWriteSession


ROOT = Path(__file__).resolve().parents[1]
# The old modules also exposed imported names through wildcard imports.
LEGACY_EXPORTS = {
    "market_factor": """
        annotations dt fcntl hashlib inspect json os pickle re shutil uuid Mapping Sequence
        contextmanager asdict dataclass Path Any Iterator duckdb np pd pa pq is_bool_dtype
        is_complex_dtype is_numeric_dtype DuplicateKeyError BarCatalog table_exists
        resolve_code_version SessionCalendar BarSchedule Frequency
        MARKET_FACTOR_IDENTITY_SCHEMA_VERSION NAN_CANONICAL_BITS DEFAULT_ROW_GROUP_SIZE
        VALID_VIEWS VALID_ANALYSIS_MODES VALID_BUCKET_STRATEGIES VALID_ADJUSTMENTS
        TABLE_COLUMNS VIEW_COLUMNS LEAF_COLUMNS IDENTITY_COLUMNS LEAF_ARROW_SCHEMA
        FactorValueConflictError FactorWriteReceipt FactorReadPlan sql_canonical_float
        canonical_float_bits validate_factor_id validate_sha256_hex compute_factor_identity_hash
        build_identity_record MarketFactorValueStore save_bar_factor_values
    """,
    "bar_evaluation": """
        annotations dt json math re struct Mapping Sequence contextmanager dataclass Path
        Any Iterator duckdb np pd ResultIdentity sha256_file CS_COLUMNS CS_IC_COLUMNS
        CS_QUANTILE_COLUMNS RESULT_SCHEMA_VERSION TS_IC_COLUMNS TS_QUANTILE_COLUMNS
        acquire_exclusive_write_conn Frequency BAR_EVALUATION_TABLES
        SUPPORTED_RESULT_SCHEMA_VERSIONS VALID_ANALYSIS_MODES BarEvaluationPreconditionError
        BarEvaluationIntegrityError BarEvaluationConflictError BarEvaluationIngestReceipt
        ingest_evaluation migrate_bar_evaluation_db query_bar_evaluation_runs query_bar_evaluation_detail
        query_bar_evaluations get_bar_evaluation_filter_options summarize_bar_evaluation_connection
        summarize_bar_evaluation_db
    """,
}


@pytest.mark.parametrize("package", LEGACY_EXPORTS)
def test_storage_public_exports_match_original_module(package: str) -> None:
    legacy = import_module(f"utils.{package}_store")
    current = import_module(f"utils.{package}")
    expected = set(LEGACY_EXPORTS[package].split())

    assert {name for name in vars(legacy) if not name.startswith("_")} == expected
    assert set(legacy.__all__) == set(current.__all__) == expected
    for name in expected:
        assert getattr(legacy, name) is getattr(current, name), name


def test_market_factor_storage_uses_no_sqlite() -> None:
    package = ROOT / "utils" / "market_factor"
    files = sorted(package.glob("*.py"))
    assert files
    for path in [*files, package.with_name("market_factor_store.py")]:
        assert "sqlite3" not in path.read_text(encoding="utf-8"), path.name


@pytest.mark.parametrize("package", LEGACY_EXPORTS)
def test_storage_module_sizes_and_import_graph(package: str) -> None:
    package_name = f"utils.{package}"
    legacy_name = f"{package_name}_store"
    directory = ROOT / "utils" / package
    modules = {
        (package_name if path.stem == "__init__" else f"{package_name}.{path.stem}"): path
        for path in sorted(directory.glob("*.py"))
    }
    assert package_name in modules
    modules[legacy_name] = directory.with_name(f"{package}_store.py")
    graph: TopologicalSorter[str] = TopologicalSorter()
    for name, path in modules.items():
        source = path.read_text(encoding="utf-8")
        assert len(source.splitlines()) <= 500, path.name
        parent = name if path.stem == "__init__" else name.rpartition(".")[0]
        graph.add(name)
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = resolve_name("." * node.level + (node.module or ""), parent)
                imports = [f"{base}.{alias.name}" for alias in node.names]
                imports = [target if target in modules else base for target in imports]
            else:
                continue
            graph.add(name, *(target for target in imports if target in modules))
    tuple(graph.static_order())


@pytest.mark.parametrize("first", [
    ("utils.bar_evaluation_store", "utils.market_factor_store"),
    ("utils.bar_evaluation.ingest", "utils.market_factor.writer"),
    ("tools.agent_harness.cli",),
], ids=["legacy-first", "implementation-first", "integration-first"])
def test_storage_integrations_import_in_fresh_process(first: tuple[str, str]) -> None:
    modules = [
        *first,
        "utils.bar_evaluation_store",
        "utils.market_factor_store",
        "tools.agent_harness.cli",
    ]
    command = (
        "from importlib import import_module\n"
        "import sys\n"
        "for module in sys.argv[1:]:\n"
        "    import_module(module)\n"
    )
    subprocess.run([sys.executable, "-c", command, *modules], cwd=ROOT, check=True, timeout=30)


@pytest.mark.parametrize("ledger", ["identities_path", "batches_path"])
def test_market_factor_uses_legacy_ledger_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ledger: str,
) -> None:
    from utils import market_factor_store as legacy

    store = legacy.MarketFactorValueStore(tmp_path)
    frame, identity, calendar = _make_sample_batch()
    original_append = legacy._append_jsonl_line
    failed_path = getattr(store, ledger)
    failures: list[Path] = []

    def fail_append(path: Path, record: Mapping[str, Any]) -> None:
        if path == failed_path:
            failures.append(path)
            raise OSError("injected ledger failure")
        original_append(path, record)

    with monkeypatch.context() as patch:
        patch.setattr(legacy, "_append_jsonl_line", fail_append)
        with pytest.raises(OSError, match="injected ledger failure"):
            store.append_factor_values(frame.head(2), identity_record=identity, calendar=calendar)

    assert failures == [failed_path]
    assert store.read_factor_values().empty
    assert not list(store.parquet_root.rglob("*.parquet"))
    assert not list(store.root.glob(".staging-*"))
    assert store.append_factor_values(frame.head(2), identity_record=identity, calendar=calendar).inserted == 2
    assert store.append_factor_values(frame.head(2), identity_record=identity, calendar=calendar).skipped == 2


def test_market_factor_uses_legacy_row_group_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from utils import market_factor_store as legacy

    store = legacy.MarketFactorValueStore(tmp_path)
    frame, identity, calendar = _make_sample_batch()
    monkeypatch.setattr(legacy, "DEFAULT_ROW_GROUP_SIZE", 1)
    receipt = store.append_factor_values(frame.head(2), identity_record=identity, calendar=calendar)
    assert receipt.inserted == 2
    assert len(receipt.shards_written) == 1
    assert pq.read_metadata(store.parquet_root / receipt.shards_written[0]).num_row_groups == 2


@pytest.fixture(scope="module")
def published_artifacts(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, Any]]:
    folder = tmp_path_factory.mktemp("storage-compat-artifacts")
    config = folder / "config.json"
    config.write_text(json.dumps({
        "bars_csv": str(_make_ts_bars(folder)),
        "analysis_mode": "time_series",
        "calendar": "NQ",
        "start_date": "2026-12-01",
        "end_date": "2026-12-01",
        "horizons": [1, 2, 5],
        "n_quantiles": 3,
    }), encoding="utf-8")
    code, result = _run_cli(folder, "--config", str(config))
    assert code == 0
    return result["artifacts"]


@pytest.mark.parametrize("stage", ["initial", "locked"])
def test_bar_evaluation_uses_legacy_digest_at_both_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, published_artifacts: list[dict[str, Any]], stage: str,
) -> None:
    from utils import bar_evaluation_store as legacy

    db_path = tmp_path / "evaluation.duckdb"
    original_lock = legacy.acquire_exclusive_write_conn
    original_digest = legacy.sha256_file
    rejected: list[Path] = []

    def incorrect_digest(path: Path) -> str:
        if path.name == "summary.json":
            rejected.append(path)
            return "0" * 64
        return original_digest(path)

    @contextmanager
    def lock_with_incorrect_digest(path: Path, *, retry_interval_sec: float) -> Iterator[DuckDBWriteSession]:
        with original_lock(path, retry_interval_sec=retry_interval_sec) as session:
            monkeypatch.setattr(legacy, "sha256_file", incorrect_digest)
            yield session

    if stage == "initial":
        monkeypatch.setattr(legacy, "sha256_file", incorrect_digest)
        message = "SHA256 mismatch"
    else:
        monkeypatch.setattr(legacy, "acquire_exclusive_write_conn", lock_with_incorrect_digest)
        message = "changed during ingestion"

    with pytest.raises(legacy.BarEvaluationIntegrityError, match=message):
        legacy.ingest_evaluation(db_path, artifacts=published_artifacts)

    assert [path.name for path in rejected] == ["summary.json"]
    if stage == "initial":
        assert not db_path.exists()
    else:
        with duckdb.connect(str(db_path), read_only=True) as conn:
            assert conn.execute("SHOW TABLES").fetchall() == []


def test_bar_evaluation_ingests_while_legacy_facade_imports(
    tmp_path: Path, published_artifacts: list[dict[str, Any]],
) -> None:
    command = """
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from threading import Event

from utils.bar_evaluation import ingest_evaluation

legacy = "utils.bar_evaluation_store"
assert legacy not in sys.modules
paused, resume = Event(), Event()

def pause_on_entry(frame, event, arg):
    if event == "line" and frame.f_globals.get("__name__") == legacy and not paused.is_set():
        paused.set()
        assert resume.wait(15), "facade import was not resumed"
    return pause_on_entry

def import_legacy():
    sys.settrace(pause_on_entry)
    try:
        import_module(legacy)
    finally:
        sys.settrace(None)

artifacts = json.load(sys.stdin)
with ThreadPoolExecutor(max_workers=1) as executor:
    importing = executor.submit(import_legacy)
    try:
        assert paused.wait(10), "facade import did not start"
        assert "_validate_db_target_path" not in vars(sys.modules[legacy])
        assert ingest_evaluation(sys.argv[1], artifacts=artifacts).status == "inserted"
    finally:
        resume.set()
    importing.result(timeout=10)

assert ingest_evaluation(sys.argv[1], artifacts=artifacts).status == "noop"
"""
    subprocess.run(
        [sys.executable, "-c", command, str(tmp_path / "evaluation.duckdb")],
        cwd=ROOT, input=json.dumps(published_artifacts), text=True, check=True, timeout=45,
    )
