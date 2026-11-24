"""The Bar package preserves legacy imports and tracks live cache dependencies."""
from importlib import import_module
import inspect
import json
from pathlib import Path
import subprocess
import sys

import duckdb
import pandas as pd
import pytest

from test_factor_base_frequency_routing import probe
from utils.factor_base import FactorBase


ROOT = Path(__file__).resolve().parents[1]
# Freeze the old module namespaces, including imported names exposed by the
# original modules and explicitly imported private helpers.
LEGACY_EXPORTS = [
    ('bar_catalog', 'catalog', '''
        annotations asdict dataclass replace Path re Mapping Sequence duckdb
        Frequency identifier table_exists CatalogEntry BarCatalog register_entry
        create_bar_table
    '''),
    ('bar_quality', 'quality', '''
        annotations logging warnings duckdb pd table_exists BarSchedule LOGGER
        BarDataQualityWarning write_quality scan_quality warn_quality _UNSCANNED
    '''),
    ('bar_source', 'source', '''
        annotations Mapping Sequence Path duckdb pd BarCatalog
        identifier table_exists warn_quality MarketBarReader Frequency
        MissingBarDataError BarSource _utc _instrument_ids
    '''),
    ('market_bar_reader', 'source', '''
        annotations Mapping Sequence Path duckdb pd ContractValidationError
        validate_bar_frame BarAvailability filter_bars_by_view
        calendars_for_bars validate_instrument_metadata
        SessionCalendar Frequency
        MarketBarReader read_market_bars _duration _text _utc _record_raw_coverage
        _instrument_ids _bounds
    '''),
    ('bar_builder', 'builder', '''
        annotations replace hashlib Path Sequence uuid duckdb pd import_transaction
        calendar_name store_instrument_metadata BarCatalog CatalogEntry create_bar_table
        identifier register_entry table_exists scan_quality write_quality MarketBarReader
        SessionCalendar BarSchedule Frequency code_version instrument_calendar
        slot_mapping _instrument_ids _aggregate _metadata
    '''),
    ('futures_roll', 'futures_roll', '''
        annotations hashlib json Integral Sequence uuid duckdb np pd import_transaction
        instrument_calendar BarCatalog identifier table_exists resolve_code_version
        MarketBarReader BarSchedule Frequency product_metadata session_close
        build_roll_schedule _instrument_ids
    '''),
    ('continuous_bars', 'continuous', '''
        annotations Mapping TYPE_CHECKING pd PRICE_COLUMNS instrument_calendar
        BarCatalog table_exists warn_quality BarSource MissingBarDataError
        product_metadata session_close BarSchedule Frequency ADJUSTMENTS
        ContinuousBarSource _utc _record_raw_coverage
    '''),
]


@pytest.mark.parametrize('legacy, current, names', LEGACY_EXPORTS,
                         ids=[row[0] for row in LEGACY_EXPORTS])
def test_legacy_exports_preserve_symbol_identity(legacy, current, names):
    old_module = import_module(f'utils.{legacy}')
    new_module = import_module(f'utils.bars.{current}')
    for name in names.split():
        current_name = 'BarSource' if name == 'MarketBarReader' else name
        assert getattr(old_module, name) is getattr(new_module, current_name), name


def test_market_bar_reader_alias_preserves_context_manager():
    from utils.bars import BarSource, ContinuousBarSource
    from utils.bar_source import BarSource as LegacyBarSource
    from utils.market_bar_reader import MarketBarReader

    assert MarketBarReader is LegacyBarSource is BarSource
    assert issubclass(ContinuousBarSource, MarketBarReader)
    with duckdb.connect() as connection:
        reader = MarketBarReader(connection, config={})
        with reader as entered:
            assert entered is reader
            assert isinstance(entered, BarSource)
            assert entered.connection is connection
            assert entered.config == {}
        assert connection.execute('SELECT 1').fetchone() == (1,)


NEW_MODULES = [f'utils.bars.{name}' for name in
               ('catalog', 'quality', 'source', 'builder', 'futures_roll', 'continuous')]
OLD_MODULES = [f'utils.{row[0]}' for row in LEGACY_EXPORTS]


@pytest.mark.parametrize('modules', [
    [*OLD_MODULES, *NEW_MODULES, 'utils.bars'],
    [*reversed(NEW_MODULES), *reversed(OLD_MODULES), 'utils.bars'],
    ['utils.bars', *NEW_MODULES, *OLD_MODULES],
], ids=['legacy-first', 'new-submodule-first', 'package-first'])
def test_bar_modules_import_in_fresh_process(modules):
    command = ('from importlib import import_module\n'
               'import sys\n'
               'for name in sys.argv[1:]:\n'
               '    import_module(name)\n')
    subprocess.run([sys.executable, '-c', command, *modules], cwd=ROOT, check=True, timeout=30)


@pytest.mark.parametrize('module_file', ['bars/source.py', 'bars/catalog.py', 'bars/quality.py',
                                       '../slots/jq_ashare_local/panel.py',
                                       '../data_pipeline/loaders/market_bar_importer.py'])
def test_bar_implementation_edits_invalidate_factor_cache(tmp_path, monkeypatch, module_file):
    utils = ROOT / 'utils'
    mapping = json.loads((utils / 'legacy_cache_identity.json').read_text())
    source = tmp_path / 'utils' / 'factor_base.py'
    source.parent.mkdir()
    source.write_bytes((utils / source.name).read_bytes())
    for name in (*mapping['dependencies'], 'factor_core/slot_binding.py'):
        target = source.parent / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((utils / name).read_bytes())

    getsourcefile = inspect.getsourcefile
    monkeypatch.setattr(inspect, 'getsourcefile',
                        lambda obj: str(source) if obj is FactorBase else getsourcefile(obj))
    raw = pd.DataFrame({'date': pd.to_datetime(['2026-01-05']), 'code': ['A'], 'close': [10.]})
    item = probe(FactorBase)
    before = item._get_cache_key(raw)
    assert before is not None
    changed = source.parent / module_file
    changed.write_bytes(changed.read_bytes() + b'\n# Future implementation edit\n')
    after = item._get_cache_key(raw)
    assert after is not None and before != after
