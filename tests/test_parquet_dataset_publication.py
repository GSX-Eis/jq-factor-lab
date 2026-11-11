"""Publication recovery is scoped to the failed dataset; all storage is temporary."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import duckdb
import pytest

from data_pipeline.loaders.price_data_to_duckdb import DuckDBPriceImporter
from data_pipeline.loaders.factor_data_to_duckdb import DuckDBFactorImporter
from data_pipeline.loaders.import_index_constituents_duckdb import DuckDBIndexConstituentsImporter
from utils.data_revision import parquet_revision
from utils.duckdb_adapter import DuckDBAdapter


@pytest.fixture
def publication(tmp_path):
    conn = duckdb.connect()
    conn.execute("CREATE TABLE price_data_daily AS SELECT DATE '2020-01-02' date, 'A' stock_code, 10.0 AS close UNION ALL SELECT DATE '2020-01-02', 'B', 20.0")
    conn.execute('CREATE TABLE price_data_5m AS SELECT * FROM price_data_daily')
    conn.execute('CREATE TABLE factor_data AS SELECT * FROM price_data_daily')
    conn.execute("CREATE TABLE index_constituents AS SELECT *, 'I' index_code FROM price_data_daily")
    owner = SimpleNamespace(conn=conn, export_parquet=True, parquet_dir=tmp_path/'lake')
    owner.parquet_dir.mkdir(exist_ok=True)
    adapter = DuckDBAdapter(parquet_dir=owner.parquet_dir, data_bindings={})
    yield owner, adapter
    adapter.close()
    conn.close()


def export(owner, dataset):
    if dataset == 'factor_data':
        return DuckDBFactorImporter.export_to_parquet(owner)
    if dataset == 'index_constituents':
        return DuckDBIndexConstituentsImporter.export_to_parquet(owner)
    return DuckDBPriceImporter.export_to_parquet(owner, table_name=dataset, partition_by=['year', 'month'])


def read(adapter):
    return adapter.fetch_price_data(['A', 'B'], '2020-01-02', '2020-01-02', ['close'])


@pytest.mark.parametrize('other', ['price_data_5m', 'factor_data', 'index_constituents'])
def test_failed_daily_remains_blocked_until_own_recovery(publication, other):
    owner, adapter = publication
    export(owner, 'price_data_daily')
    assert read(adapter).close.tolist() == [10, 20]
    conn = owner.conn

    class PartialCopy:
        def execute(self, sql):
            part = owner.parquet_dir/'price_data_daily'/'year=2020'/'month=1'
            part.mkdir(parents=True, exist_ok=True)
            conn.execute("COPY (SELECT * FROM price_data_daily WHERE stock_code='A') TO ? (FORMAT PARQUET)", [str(part/'partial.parquet')])
            raise OSError('synthetic export failure')

    with patch.object(owner, 'conn', PartialCopy()):
        with pytest.raises(OSError, match='synthetic'):
            export(owner, 'price_data_daily')
    export(owner, other)
    with pytest.raises(RuntimeError, match='incomplete'):
        read(adapter)
    state = json.loads((owner.parquet_dir/'.cache-revision.json').read_text())
    assert state['version'] == 2
    assert state['state'] == 'pending'
    assert state['pending_datasets'] == ['price_data_daily']
    export(owner, 'price_data_daily')
    state = json.loads((owner.parquet_dir/'.cache-revision.json').read_text())
    assert state['state'] == 'ready' and state['pending_datasets'] == []
    assert read(adapter).close.tolist() == [10, 20]


def test_legacy_pending_cannot_be_cleared_by_any_export(publication):
    owner, adapter = publication
    owner.parquet_dir.mkdir(exist_ok=True)
    marker = owner.parquet_dir/'.cache-revision.json'
    content = '{"state":"pending","generation":"legacy"}'
    marker.write_text(content)
    for dataset in ['price_data_daily', 'price_data_5m', 'factor_data', 'index_constituents']:
        with pytest.raises(RuntimeError, match='incomplete'):
            export(owner, dataset)
        assert marker.read_text() == content
        assert not (owner.parquet_dir/dataset).exists()
        with pytest.raises(RuntimeError, match='incomplete'):
            read(adapter)


def test_competing_dataset_keeps_nonblocking_root_lock(publication):
    owner, _ = publication
    conn = owner.conn
    seen = []

    class CompetingCopy:
        def execute(self, sql):
            before = (owner.parquet_dir/'.cache-revision.json').read_bytes()
            with patch.object(owner, 'conn', conn):
                with pytest.raises(BlockingIOError):
                    export(owner, 'price_data_5m')
            assert (owner.parquet_dir/'.cache-revision.json').read_bytes() == before
            seen.append(True)
            return conn.execute(sql)

    with patch.object(owner, 'conn', CompetingCopy()):
        export(owner, 'price_data_daily')
    assert seen == [True]
    assert parquet_revision(owner.parquet_dir)


@pytest.mark.parametrize('payload', [
    {'version': 2, 'state': 'ready', 'generation': 'x', 'pending_datasets': ['price_data_daily']},
    {'version': 3, 'state': 'ready', 'generation': 'x', 'pending_datasets': []},
])
def test_reader_rejects_inconsistent_or_unknown_version(publication, payload):
    owner, _ = publication
    owner.parquet_dir.mkdir(exist_ok=True)
    (owner.parquet_dir/'.cache-revision.json').write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match='incomplete'):
        parquet_revision(owner.parquet_dir)


def test_legacy_ready_is_readable_and_upgraded_on_export(publication):
    owner, _ = publication
    owner.parquet_dir.mkdir(exist_ok=True)
    marker = owner.parquet_dir/'.cache-revision.json'
    marker.write_text('{"state":"ready","generation":"legacy"}')
    assert parquet_revision(owner.parquet_dir) == 'legacy'
    export(owner, 'price_data_daily')
    state = json.loads(marker.read_text())
    assert state['version'] == 2 and state['pending_datasets'] == []
    assert parquet_revision(owner.parquet_dir) != 'legacy'


def test_two_failed_datasets_require_two_recoveries(publication):
    owner, _ = publication

    class FailingCopy:
        def execute(self, sql):
            raise OSError('synthetic')

    with patch.object(owner, 'conn', FailingCopy()):
        for dataset in ['price_data_daily', 'price_data_5m']:
            with pytest.raises(OSError):
                export(owner, dataset)
    export(owner, 'price_data_5m')
    with pytest.raises(RuntimeError, match='incomplete'):
        parquet_revision(owner.parquet_dir)
    assert json.loads((owner.parquet_dir/'.cache-revision.json').read_text())['pending_datasets'] == ['price_data_daily']
    export(owner, 'price_data_daily')
    assert parquet_revision(owner.parquet_dir)
