"""PIT-derived cache identities and mixed cache hits use the same money values."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from factor_mining.symbolic import data_loader as loader
from factor_mining.symbolic.feature_space import FeatureSpec


def context(cache_dir):
    return SimpleNamespace(
        runtime_config={'cache_dir': str(cache_dir), 'start_date': '2024-01-02',
                        'end_date': '2024-01-02', 'data_snapshot_hash': 'synthetic'},
        raw_config={'pit_adjustment': {'enabled': True}}, config_path='synthetic.json')


@pytest.mark.parametrize('kind', ['intraday', 'panel'])
def test_pit_semantic_version_changes_derived_cache_identity(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(loader, 'binding_contract', lambda: {})
    ctx = context(tmp_path)
    spec = FeatureSpec('intraday_mean_money', 'price_5m', 'money', '5m', 'mean')
    def path():
        if kind == 'intraday':
            return loader._intraday_feature_cache_path(ctx, spec, ['A'])
        return loader._panel_cache_path({}, ctx, ['A'], [])

    monkeypatch.setattr(loader, '_PIT_CACHE_SEMANTICS', 'legacy', raising=False)
    old = path()
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_bytes(b'historical-cache-must-survive')
    monkeypatch.setattr(loader, '_PIT_CACHE_SEMANTICS', 'pit-money-v2')
    assert path() != old
    assert old.read_bytes() == b'historical-cache-must-survive'
    assert not path().exists()


def test_symbolic_money_cold_warm_and_mixed_requests_agree(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, 'binding_contract', lambda: {})
    raw = pd.DataFrame({'date': pd.to_datetime(['2024-01-02 09:35', '2024-01-02 09:40']),
                        'code': ['A', 'A'], 'factor': [1., 0.],
                        'money': [10., 0.], 'close': [10., np.nan]})
    queries = []
    def query(ctx, table, fields, stocks):
        queries.append(list(fields))
        return raw[['date', 'code'] + fields].copy()

    monkeypatch.setattr(loader, '_query_table', query)
    money = FeatureSpec('intraday_mean_money', 'price_5m', 'money', '5m', 'mean')
    close = FeatureSpec('intraday_mean_close', 'price_5m', 'close', '5m', 'mean')
    both = {money.name: money, close.name: close}
    ctx = context(tmp_path / 'mixed')
    # Cache close first, so the next request fetches only money.
    loader._load_cached_intraday_feature_panel(ctx, {close.name: close}, ['A'])
    mixed = loader._load_cached_intraday_feature_panel(ctx, both, ['A'])
    assert queries[-1] == ['money', 'factor']
    count = len(queries)
    warm = loader._load_cached_intraday_feature_panel(ctx, both, ['A'])
    assert len(queries) == count
    cold = loader._load_cached_intraday_feature_panel(context(tmp_path / 'cold'), both, ['A'])
    pd.testing.assert_frame_equal(mixed.sort_index(axis=1), warm.sort_index(axis=1), check_exact=True)
    pd.testing.assert_frame_equal(mixed.sort_index(axis=1), cold.sort_index(axis=1), check_exact=True)
    assert cold[money.name].iloc[0] == 10.


def test_symbolic_daily_reads_same_paused_evidence(tmp_path, monkeypatch):
    import duckdb
    path = tmp_path / 'prices.duckdb'
    with duckdb.connect(str(path)) as conn:
        conn.execute("""CREATE TABLE price_data_daily AS SELECT 'A' AS stock_code,
            DATE '2022-03-08' AS date, 1. AS factor, 0. AS money, 1 AS paused""")
    ctx = SimpleNamespace(runtime_config={
        'start_date': '2022-03-08', 'end_date': '2022-03-08', 'duckdb_path': str(path)},
        raw_config={'pit_adjustment': {'enabled': True}})
    monkeypatch.setattr(loader, '_input_duckdb_path', lambda _: str(path))
    result = loader.load_daily_price_panel(ctx, ['money'], ['A'])
    assert pd.isna(result.daily_money.iloc[0])
    assert pd.isna(result.factor.iloc[0])
    assert 'paused' not in result
