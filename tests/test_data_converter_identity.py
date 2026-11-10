"""Qlib conversion is reversible and cannot merge distinct input keys."""
import pandas as pd
import pytest

from factor_synthesis.qlib_adapter.data_converter import DataConverter, FactorPilotDataset


def original_index():
    return pd.MultiIndex.from_product(
        [pd.date_range('2024-01-02', periods=3), ['600000.SH', '000001.XSHE', 'DEMO000', '430001.BJ']],
        names=['date', 'stock_code'],
    )


def test_predictions_restore_original_keys_in_prediction_order():
    original = original_index()
    converted = DataConverter.convert_index_to_qlib_format(original)
    positions = [11, 0, 5, 3]
    predictions = pd.Series([.1, .2, .3, .4], index=converted.take(positions), name='score')
    restored = DataConverter.convert_predictions_to_series(predictions, original)
    pd.testing.assert_index_equal(restored.index, original.take(positions))
    assert restored.tolist() == predictions.tolist()
    assert restored.name == predictions.name


@pytest.mark.parametrize('same_day', [False, True])
def test_alias_collisions_are_rejected_even_across_dates(same_day):
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp('2024-01-02'), '600000.SH'),
         (pd.Timestamp('2024-01-02' if same_day else '2024-01-03'), '600000.XSHG')],
        names=['date', 'stock_code'],
    )
    with pytest.raises(ValueError, match='alias collision'):
        DataConverter.convert_index_to_qlib_format(index)


@pytest.mark.parametrize('duplicate_side', ['factors', 'returns'])
def test_dataset_rejects_duplicate_keys_before_alignment(duplicate_side):
    index = original_index()
    factors = pd.DataFrame({'signal': range(len(index))}, index=index)
    returns = pd.Series(.01, index=index)
    if duplicate_side == 'factors':
        factors = pd.concat([factors, factors.iloc[:1]])
    else:
        returns = pd.concat([returns, returns.iloc[:1]])
    with pytest.raises(ValueError, match='Duplicate input keys'):
        DataConverter.create_dataset_from_factors(factors, returns, *(['2024-01-02', '2024-01-04'] * 3))




def test_unknown_or_duplicate_prediction_keys_fail():
    index = original_index()
    converted = DataConverter.convert_index_to_qlib_format(index)
    with pytest.raises(ValueError, match='Duplicate Qlib'):
        DataConverter.convert_index_from_qlib_format(converted.append(converted[:1]), index)
    with pytest.raises(ValueError, match='missing from original'):
        DataConverter.convert_index_from_qlib_format(converted, index[1:])


@pytest.mark.parametrize('method', ['lasso', 'lightgbm'])
@pytest.mark.parametrize('issue', ['duplicate_factors', 'duplicate_returns', 'alias'])
def test_pipeline_rejects_invalid_keys_before_preprocessing(method, issue):
    import importlib
    import logging
    from types import SimpleNamespace

    index = original_index()
    factors = pd.DataFrame({'signal': range(len(index))}, index=index)
    returns = pd.Series(.01, index=index)
    expected = 'Duplicate input keys'
    if issue == 'duplicate_factors':
        factors = pd.concat([factors, factors.iloc[:1]])
    elif issue == 'duplicate_returns':
        returns = pd.concat([returns, returns.iloc[:1]])
    else:
        alias = factors.iloc[:1].copy()
        alias.index = pd.MultiIndex.from_tuples([(index[0][0], '600000.XSHG')], names=index.names)
        factors = pd.concat([factors, alias])
        expected = 'alias collision'
    runner = getattr(importlib.import_module(f'factor_synthesis.pipeline.{method}_method'),
                     f'run_{method}_synthesis')
    with pytest.raises(ValueError, match=expected):
        runner(SimpleNamespace(logger=logging.getLogger(__name__)), factors, returns,
               '2024-01-02', '2024-01-03', {})
