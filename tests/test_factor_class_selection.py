"""A native single-factor job must distinguish a class from its compatibility alias."""
import pytest
from backtest_and_analysis.factor_loader import FactorLoader


def test_explicit_class_selection_preserves_default_and_rejects_unknown(tmp_path):
    source = tmp_path / 'factor.py'
    source.write_text('from utils.factor_base import FactorBase\n'
                      'class Original(FactorBase): pass\n'
                      'class Compatibility(Original): pass\n')
    assert {c.__name__ for c in FactorLoader.load_factor_class_from_file(str(source))} == {'Original', 'Compatibility'}
    assert [c.__name__ for c in FactorLoader.load_factor_class_from_file(str(source), 'Original')] == ['Original']
    with pytest.raises(ValueError, match='Expected one factor class'):
        FactorLoader.load_factor_class_from_file(str(source), 'Missing')
    from run_analysis_main import main
    with pytest.raises(ValueError, match='explicit factor_file'):
        main({'factor_class': 'Original'})
