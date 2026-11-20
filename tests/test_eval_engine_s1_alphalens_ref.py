"""Release parity and import boundary for the vendored Alphalens reference."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reference.alphalens_ref import performance as ref_perf
from reference.alphalens_ref import utils as ref_utils


def test_alphalens_reference_matches_installed_package() -> None:
    alphalens = pytest.importorskip("alphalens")
    dates = pd.bdate_range("2024-01-02", periods=6, name="date")
    assets = pd.Index(list("ABCDEFGH"), name="asset")
    index = pd.MultiIndex.from_product([dates, assets])
    rng = np.random.default_rng(791)
    factor = pd.Series(rng.normal(size=len(index)), index=index)
    returns = pd.DataFrame(rng.normal(0, 0.02, (len(index), 2)), index=index, columns=["1D", "5D"])
    factor.iloc[0] = np.nan
    returns.iloc[-1, 1] = np.nan
    groups = {asset: number % 2 for number, asset in enumerate(assets)}

    clean = ref_utils.get_clean_factor(factor, returns, groupby=groups, quantiles=2)
    installed_clean = alphalens.utils.get_clean_factor(factor, returns, groupby=groups, quantiles=2)
    pd.testing.assert_frame_equal(clean, installed_clean, check_exact=True)
    assert len(clean) == len(index) - 2
    pd.testing.assert_series_equal(
        ref_utils.quantize_factor(clean, quantiles=2, by_group=True),
        alphalens.utils.quantize_factor(installed_clean, quantiles=2, by_group=True),
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        ref_perf.factor_information_coefficient(clean, group_adjust=True, by_group=True),
        alphalens.performance.factor_information_coefficient(installed_clean, group_adjust=True, by_group=True),
        check_exact=True,
    )
    means, errors = ref_perf.mean_return_by_quantile(clean, by_date=True)
    installed_means, installed_errors = alphalens.performance.mean_return_by_quantile(installed_clean, by_date=True)
    pd.testing.assert_frame_equal(means, installed_means, check_exact=True)
    pd.testing.assert_frame_equal(errors, installed_errors, check_exact=True)
    spread, spread_error = ref_perf.compute_mean_returns_spread(means, 2, 1, errors)
    installed_spread, installed_spread_error = alphalens.performance.compute_mean_returns_spread(
        installed_means, 2, 1, installed_errors
    )
    pd.testing.assert_frame_equal(spread, installed_spread, check_exact=True)
    pd.testing.assert_frame_equal(spread_error, installed_spread_error, check_exact=True)
    pd.testing.assert_series_equal(
        ref_utils.std_conversion(errors["5D"], "1D"),
        alphalens.utils.std_conversion(installed_errors["5D"], "1D"),
        check_exact=True,
    )
    pd.testing.assert_series_equal(
        ref_perf.factor_rank_autocorrelation(clean),
        alphalens.performance.factor_rank_autocorrelation(installed_clean),
        check_exact=True,
    )
    pd.testing.assert_series_equal(
        ref_perf.quantile_turnover(clean["factor_quantile"], 2),
        alphalens.performance.quantile_turnover(installed_clean["factor_quantile"], 2),
        check_exact=True,
    )


@pytest.mark.slow
def test_production_does_not_import_alphalens_reference() -> None:
    root = Path(__file__).resolve().parents[1]
    source_dirs = (
        "backtest_and_analysis", "config", "data_pipeline", "examples", "factor_mining",
        "factor_synthesis", "factors", "scripts", "tools", "utils",
    )
    paths = list(root.glob("*.py"))
    for directory in source_dirs:
        paths.extend((root / directory).rglob("*.py"))

    forbidden = ("tests.reference", "reference", "alphalens_ref")
    violations = []
    for path in sorted(paths):
        for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports = [module, *(f"{module}.{alias.name}".lstrip(".") for alias in node.names)]
            else:
                continue
            if any(name == prefix or name.startswith(prefix + ".") for name in imports for prefix in forbidden):
                violations.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not violations, "\n".join(violations)
