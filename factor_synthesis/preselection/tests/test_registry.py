"""Factory contracts checked without imports from the parent test process."""

from pathlib import Path
import subprocess
import sys
import textwrap


def test_mi_factory_selects_signal_in_fresh_process() -> None:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            import numpy as np
            import pandas as pd
            from factor_synthesis.preselection import get_preselection_strategy

            strategy = get_preselection_strategy('mi', {})
            assert type(strategy).__name__ == 'MIPreSelectionStrategy'
            rng = np.random.default_rng(42)
            signal = rng.normal(size=300)
            X = pd.DataFrame({'signal': signal, 'noise': rng.normal(size=300)})
            y = pd.Series(signal ** 2)
            strategy = get_preselection_strategy('mi', {
                'metrics': {'min_mi': 0.5}, 'fallback_top_n': 0,
            })
            assert strategy.select(X, y) == ['signal']
            report = strategy.get_performance_report().set_index('factor_name')
            assert report.loc['signal', 'mi_score'] >= 0.5
            assert report.loc['noise', 'mi_score'] < 0.5
        """)],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
