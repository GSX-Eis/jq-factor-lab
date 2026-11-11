import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from examples.generate_data import make_prices
from examples.momentum import DemoMomentum
from tools.demo_cli import analyze, read_prices
from utils.pit import apply_pit_adjustment
from backtest_and_analysis.forward_returns_calculator import ForwardReturnsCalculator


def test_full_public_demo_and_no_output_overwrite(tmp_path):
    frame = make_prices(sessions=80)
    frame.to_csv(tmp_path / "input.csv", index=False)
    config = {
        "input_csv": "input.csv", "output_dir": "result",
        "calendar_file": str(Path(__file__).resolve().parents[1] / "config/demo_calendar.json"),
        "factor_file": str(Path(__file__).resolve().parents[1] / "examples/momentum.py"),
        "factor_class": "DemoMomentum", "periods": [1, 5], "execution_lag": 1,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    result = analyze(path)
    assert result["assets"] == 20
    assert result["ic_observations"]["5D"] == 69
    assert np.isfinite(list(result["ic_mean"].values())).all()
    assert (tmp_path / "result/summary.json").is_file()
    with pytest.raises(FileExistsError):
        analyze(path)


@pytest.mark.parametrize("issue", ["duplicate", "code", "price", "date", "factor"])
def test_csv_rejects_invalid_input(tmp_path, issue):
    frame = make_prices(sessions=2, assets=2)
    if issue == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif issue == "code":
        frame.loc[0, "stock_code"] = "X';DROP TABLE prices;--"
    elif issue == "price":
        frame.loc[0, "close"] = np.nan
    elif issue == "factor":
        frame.loc[0, "factor"] = 0
    else:
        frame.loc[0, "date"] = pd.NaT
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        read_prices(path)


def test_lagged_returns_have_exact_entry_and_exit():
    dates = pd.bdate_range("2020-01-01", periods=5)
    prices = pd.DataFrame({"A": [10., 20., 30., 40., 50.]}, index=dates).stack()
    prices.index.names = ["date", "asset"]
    from utils.trading_calendar import TradingCalendar
    result = ForwardReturnsCalculator(trading_calendar=TradingCalendar(dates, dates.min(), dates.max())).calculate_forward_returns(prices, (2,), execution_lag=1)
    assert result.loc[(dates[0], "A"), "2D"] == pytest.approx(1.0)
    assert result.index.get_level_values("date").max() == dates[1]


def test_pit_adjustment_uses_window_anchor_and_rejects_invalid_factor():
    frame = pd.DataFrame({"code": ["A", "A"], "date": pd.date_range("2020-01-01", periods=2),
                          "close": [100., 50.], "volume": [10., 20.], "factor": [1., 2.]})
    result = apply_pit_adjustment(frame)
    assert result.close.tolist() == [50., 50.]
    assert result.volume.tolist() == [20., 20.]
    frame.loc[0, "factor"] = 0
    with pytest.raises(ValueError):
        apply_pit_adjustment(frame)


def test_demo_signal_does_not_read_future_rows():
    frame = make_prices(sessions=20, assets=2).rename(columns={"stock_code": "code"})
    factor = object.__new__(DemoMomentum)
    full = factor.calculate_factor(frame)
    prefix = factor.calculate_factor(frame[frame.date <= full.index[10]])
    pd.testing.assert_frame_equal(full.loc[prefix.index], prefix)


def test_cli_rejects_constant_factor_without_publishing_results(tmp_path, capsys):
    from tools.demo_cli import main

    make_prices(sessions=80).to_csv(tmp_path / 'input.csv', index=False)
    factor_file = tmp_path / 'constant.py'
    factor_file.write_text(
        'from examples.momentum import DemoMomentum\n'
        'class ConstantSignal(DemoMomentum):\n'
        '    def calculate_factor(self, data):\n'
        '        return data.pivot(index="date", columns="code", values="close") * 0 + 1\n'
    )
    config = tmp_path / 'constant.json'
    config.write_text(json.dumps({
        'input_csv': 'input.csv', 'output_dir': 'result',
        'calendar_file': str(Path(__file__).resolve().parents[1] / 'config/demo_calendar.json'),
        'factor_file': str(factor_file), 'factor_class': 'ConstantSignal',
        'periods': [1, 5], 'execution_lag': 1,
    }))
    with pytest.raises(SystemExit) as stopped:
        main(['single-factor-analysis', '--config', str(config)])
    assert stopped.value.code == 1
    assert capsys.readouterr().err == (
        "Analysis failed: max_loss exceeded: loss=1, max_loss=0.35, retained=0/1600\n"
    )
    assert not (tmp_path / 'result').exists()


@pytest.mark.parametrize('payload, expected', [
    ([], 'JSON object'),
    ({'input_csv': 'a', 'factor_file': 'b', 'factor_class': 'C', 'output_dir': 'out', 'periods': 1}, 'periods'),
    ({'input_csv': 123, 'factor_file': 'b', 'factor_class': 'C', 'output_dir': 'out'}, 'input_csv'),
    ({'input_csv': 'a', 'factor_file': 'b', 'output_dir': 'out'}, 'factor_class'),
])
def test_cli_config_errors_are_actionable_without_tracebacks(tmp_path, capsys, payload, expected):
    from tools.demo_cli import main
    config = tmp_path/'bad.json'
    config.write_text(json.dumps(payload))
    with pytest.raises(SystemExit) as error:
        main(['single-factor-analysis', '--config', str(config)])
    assert error.value.code == 1
    output = capsys.readouterr()
    assert output.out == '' and expected in output.err and 'Traceback' not in output.err


def test_cli_stdout_is_one_json_document(tmp_path, capsys):
    from tools.demo_cli import main
    frame = make_prices(sessions=80)
    frame.to_csv(tmp_path/'input.csv', index=False)
    source = tmp_path/'noisy.py'
    source.write_text('from examples.momentum import DemoMomentum\nprint("user module log")\nclass NoisyMomentum(DemoMomentum):\n    pass\n')
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'input_csv': 'input.csv', 'factor_file': str(source),
        'calendar_file': str(Path(__file__).resolve().parents[1] / 'config/demo_calendar.json'),
        'factor_class': 'NoisyMomentum', 'output_dir': 'out', 'periods': [1,5]}))
    assert main(['single-factor-analysis', '--config', str(config)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)['factor'] == 'NoisyMomentum'
    assert 'user module log' in captured.err


def test_generate_existing_file_has_short_error(tmp_path):
    import subprocess
    import sys
    path = tmp_path/'existing.csv'
    path.write_text('keep me')
    result = subprocess.run([sys.executable, '-m', 'examples.generate_data', '--output', str(path)], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'choose a new writable file path' in result.stderr and 'Traceback' not in result.stderr
    assert path.read_text() == 'keep me'


def test_cli_rejects_constant_returns_without_publishing_results(tmp_path, capsys):
    from tools.demo_cli import main

    prices = make_prices(sessions=80)
    prices['close'] = prices['stock_code'].map(
        {code: float(i + 1) for i, code in enumerate(sorted(prices.stock_code.unique()))}
    )
    prices['factor'] = 1.0
    prices.to_csv(tmp_path / 'input.csv', index=False)
    factor_file = tmp_path / 'levels.py'
    factor_file.write_text(
        'from examples.momentum import DemoMomentum\n'
        'class LevelSignal(DemoMomentum):\n'
        '    def calculate_factor(self, data):\n'
        '        return data.pivot(index="date", columns="code", values="close")\n'
    )
    config = tmp_path / 'levels.json'
    config.write_text(json.dumps({
        'input_csv': 'input.csv', 'output_dir': 'result',
        'calendar_file': str(Path(__file__).resolve().parents[1] / 'config/demo_calendar.json'),
        'factor_file': str(factor_file), 'factor_class': 'LevelSignal',
        'periods': [1, 5], 'execution_lag': 1,
    }))
    with pytest.raises(SystemExit) as stopped:
        main(['single-factor-analysis', '--config', str(config)])
    assert stopped.value.code == 1
    assert 'no usable IC observations' in capsys.readouterr().err
    assert not (tmp_path / 'result').exists()
