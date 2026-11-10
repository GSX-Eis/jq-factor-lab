"""Representative versioned reads use canonical stores without legacy aliases."""

import copy
from dataclasses import asdict

import duckdb
import pandas as pd
import pytest

from utils.data_store import migrate_table, publish_artifact


@pytest.fixture
def config(tmp_path, monkeypatch):
    source, inputs, outputs = (tmp_path / name for name in ("old.duckdb", "input.duckdb", "output.duckdb"))
    dates = pd.bdate_range("2019-01-02", periods=6)
    prices = pd.DataFrame([
        {"date": date, "stock_code": stock, "close": float(10 + day * (i + 1))}
        for day, date in enumerate(dates) for i, stock in enumerate(["A", "B", "C"])
    ])
    with duckdb.connect(str(source)) as conn:
        conn.register("frame", prices)
        conn.execute("CREATE TABLE price_data_daily AS SELECT * FROM frame")
    frozen = migrate_table(source, inputs, "price_data_daily", role="input", execute=True)["ref"]
    with duckdb.connect(str(source)) as conn:
        conn.execute("UPDATE price_data_daily SET close=close*2")
    current = migrate_table(source, inputs, "price_data_daily", role="input", execute=True)["ref"]
    factors = prices.rename(columns={"close": "factor_value"}).assign(factor_name="demo")
    factor = publish_artifact(outputs, "factor_values", frame=factors, metadata={"run_id": "synthetic-read-check"})
    source.unlink()
    monkeypatch.setenv("QUANT_DATA_BINDINGS", '{"input_db":"/missing/legacy.duckdb"}')
    return {
        "input_db": str(inputs), "output_db": str(outputs),
        "price_versions": {"current": current, "frozen": frozen},
        "factor_version": asdict(factor), "factor_name": "demo",
        "stocks": ["A", "B", "C"], "start_date": "2019-01-02", "end_date": "2019-01-09",
    }


def test_canonical_read_paths_match_exact_versions_and_complete_minimal_flow(config):
    from examples.canonical_data_reads import run_checks

    result = run_checks(config)
    assert result["status"] == "passed"
    assert result["price_reads"]["current"]["rows"] == 18
    assert result["price_reads"]["frozen"]["rows"] == 18
    assert result["price_reads"]["current"]["sha256"] != result["price_reads"]["frozen"]["sha256"]
    assert result["factor_read"]["rows"] == 18
    assert result["minimal_flow"]["aligned_rows"] == 15
    assert result["minimal_flow"]["trade_or_publication_executed"] is False


@pytest.mark.parametrize("target", ["current", "frozen", "factor"])
def test_missing_version_fails_instead_of_selecting_current_data(config, target):
    from examples.canonical_data_reads import run_checks

    config = copy.deepcopy(config)
    ref = config["factor_version"] if target == "factor" else config["price_versions"][target]
    ref["version"] = "missing"
    with pytest.raises(ValueError, match="version missing|reference does not exist"):
        run_checks(config)


def test_old_alias_contract_is_rejected(config):
    from examples.canonical_data_reads import run_checks

    config["sources"] = {"/old/database.duckdb": {"role": "input"}}
    with pytest.raises(ValueError, match="explicit DataRefs"):
        run_checks(config)


def test_empty_factor_selection_is_not_reported_as_success(config):
    from examples.canonical_data_reads import run_checks

    config["factor_name"] = "missing"
    with pytest.raises(ValueError, match="No factor rows"):
        run_checks(config)


def test_cli_preserves_existing_receipt(config, tmp_path):
    import json
    from examples.canonical_data_reads import main

    source, receipt = tmp_path / "config.json", tmp_path / "receipt.json"
    source.write_text(json.dumps(config))
    receipt.write_text("original evidence")
    with pytest.raises(SystemExit):
        main(["--config", str(source), "--receipt", str(receipt)])
    assert receipt.read_text() == "original evidence"


@pytest.mark.parametrize('payload', ['{', '[]', '{}', '{"price_versions": null}',
                                   '{"input_db": 1}'])
def test_cli_config_errors_are_structured_without_receipt(payload, tmp_path, capsys):
    import json
    from examples.canonical_data_reads import main

    source, receipt = tmp_path / 'config.json', tmp_path / 'receipt.json'
    source.write_text(payload)
    assert main(['--config', str(source), '--receipt', str(receipt)]) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.err)['error']['code'] == 'canonical_data_reads_config_error'
    assert 'Traceback' not in captured.err + captured.out
    assert not receipt.exists()


@pytest.mark.parametrize(('key', 'value'), [('price_versions', None), ('factor_version', []),
                                         ('stocks', [{}]), ('start_date', 42)])
def test_cli_rejects_invalid_nested_config(config, key, value, tmp_path, capsys):
    import json
    from examples.canonical_data_reads import main

    config[key] = value
    source, receipt = tmp_path / 'config.json', tmp_path / 'receipt.json'
    source.write_text(json.dumps(config))
    assert main(['--config', str(source), '--receipt', str(receipt)]) == 2
    assert json.loads(capsys.readouterr().err)['error']['message']
    assert not receipt.exists()
