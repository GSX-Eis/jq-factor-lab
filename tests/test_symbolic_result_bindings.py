import json

import duckdb
import pytest

from factor_mining.symbolic.config import AnalysisContext
from factor_mining.symbolic import reranker
from utils.data_store import migrate_table


def test_reranker_reads_only_frozen_result_versions_without_legacy_files(
    tmp_path, monkeypatch
):
    output = tmp_path / "output.duckdb"
    sources = []
    refs = []
    for index, value in enumerate([1.0, 9.0]):
        source = tmp_path / f"old{index}.duckdb"
        with duckdb.connect(str(source)) as conn:
            conn.execute(
                "CREATE TABLE factor_values AS SELECT DATE '2019-01-02' AS date, 'a' AS stock_code, "
                "'same' AS factor_name, ?::DOUBLE AS factor_value",
                [value],
            )
            conn.execute(
                "CREATE TABLE factor_analysis_results AS SELECT DATE '2019-01-02' AS date, 'same' AS factor_name, "
                "1 AS period, ?::DOUBLE AS ic",
                [value / 10],
            )
            conn.execute(
                "CREATE TABLE monthly_pool_results AS SELECT 'same' AS factor_name, 'p' AS pool_type, "
                "'2019-01' AS year_month, 1 AS period_days, ?::DOUBLE AS long_short_return",
                [value / 100],
            )
        refs.append(
            {
                name: migrate_table(source, output, name, role="output", execute=True)[
                    "ref"
                ]
                for name in [
                    "factor_values",
                    "factor_analysis_results",
                    "monthly_pool_results",
                ]
            }
        )
        sources.append(source)
        source.unlink()
    selected = {
        "role": "output",
        "outputs": {"factor_values": refs[0]["factor_values"]},
        "evaluations": {
            name: refs[0][name]
            for name in ["factor_analysis_results", "monthly_pool_results"]
        },
    }
    contract = {"output_db": str(output), **selected}
    monkeypatch.setenv("QUANT_DATA_BINDINGS", json.dumps(contract))
    context = AnalysisContext(
        "",
        {},
        {},
        {},
        {},
        {
            "results_db_backend": "duckdb",
            "results_duckdb_path": str(output),
            "factor_values_table": "main.factor_values",
            "results_output_table": "main.factor_analysis_results",
            "results_monthly_output_table": "main.monthly_pool_results",
        },
    )
    dates = ("2019-01-01", "2019-01-31")
    assert reranker._query_factor_values(
        context, ["same"], *dates
    ).factor_value.tolist() == [1.0]
    metrics, warnings = reranker._query_validation_metrics(
        context,
        "same",
        {"periods": [1], "validation_start": dates[0], "validation_end": dates[1]},
    )
    assert not warnings and metrics["1"]["rank_ic_mean"] == pytest.approx(0.1)
    assert metrics["1"]["long_short_return_mean"] == pytest.approx(0.01)
    assert reranker._query_monthly_long_short(
        context, ["same"], 1, *dates
    ).long_short_return.tolist() == [0.01]
    selected["outputs"]["factor_values"] = refs[1]["factor_values"]
    monkeypatch.setenv("QUANT_DATA_BINDINGS", json.dumps(contract))
    assert reranker._query_factor_values(
        context, ["same"], *dates
    ).factor_value.tolist() == [9.0]
    selected["outputs"].clear()
    selected["evaluations"].pop("monthly_pool_results")
    monkeypatch.setenv("QUANT_DATA_BINDINGS", json.dumps(contract))
    with pytest.raises(ValueError, match="no frozen version"):
        reranker._query_factor_values(context, ["same"], *dates)
    with pytest.raises(ValueError, match="no frozen version"):
        reranker._query_monthly_long_short(context, ["same"], 1, *dates)


def test_analysis_child_records_generated_source_without_mutating_parent(tmp_path, monkeypatch):
    import hashlib
    import json
    import os
    from types import SimpleNamespace
    from factor_mining.symbolic import validator

    factor, config = tmp_path / "factor.py", tmp_path / "analysis.json"
    factor.write_text("class GeneratedFactor: pass\n")
    config.write_text('{"analysis_period":{"end_date":"2023-12-31"}}')
    context = {"run_id": "symbolic", "source_hash": "compiler", "symbolic_candidate_rank": 2}
    monkeypatch.setenv("QUANT_OUTPUT_CONTEXT", json.dumps(context))
    def run(command, **kwargs):
        child = json.loads(kwargs["env"]["QUANT_OUTPUT_CONTEXT"])
        assert child == {**context,
            "generated_factor_source_hash": hashlib.sha256(factor.read_bytes()).hexdigest(),
            "effective_analysis_config_hash": hashlib.sha256(config.read_bytes()).hexdigest()}
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(validator.subprocess, "run", run)
    assert validator.run_analysis_smoke(factor, str(config))["ok"]
    assert json.loads(os.environ["QUANT_OUTPUT_CONTEXT"]) == context
