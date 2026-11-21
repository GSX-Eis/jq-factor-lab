# AI factor research workflow on synthetic Bars

The agent proposes two candidate sensors of a fictional market state. Each sees a different noisy measurement of the same state. The generator uses that state to produce the **next** Bar's return. The sensor/return pairs are Gaussian with population correlation near 0.35, matching the declared positive-effect scenario for the two sensor checks. The agent also registers a constant-return control, a constant-signal control, and a sparse control with only eight observed factor values on the same Bar grid. No market data or account is required.

## Reproduce the experiment

From the source checkout, install the demo dependencies with `python3 -m pip install '.[viz]'`. Run this in a writable directory with the environment active.

```bash
python3 -m examples.ai_workflow_showcase > showcase.json
```

The [script](../examples/ai_workflow_showcase.py) prints its fresh evidence directory to stderr. It saves preregistration, CSV inputs, configs, preflight responses, evaluation JSON and stderr, hashed artifacts, and the read-back and dashboard plans. It uses the public CLI entry point for every operational step.

1. Freeze the hypothesis, candidate family, 1-Bar horizon, training and OOS timestamps, costs, and a descriptive training screen before running the CLI. Eligibility for the toy backtest requires at least 512 valid training pairs, Spearman IC of at least 0.20 and a `calibrated_pass` interval.
2. Preflight with `time-series-analysis --dry-run`. Then evaluate the same configs with diagnostics enabled. Read missing IC reasons and interval usability from the training split. Verify that each candidate ledger uses the registered training split and import its recorded trial into the same Holm family. The script also preflights a config that places OOS before the training end to show the structured CLI error.
3. Combine the two eligible signals with equal weights. Run the composite through `time-series-analysis --results-db` so that DuckDB stores its evaluation. In the synthetic OOS slice, use the composite's sign to trade each nonoverlapping 1-Bar open-to-close label at a fixed two-basis-point round-trip cost. This uses the evaluator's actual label timestamps and is a toy transaction simulation.
4. Use `read-bar-evaluations --factor-name equal_weight` and `dashboard --dry-run` against that DuckDB. Compare the database bytes before and after both operations.

See the [statistical gates](../skills/factor-research/references/statistical_gates.md) for the interpretation of calibration and trial completeness. The screen uses synthetic development evidence. A calibrated interval by itself supplies no raw p value, so the example records the full family without making a Holm-adjusted significance claim.

## Recorded output

Selected fields from a run with Python 3.12. The short sample has a finite IC and misses the registered calibration domain.

```json
{
  "precondition": {
    "ok": false,
    "exit_code": 2,
    "error": {"code": "time_series_precondition_failed", "message": "Expected train_end < oos_start"}
  },
  "candidates": {
    "sensor_a": {"eligible_for_toy_backtest": true, "sample_count": 512, "spearman_ic": 0.3464501901157002, "calibration_status": "calibrated_pass"},
    "sensor_b": {"eligible_for_toy_backtest": true, "sample_count": 512, "spearman_ic": 0.35383237274121376, "calibration_status": "calibrated_pass"},
    "constant_return_control": {"eligible_for_toy_backtest": false, "spearman_ic": null, "reason": "constant_return", "calibration_status": "uncalibrated"},
    "constant_signal_control": {"eligible_for_toy_backtest": false, "spearman_ic": null, "reason": "constant_factor", "calibration_status": "uncalibrated"},
    "short_sample": {"eligible_for_toy_backtest": false, "sample_count": 8, "spearman_ic": -0.07142857142857142, "calibration_status": "uncalibrated"}
  },
  "trial_family": {
    "method": "holm",
    "attempt_count": 5,
    "pending_candidates": [],
    "missing_p_value_candidates": [
      ["sensor_a", 1], ["sensor_b", 1], ["constant_return_control", 1],
      ["constant_signal_control", 1], ["short_sample", 1]
    ],
    "selection_complete": false,
    "best": null
  },
  "synthesis": {"factors": ["sensor_a", "sensor_b"], "results_db_status": "inserted"},
  "toy_backtest": {"oos_trades": 532},
  "read_back": {"run_count": 1, "dashboard_read_only": true}
}
```
