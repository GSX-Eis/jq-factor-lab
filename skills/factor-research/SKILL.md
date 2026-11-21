---
name: factor-research
description: Plan, run, and review quantitative factor experiments with JQ Factor Lab. Use for synthetic or local daily and Bar evaluation, IC diagnostics, calibration and TrialLedger review, multi-factor synthesis, backtesting, or read-only DuckDB result inspection.
---

# Factor research

1. State the economic hypothesis, factor formula, signal availability, universe, calendar, execution lag, and forward horizon. Before execution, freeze the candidate family, acceptance rule, split, inference settings, and evidence needed for the conclusion. Use the synthetic fixture unless the user provides trusted local inputs and authorizes their use.
2. Preflight the exact command with `--dry-run`. Choose `single-factor-analysis` for a daily cross-section and `time-series-analysis` for Bar CSV. Explicitly set `analysis_mode` for Bar input. Read the [CLI and MCP contract](references/cli_contract.md) for paths, outputs, and invocation details. Keep each attempt in a fresh caller-owned directory.
3. Run the planned evaluation. Record the process exit code, JSON stdout, stderr, config and current artifact paths. Verify the published Bar summary and file hashes before using the metrics. Preserve null ICs and their `reason`. Inspect each split's `interval_usability.calibration_status` and the complete selection `trial_ledger`. Apply the [statistical gates](references/statistical_gates.md) before drawing conclusions. A completed command can report unusable research evidence.
4. Synthesize only the eligible inputs for the declared purpose. Use fixed weights for the supplied toy demos. For fitted methods, fit weights, preprocessing and quantile edges on training only, with purge based on verified label exits. Keep the final holdout sealed until the plan is frozen and access is authorized.
5. Backtest frozen scores with tradable timestamps, execution lag, holding period and explicit costs. Distinguish IC and group-return diagnostics from simulated portfolio returns. When requested, persist Bar evaluations to a caller-owned DuckDB and verify them with `read-bar-evaluations`. Use `dashboard --dry-run` to check the read-only dashboard plan.
6. Report all attempts and exclusions, split sizes, IC and diagnostics, interval usability, adjusted inference where available, costs, artifacts, and the evidence that supports each decision. Preserve the original decision rule and do not overwrite the run.

Never use future observations in signal construction, leak local data or credentials, open an unauthorized holdout, fabricate label exits, hide missing trials, or modify statistical gates to obtain a passing result. Synthetic runs verify the workflow and support no trading recommendation.

For a complete offline example, see the [synthetic workflow](https://github.com/GSX-Eis/jq-factor-lab/blob/dev/docs/AI_WORKFLOW_SHOWCASE.md).
