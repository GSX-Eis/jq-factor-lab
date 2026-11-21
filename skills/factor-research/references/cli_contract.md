# Public CLI and MCP contract

Use Python 3.11–3.12 and the `jq-factor-lab` console entry point. For a source checkout, install it with `python3 -m pip install '.[synthesis,viz,mcp]'` from the repository root.

| Task | CLI command | stdio MCP tool |
| --- | --- | --- |
| Synthetic daily evaluation | `jq-factor-lab daily-demo` | `run_daily_demo` |
| Synthetic Bar evaluation and DuckDB write | `jq-factor-lab bar-demo` | `run_bar_demo` |
| Trusted daily factor and price CSV | `jq-factor-lab single-factor-analysis --config daily.json` | `run_single_factor_analysis(config_path)` |
| Bar CSV evaluation | `jq-factor-lab time-series-analysis --config bars.json --diagnostics-report --results-db evaluation.duckdb` | `run_time_series_analysis(config_path, results_db)` |
| Bar evaluation lookup | `jq-factor-lab read-bar-evaluations --results-db evaluation.duckdb --limit 20` | `read_bar_evaluations(results_db, factor_name, limit)` |
| Fixed-weight synthesis demo | `jq-factor-lab factor-synthesis-demo --method weighted_sum` | `run_factor_synthesis_demo` |
| Local Streamlit dashboard with `[viz]` | `jq-factor-lab dashboard --results-db evaluation.duckdb` | CLI |

`--dry-run` returns `dry_run: true` and a `plan`. Add it to the evaluation, synthesis, read-back or dashboard CLI command before execution. The MCP read tool is already read-only. Omit diagnostics or pass `--no-diagnostics-report` for cross-sectional Bar analysis. In MCP, set `diagnostics_report: false` for that mode because the tool defaults to `true`. MCP supports both `jq-factor-lab-mcp` and `factorpilot-mcp` as entry points. Use absolute MCP input paths, since the client controls the server's working directory.

Each operational CLI command prints one strict JSON object. Human logs go to stderr and help is plain text. Use `--compact` for compact stdout without child log tails. A success has `ok: true`, `exit_code: 0` and `next_actions`. Daily results include top-level `output_dir`, `ic_mean` and `ic_observations`. Bar analyses include `metrics` and `artifacts`. Errors have `ok: false` and an `error` object with `code`, `message`, `fix` and `retryable`. Undefined numbers are JSON `null`. For exit codes and statistical evidence, see [statistical gates](statistical_gates.md).

Relative daily config paths resolve from the config file. Bar paths and CLI path overrides resolve from the caller's working directory. Use writable paths outside the installed wheel, with the DuckDB outside the artifact directory. Use `--overwrite` only when intentionally replacing generated artifacts. For a working example and read-back procedure, see the [synthetic workflow](https://github.com/GSX-Eis/jq-factor-lab/blob/dev/docs/AI_WORKFLOW_SHOWCASE.md).
