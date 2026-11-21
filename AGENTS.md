# JQ Factor Lab agent guide

Use [the quickstart](README.md#60-second-quickstart) to install JQ Factor Lab with Python 3.11–3.12. Work in a writable directory you own. Start with the synthetic fixtures, then follow the [data import](docs/DATA_IMPORT.md), [factor interface](docs/FACTOR_INTERFACE.md), [Bar input](docs/TIME_SERIES_CONTRACT.md), and [calendar](docs/TRADING_CALENDAR.md) contracts when switching to your own inputs.

Use the [Factor Research Skill](skills/factor-research/SKILL.md) for the research loop and the [AI workflow showcase](docs/AI_WORKFLOW_SHOWCASE.md) for a reproducible example.

## Commands

| Task | Command |
| --- | --- |
| Evaluate synthetic daily cross-sections | `jq-factor-lab daily-demo` |
| Evaluate one synthetic minute instrument and persist the result | `jq-factor-lab bar-demo` |
| Evaluate a custom `FactorBase` factor against daily multi-asset CSV | `jq-factor-lab single-factor-analysis --config config.json` |
| Evaluate local Bar inputs in a declared mode and request diagnostics in time-series mode | `jq-factor-lab time-series-analysis --config bar-config.json --diagnostics-report` |
| Query a persisted Bar evaluation read-only | `jq-factor-lab read-bar-evaluations --results-db evaluations.duckdb` |
| Try synthetic fixed-weight synthesis after installing `.[synthesis]` | `jq-factor-lab factor-synthesis-demo --method weighted_sum` |
| Inspect the local read-only dashboard plan after installing `.[viz]` | `jq-factor-lab dashboard --results-db evaluations.duckdb --dry-run` |
| Package trusted factor files into one JoinQuant Python file offline | `python3 -m scripts.build_jq_strategy --help` |

For the full builder invocation and platform inputs, see [JoinQuant packaging](docs/JOINQUANT.md).

## Automation and outputs

- Preflight with `--dry-run`. Capture stdout and stderr separately. Operational commands emit strict JSON with `ok` and `exit_code`. Success includes `next_actions`, and failures include `error.code`, `message`, `fix`, and `retryable`. Daily metrics stay at the top level, and non-finite values are `null`. Bar results include `metrics` and `artifacts`.
- **0** means completion, including a dry run. **1** means an execution or dependency failure, for example `insufficient_cross_section_assets` on a single-asset daily input. **2** means invalid arguments or preconditions. See the [CLI and MCP mapping](skills/factor-research/references/cli_contract.md) for details.
- Let the demos allocate fresh runs. For explicit output directories use a new path per experiment. An existing directory requires `--overwrite` to update generated artifacts. Keep output and DuckDB paths disjoint. Relative paths in public Bar commands, including config values, resolve from the caller's current working directory (cwd). Daily config paths are relative to the config file. The daily CLI flag `--output-dir` resolves from cwd.
- In a Bar result, find `artifact_type == "evaluation_summary"`. This evaluation `summary.json` is written atomically last and marks publication of the four files (the summary plus `per_bar_labels.csv`, `ic_by_horizon.csv` and `quantile_stats.csv`). Verify the files and recorded hashes for the same attempt. The separate Harness summary records process status. When requested, `metrics.diagnostics_report` and its CSV and Markdown artifacts describe IC, sample exclusions and uncertainty. Keep a missing IC as `null` and inspect its reason.
- From a read-only wheel installation, use writable caller-owned paths for outputs, scratch and cache. Public Bar commands place them under the caller's Harness `--output-root`, which defaults to `outputs/agent_harness` under cwd. See the [output layout](docs/PUBLIC_SCOPE.md) and [daily configuration](docs/CONFIG.md).

## Time and data rules

- Calculate each signal using information available by its signal time. Supply raw prices and positive adjustment factors for point-in-time (PIT) adjustment. `FactorBase` applies PIT when enabled. Use trailing windows and avoid applying adjustment twice.
- Set daily `execution_lag` (default 1 session) and Bar `entry_offset_bars` (default 1) to reflect execution availability. Check explicit trading-calendar coverage across inputs and actual label exits. Derive session boundaries from the calendar, including holidays and interruptions. The demo calendar is for the synthetic fixture.
- Use actual label exit dates and time-isolated training and validation. If `LabelIsolationError` reports missing exit evidence, repair or exclude those labels. Fabricating exits undermines the evaluation. See the [synthesis contract](docs/SYNTHESIS.md).
- Load trusted factor files only. Keep real market data, credentials and outputs outside version control. Use synthetic results and single-factor evaluation for screening. Strategy performance, cloud execution and live trading each require separate validation.

## Three-minute autonomous factor screen

The following standalone example shows a descriptive daily IC screen. For the multi-candidate Bar workflow with a frozen split, calibration and DuckDB read-back, use the [AI workflow showcase](docs/AI_WORKFLOW_SHOWCASE.md).

### Standalone example

With the quickstart environment active, run this from a writable directory. It creates a new research directory on every invocation and uses only the installed examples and synthetic calendar. The screening rule is fixed before evaluation and uses the 1D IC from the persisted `summary.json`. If the 1D IC is unavailable, the example reports an error and defers the decision.

```bash
(
set -e
work_dir="$(mktemp -d "$PWD/jq-research.XXXXXX")"
cd "$work_dir"
python3 -m examples.generate_data --output prices.csv
python3 - <<'PY'
from pathlib import Path

from examples import generate_data

calendar = Path(generate_data.__file__).resolve().parents[1] / "config" / "demo_calendar.json"
Path("calendar.json").write_bytes(calendar.read_bytes())
PY

cat > reversal.py <<'PY'
import pandas as pd

from utils.factor_base import FactorBase


class ThreeSessionReversal(FactorBase):
    def _get_cache_inputs(self) -> dict[str, object]:
        return {}

    def _get_raw_data_for_factor(self) -> pd.DataFrame:
        return self._fetch_raw_data(["close", "factor"], table_name="price_data_daily")

    def calculate_factor(self, data: pd.DataFrame) -> pd.DataFrame:
        prices = data.pivot(index="date", columns="code", values="close").sort_index()
        return -prices.pct_change(periods=3, fill_method=None)
PY

cat > config.json <<'JSON'
{
  "input_csv": "prices.csv",
  "calendar_file": "calendar.json",
  "factor_file": "reversal.py",
  "factor_class": "ThreeSessionReversal",
  "execution_lag": 1,
  "periods": [1, 5],
  "output_dir": "evaluation"
}
JSON

jq-factor-lab single-factor-analysis --config config.json > daily.json 2> daily.stderr.log
python3 - <<'PY'
import json
import math
from pathlib import Path

config = json.loads(Path("config.json").read_text())
path = (Path(config["output_dir"]) / "summary.json").resolve()
result = json.loads(path.read_text())
ic_1d = result["ic_mean"].get("1D")
if ic_1d is None or not math.isfinite(ic_1d):
    raise SystemExit("No usable 1D IC for the screening decision")
n_1d = result["ic_observations"]["1D"]
decision = "keep for further study" if n_1d >= 100 and ic_1d >= 0.02 else "drop"
print(json.dumps({
    "decision": decision,
    "ic_mean": result["ic_mean"],
    "ic_observations": result["ic_observations"],
    "summary_path": str(path),
}))
PY
)
```

If the evaluator exits nonzero, inspect `daily.stderr.log` and `daily.json` in that experiment directory. For the Bar check, use the quickstart's `bar-demo` and `read-bar-evaluations` commands and compare persisted `metrics.ic_by_horizon` with the actual diagnostic sample counts, IC values and reasons.
