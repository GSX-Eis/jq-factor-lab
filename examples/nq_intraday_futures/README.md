# NQ 1min Intraday Futures Example (ETH & RTH)

Self-contained declarative `MarketSlot` example for single-contract `NQZ26` 1-minute bars (`CMEGlobex_Equity` with `NYSE` RTH reference calendar) using purely synthetic integer-priced data (`bars.csv`) and a strictly causal vectorized factor (`sample_factor.py`).

Both `slot_eth.yaml` (`view: ETH`) and `slot_rth.yaml` (`view: RTH`) read the exact same full-session ETH CSV (`timestamp,symbol,open,high,low,close,volume`) with UTC `bar_start` timestamps, `volume_mode: interval`, and `NQZ26` contract metadata (`contract_multiplier: 20.0`, `expiry_date: "2026-12-18"`).

## Execution & Evaluation Contract

- `execution_profile`: `entry_price_field: open`, `exit_price_field: close`, `min_entry_offset_bars: 1`, `exit_adjustment: -1`, `boundary_policy: same_session`, `return_unit: decimal`, `gap_rule: endpoints_only`, `allow_short: true`.
- Evaluation settings (`eval_eth.json` / `eval_rth.json`): `analysis_mode: time_series`, `frequency: 1m`, `entry_offset_bars: 1`, `horizons: [1, 5, 15, 60]`, `boundary_policy: same_session`, `n_quantiles: 5`, `min_ic_samples: 3`, `apply_tradability_mask: true`.
- **Return unit & gap-rule notes**:
  - Per the Slot evaluation contract (`docs/TIME_SERIES_CONTRACT.md`), `per_bar_labels.csv` (`return_pct` column) and `quantile_stats.csv` report **decimal** returns (`exit_price / entry_price - 1`) matching `return_unit: decimal`, while raw execution prices remain in `entry_price` and `exit_price`.
  - `gap_rule: endpoints_only` checks completeness of the entry and exit bars; if an evaluation requires every intermediate bar in `[entry, exit]` to be present (matching `BarReturnLabeler` / `ExecutionConvention.OPEN_TO_CLOSE`), switch `gap_rule: require_complete_bars` and `return_unit: percent`.

## Quick Start

### 1. Harness CLI (recommended)

Outputs are written to disposable temporary directories so the repository `results/` directory stays untouched:

```sh
# ETH view evaluation
python3 -m tools.agent_harness.cli time-series-analysis \
  --config examples/nq_intraday_futures/eval_eth.json \
  --output-dir /tmp/nq_intraday_eth_eval \
  --output-root /tmp/nq_intraday_harness

# RTH view evaluation
python3 -m tools.agent_harness.cli time-series-analysis \
  --config examples/nq_intraday_futures/eval_rth.json \
  --output-dir /tmp/nq_intraday_rth_eval \
  --output-root /tmp/nq_intraday_harness
```

### 2. Minimal Python API (`build_slot_from_config` + `evaluate_slot`)

```python
import pandas as pd
from backtest_and_analysis.time_series import evaluate_slot
from slots import build_slot_from_config

slot_rth = build_slot_from_config("examples/nq_intraday_futures/slot_rth.yaml")
bars_rth = slot_rth.bar_feed.load(instruments=["NQZ26"])
factors = bars_rth[["timestamp_utc", "instrument_id"]].assign(
    factor_value=bars_rth["close"] - bars_rth["open"],
)
result = evaluate_slot(
    slot_rth,
    factors,
    frequency="1m",
    start_date="2026-12-01",
    end_date="2026-12-02",
    instruments=["NQZ26"],
    horizons=[1, 5, 15, 60],
    entry_offset_bars=1,
    analysis_mode="time_series",
    apply_tradability_mask=True,
)
print(result.ic_by_horizon.query("scope == 'per_instrument'"))
```

## Synthetic Fixture Scenarios & Hand-Computed Reference

`bars.csv` covers two sessions (`2026-12-01` and `2026-12-02`, all `open = 20000`, `factor_value = close - open`) and exercises three core scenarios:

1. **Valid multi-bar block (`2026-12-01T14:30:00Z..14:40:00Z`, 11 bars)**:
   - Shared across both `ETH` and `RTH` with strictly increasing `factor_value = 20, 40, ..., 220`.
   - With `entry_offset_bars=1, exit_adjustment=-1`, signal `t_i` enters at `t_{i+1}` `open=20000` and exits at `t_{i+h}` `close` (`t_{i+h} + 1m`).
2. **Near-close `same_session` boundary (`2026-12-01` RTH close `21:00Z` vs ETH close `22:00Z`, plus `2026-12-02` bars)**:
   - Around RTH close (`20:57Z..21:01Z`): `20:59Z` (`h=1`) and `20:57Z..20:59Z` (`h=5,15,60`) are excluded as `session_boundary` in `RTH` (even though `21:00Z, 21:01Z` and `2026-12-02T14:30Z+` exist in `bars.csv`), while `ETH` executes `20:57Z..21:00Z` (`h=1`) and `20:57Z..20:59Z` (`h=60`) cleanly across `21:00Z`.
   - Around ETH close (`21:57Z..21:59Z`): `21:59Z` (`h=1`) and `21:57Z..21:59Z` (`h=5,15,60`) are excluded as `session_boundary` in `ETH` instead of peeking into `2026-12-01T23:00Z` (`2026-12-02` ETH open).
3. **Missing bar (`2026-12-01T15:02:00Z` omitted inside RTH/ETH)**:
   - Grid slot `15:02Z` has no observed bar (`factor_value=NaN` -> `factor_missing`).
   - `15:01Z` (`h=1, 5`) and `15:00Z` (`h=5`, missing exit `15:05Z`) are excluded as `missing_bar` without shifting to `15:03Z`, while `15:00Z` (`h=1`) and `15:03Z` (`h=1`) align to `15:01Z` and `15:04Z` respectively.

### Selected Signal Rows (`horizon_bars = 1`, `entry_offset_bars = 1`, `exit_adjustment = -1`)

| Scenario | Signal Timestamp (UTC) | `factor_value` | ETH Entry -> Exit (UTC) | ETH Result / Reason | RTH Result / Reason |
| --- | --- | ---: | --- | --- | --- |
| Overnight ETH-only | `2026-12-01T04:00:00Z` | `10` | `04:01Z` -> `04:02Z` | `valid`, `20000 -> 20015` (`+0.00075`) | Not in RTH grid |
| Valid block #1 | `2026-12-01T14:30:00Z` | `20` | `14:31Z` -> `14:32Z` | `valid`, `20000 -> 20040` (`+0.0020`) | `valid`, `20000 -> 20040` (`+0.0020`) |
| Valid block #2 | `2026-12-01T14:31:00Z` | `40` | `14:32Z` -> `14:33Z` | `valid`, `20000 -> 20060` (`+0.0030`) | `valid`, `20000 -> 20060` (`+0.0030`) |
| Valid block #5 | `2026-12-01T14:34:00Z` | `100` | `14:35Z` -> `14:36Z` | `valid`, `20000 -> 20120` (`+0.0060`) | `valid`, `20000 -> 20120` (`+0.0060`) |
| Valid block #10 | `2026-12-01T14:39:00Z` | `200` | `14:40Z` -> `14:41Z` | `valid`, `20000 -> 20220` (`+0.0110`) | `valid`, `20000 -> 20220` (`+0.0110`) |
| Missing-bar pre-gap | `2026-12-01T15:01:00Z` | `250` | — | `excluded` (`missing_bar`, `15:02Z` absent) | `excluded` (`missing_bar`, `15:02Z` absent) |
| Missing-bar grid slot | `2026-12-01T15:02:00Z` | `NaN` | — | `excluded` (`factor_missing`) | `excluded` (`factor_missing`) |
| Post-gap recovery | `2026-12-01T15:03:00Z` | `260` | `15:04Z` -> `15:05Z` | `valid`, `20000 -> 20280` (`+0.0140`) | `valid`, `20000 -> 20280` (`+0.0140`) |
| Near RTH close | `2026-12-01T20:58:00Z` | `320` | `20:59Z` -> `21:00Z` | `valid`, `20000 -> 20340` (`+0.0170`) | `valid`, `20000 -> 20340` (`+0.0170`) |
| At RTH close | `2026-12-01T20:59:00Z` | `340` | `21:00Z` -> `21:01Z` | `valid`, `20000 -> 20360` (`+0.0180`) | `excluded` (`session_boundary`) |
| Near ETH close | `2026-12-01T21:58:00Z` | `420` | `21:59Z` -> `22:00Z` | `valid`, `20000 -> 20440` (`+0.0220`) | Not in RTH grid |
| At ETH close | `2026-12-01T21:59:00Z` | `440` | — | `excluded` (`session_boundary`) | Not in RTH grid |
| Next-session tail | `2026-12-02T15:35:00Z` | `540` | — | `excluded` (`insufficient_history`) | `excluded` (`insufficient_history`) |

### Selected Signal Rows (`horizon_bars = 5` & `60`)

- `horizon_bars = 5` (identical valid block in `ETH` & `RTH`): `2026-12-01T14:30:00Z..14:35:00Z` (6 valid rows, `entry = 20000`, `exit = 20120..20220`, `return_pct = 0.006, 0.007, 0.008, 0.009, 0.010, 0.011`); `15:00Z` is excluded as `missing_bar` (`15:05Z` absent); `20:57Z..20:59Z` are `session_boundary` in `RTH` while `21:57Z..21:59Z` are `session_boundary` in `ETH`.
- `horizon_bars = 60`: `RTH` has `0` valid rows (`20:57Z..20:59Z` all hit `session_boundary`); `ETH` has `3` valid rows (`20:57Z -> 20:58Z..21:58Z = +0.020`, `20:58Z -> 20:59Z..21:59Z = +0.021`, `20:59Z -> 21:00Z..22:00Z = +0.022`, `spearman_ic = 1.0`).

### Expected `ic_by_horizon` & `quantile_stats` Summary

| View | `horizon_bars` | `total_count` | `valid_count` | `session_boundary` | `missing_bar` | `insufficient_history` | `spearman_ic` | `ic_status` | `actual_quantiles` (`quantile_status`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `ETH` | `1` | `2760` | `21` | `1` | `7` | `1` | `1.0` | `ok` | `5` (`ok`) |
| `ETH` | `5` | `2760` | `6` | `3` | `20` | `1` | `1.0` | `ok` | `5` (`ok`) |
| `ETH` | `15` | `2760` | `0` | `3` | `26` | `1` | `NaN` | `insufficient_samples` | `0` (`insufficient_samples`) |
| `ETH` | `60` | `2760` | `3` | `5` | `21` | `1` | `1.0` | `ok` | `3` (`reduced_quantiles`) |
| `RTH` | `1` | `780` | `15` | `1` | `4` | `1` | `1.0` | `ok` | `5` (`ok`, 3 rows/bin) |
| `RTH` | `5` | `780` | `6` | `3` | `11` | `1` | `1.0` | `ok` | `5` (`ok`) |
| `RTH` | `15` | `780` | `0` | `3` | `17` | `1` | `NaN` | `insufficient_samples` | `0` (`insufficient_samples`) |
| `RTH` | `60` | `780` | `0` | `3` | `17` | `1` | `NaN` | `insufficient_samples` | `0` (`insufficient_samples`) |

For `RTH` `horizon_bars = 1`, the 5 quantile buckets each contain 3 rows with exact `(mean_return_pct, median_return_pct)`: Q1 `(0.003, 0.003)`, Q2 `(0.006, 0.006)`, Q3 `(0.009, 0.009)`, Q4 `(0.0125, 0.0125)`, Q5 `(0.059 / 3, 0.017)`.
