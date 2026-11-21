# Statistical gates

| CLI exit code | Meaning |
| --- | --- |
| 0 | Operation completed, including a requested preflight. Inspect the evidence before making a research claim. |
| 1 | Execution or dependency failure. Use `error.code`, `error.fix`, and stderr. |
| 2 | Invalid arguments or preconditions. Repair the inputs or plan and repeat the preflight. |

## Read the evidence in order

- **Point-in-time split.** Assign observations from signal availability. Purge training labels that exit after the first expected OOS signal is available. Use actual calendar-derived exit times and freeze fitted direction, scaling and bins on training only.
- **IC estimability.** Inspect counts, non-overlapping IC and `reason` in Bar diagnostics. A constant signal may yield `constant_factor`, constant returns may yield `constant_return`, and a short series may yield `insufficient_samples`. Preserve the nulls and report them as missing evidence.
- **Dependence-aware inference.** Cross-sectional mean IC can use Bartlett HAC with adaptive lag. Time-series Spearman IC can use paired Adaptive Bootstrap v2, which resamples blocks on the scheduled grid, preserves the missing-slot layout of each sampled block, and reranks each replicate. Holding horizons impose minimum lag or block length. Freeze inference settings and check the realized diagnostics.
- **Coverage calibration.** Look at `metrics.splits.<split>.horizons[].interval_usability` in a Bar result. A finite interval can have `calibration_status: uncalibrated` when matching coverage evidence is absent. For significance interpretation, require `calibrated_pass` and `significance_interpretation_usable: true` for the relevant window. Verify that the declared dependence scenario is justified. Adaptive v2 interval coverage does not supply a raw p value on its own.
- **Multiple testing.** Register all tested factor/horizon pairs before looking at outcomes. Each public CLI config evaluates one factor, so keep an experiment-wide family across CLI calls. Per-run `trial_ledger` records from Bar evaluations supply candidate evidence and must be retained, including missing p values. Require a complete family and adjusted p values for selection claims. Use Holm for family-wise error control or BH when its dependence assumptions are justified. Incomplete families support no significance ranking.
- **Selection and backtesting.** Record eligibility independently of process success. Compare frozen candidates on like-for-like windows. Keep cost and exposure assumptions explicit and keep evaluation on a genuine unseen holdout separate from development screening.

See the [evaluation policy](https://github.com/GSX-Eis/jq-factor-lab/blob/dev/docs/EVAL_POLICY.md) and [time-series contract](https://github.com/GSX-Eis/jq-factor-lab/blob/dev/docs/TIME_SERIES_CONTRACT.md) for the full contracts.
