# Evaluation Policy

## Unified Evaluation Engine

- Use the signal availability clock to assign train and OOS samples. For each
  horizon, purge training labels whose actual exit is later than the first
  expected OOS signal's availability. Keep exits at that boundary. Map daily
  exits to the scheduled trading-session close and Bar exits to the scheduled
  Bar end, accounting for early closes. Keep each horizon's eligibility and
  purge counts, including scheduled crossings whose exit prices are missing.
- Use `per_cross_section` with the signal-time universe and finite available
  factors. Missing future labels affect valid-pair counts. Use
  `fitted_on_train` for a time-series split. Freeze edges, direction and any
  standardization from training only and report the same `fit_identity` for
  train and OOS. Full-sample time-series bins are descriptive.
- Freeze the confidence level, time grid and inference settings before viewing
  outcomes. For cross-sectional mean IC use Bartlett HAC on the intact grid,
  with lag at least `h - 1`. For time-series Spearman IC use paired moving
  blocks of length at least `h`, preserving missing slots and reranking
  each replicate. Record the seed, replicate count and minimum valid count.
  The opt-in `ts_boot_adaptive_v3_p` reports a diagnostic two-sided P value
  from the same studentized pivots. It requires at least 9,999 requested
  replicates, 90% valid replicates and a matching preregistered single-asset
  design. Validate raw P-value size separately at 0.05, 0.01 and 0.0025 before
  using this method for candidate selection.
  NERE-1279 combines Monte Carlo uncertainty from the warp-speed cutoff
  and evaluation draws. The 162 null threshold rows form one Holm family
  at 1%. Require each one-sided 99% size upper bound to be at most 1.5α,
  size pooled within each (T, α) to be at most 1.1α, and directional
  power of at least 80% for both effect signs. Require the preregistered
  production P-value equivalence spot check and apply only preregistered
  scope reductions.
  Validate coverage against independent, serially correlated and overlapping
  synthetic nulls with a predetermined tolerance, plus known positive and
  negative effects. Report coverage failures and the tested settings. The
  frozen AR(1) simulation fell below tolerance at lag 16 and block length 17.
- Default interval calibration to `uncalibrated`. Resolve registered evidence
  only for an explicitly declared dependence scenario and matching method,
  effective window, expected grid length and nominal coverage. A significance
  claim requires an estimable interval and applicable evidence for
  `calibrated_pass`, plus the research and multiple-testing gates. The frozen
  coverage evidence requires a complete grid of ICs for HAC and a complete
  grid of factor-return pairs with 399 requested and valid replicates for
  moving-block Bootstrap. Report finite descriptive statistics independently.
  Treat legacy daily HAC t and p values as uncalibrated diagnostics.
- Register every factor/horizon tried in one selection in `TrialLedger` before
  inspecting outcomes. Record the statistic, sample count, raw p value and
  available interval for each attempt. Keep missing p values in the ledger.
  Report a selected test only after every registered p value is available and
  adjusted. Use Holm for family-wise error control under arbitrary dependence,
  or BH for false discovery rate control under independence or suitable
  positive dependence. Raw P values require calibration applicable to the
  method, design and selection threshold. The opt-in time-series v3 P value
  stays diagnostic in its existing single-asset domain. The NERE-1279
  confirmation failed the hard upper bound for the Gaussian independent
  null at T=512, h=20 and α=0.0025, outside the registered fallback.
  Report the complete family and the adjustment results.
- Require the applicable exposure protocol before opening real OOS data. Record
  data provenance and distinguish synthetic engineering acceptance from real
  OOS evidence.
- For the NERE-1280 cross-sectional HAC study, freeze exact N, T, horizon
  and alpha levels before simulation. Require each null cell's one-sided
  99% exact binomial upper bound to be at most 1.5α. Apply Holm at 1% to
  the two-sided nominal-size tests over the full preregistered null family.
  Require the pooled null rejection rate for each (N, T, α) to be at most
  1.1α and each signed control's power to reach 80%. Apply only the
  preregistered narrowing rule and retain the original Holm results.
  The resulting evidence covers complete constant-width Spearman grids at
  T=1024, N in {256, 1024}, h in {1, 20}, and α in {0.01, 0.05}, in the three
  registered Gaussian null scenarios. For these new domains, the audit retains
  the published interval and its raw analytic P value. For selection, compute
  the preregistered 95% interval's P value and round it up to the smallest
  supported threshold, or use 1 if it exceeds 0.05. A 95% or 99% interval may
  be paired with either supported family alpha. Earlier registrations retain
  their gates.
  See local archive `reports/NERE-1280/cs_hac_calibration.md` and the
  package-local calibration manifests
  ([cs_hac_adaptive_v1_coverage.json](../backtest_and_analysis/evaluation/calibration_data/cs_hac_adaptive_v1_coverage.json),
  [ts_bootstrap_v2_coverage.json](../backtest_and_analysis/evaluation/calibration_data/ts_bootstrap_v2_coverage.json),
  [cs_hac_adaptive_v1_size.json](../backtest_and_analysis/evaluation/calibration_data/cs_hac_adaptive_v1_size.json)).

See [Unified Evaluation Engine](EVALUATION_ENGINE.md) for formulas, calibration
results and fixture links.

## Exit Code Semantics
- `0`
  Success or dry-run success
- `2`
  Precondition failure such as missing database, config, or alignment evidence
- `3`
  Execution failure such as subprocess non-zero or timeout
- `4`
  Quality gate failure

## Required JSON Fields
Every wrapper emits:
- `ok`
- `command`
- `summary`
- `artifacts`
- `metrics`
- `warnings`
- `exit_code`

Failures also emit:
- `error.code`
- `error.message`
- `error.details`
