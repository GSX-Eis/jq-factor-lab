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
  Validate coverage against independent, serially correlated and overlapping
  synthetic nulls with a predetermined tolerance, plus known positive and
  negative effects. Report coverage failures and the tested settings. The
  frozen AR(1) simulation fell below tolerance at lag 16 and block length 17.
- Register every factor/horizon tried in one selection in `TrialLedger` before
  inspecting outcomes. Record the statistic, sample count, raw p value and
  available interval for each attempt. Keep missing p values in the ledger.
  Report a selected test only after every registered p value is available and
  adjusted. Use Holm for family-wise error control under arbitrary dependence,
  or BH for false discovery rate control under independence or suitable
  positive dependence. An interval alone supplies insufficient evidence for a
  raw p value. Report the complete family and the adjustment results.
- Require the applicable exposure protocol before opening real OOS data. Record
  data provenance and distinguish synthetic engineering acceptance from real
  OOS evidence.

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
