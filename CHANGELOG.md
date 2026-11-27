# Changelog

## v0.2.0

- Unified execution scheduling and causal gates: aligned `ExecutionScheduler` cadence, added a point-in-time replay fast path and declarative data-slot query pushdown, tightened Parquet as-of precision, and enforced strict causal fast-path gate diagnostics across the CLI and MCP server.
- Futures execution and synthesis-replay parity: added single-account futures contract execution with margin-rate liquidity caps, unified LASSO missing-sample and intercept scoring parity with bundle replay, enforced capacity parity guards, and kept regime-bounded weights within verified contracts.
- Market data normalization and validation: unified intraday turnover units and paused-bar turnover normalization, enforced DuckDB path market-identity validation in [the local workflow](docs/LOCAL_WORKFLOW.md) and [data import guide](docs/DATA_IMPORT.md), and added declarative slot `--dry-run` parity verification.
- Agent Harness and framework reliability: added content-level result assertions (`tools/agent_harness/result_content.py`), deterministic experiment identity hashing, caller-relative public CLI path resolution, and sharded public CI checks.

Try the offline demos from the [Quickstart](README.md#60-second-quickstart) or [share feedback](https://github.com/GSX-Eis/jq-factor-lab/issues/new/choose) with synthetic or sanitized examples.

## v0.1.0

- The unified evaluation engine supports multi-asset daily cross-sections and single-instrument minute/Bar time series. The Bar CLI emits `diagnostics_report` with sample counts, Spearman IC and reasons for undefined metrics, and can store and query evaluations in DuckDB.
- Trading calendars cover XSHG, CME and NYSE via pandas-market-calendars. The [calendar guide](docs/TRADING_CALENDAR.md) describes session alignment and verification.
- The `jq-factor-lab` CLI ships two offline synthetic demos for the daily and Bar workflows. Python 3.11 and 3.12 use a lean core install, with development and research dependencies available as extras.
- The [English Quickstart](README.md), [简体中文快速开始](README.zh-CN.md), and [AGENTS.md](AGENTS.md) / [CLAUDE.md](CLAUDE.md) show how to screen a factor, read structured metrics and state a reproducible screening decision.
- GitHub Actions runs the public test suite, builds and checks the source distribution and wheel, and smoke-tests the installed wheel on both supported Python versions.
- Local research uses the documented [CSV, Parquet and DuckDB contracts](docs/DATA_IMPORT.md). The [JoinQuant builder](docs/JOINQUANT.md) packages a strategy for offline use. Real market data, trained models, automated login or upload, and cloud or live trading validation are outside this release.

Both READMEs show the version from the public main branch. After the first PyPI release, switch the version badges to `img.shields.io/pypi/v/jq-factor-lab`.

The distributions include the Apache 2.0 license and attribution notice.

Try the offline demos from the [Quickstart](README.md#60-second-quickstart), then [share feedback](https://github.com/GSX-Eis/jq-factor-lab/issues/new/choose). The issue chooser has forms for bug reports, market calendar and data contract requests, and AI factor research showcases. Use synthetic or sanitized examples and keep real market data and credentials out of public issues.
