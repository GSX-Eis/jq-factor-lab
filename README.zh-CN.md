<div align="center">

# JQ Factor Lab

**面向 AI Agent 的量化研究 Harness，通过 CLI 和 MCP 串联多因子计算与合成、回测和抗过拟合统计评测。**

[English](README.md) · 简体中文

[![CI](https://img.shields.io/github/actions/workflow/status/GSX-Eis/jq-factor-lab/ci.yml?branch=main&label=CI)](https://github.com/GSX-Eis/jq-factor-lab/actions/workflows/ci.yml?query=branch%3Amain)
[![版本](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2FGSX-Eis%2Fjq-factor-lab%2Fmain%2Fpyproject.toml&query=%24.project.version&label=version&prefix=v&color=blue)](CHANGELOG.md)
![Python 3.11–3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
[![Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

</div>

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/architecture-zh-dark.svg">
    <img src="docs/assets/architecture-zh-light.svg" width="100%" alt="JQ Factor Lab 架构图，从 AI Agent 经 CLI 契约和量化核心到本地证据与只读看板">
  </picture>
</div>

## 适用场景

JQ Factor Lab 在单机上使用 DuckDB 完成可追溯的研究流程。日频截面和分钟 Bar 时序共用一套评测引擎。Agent 可以先预检命令，再解析严格 JSON，并将指标追溯到产物和统计门禁。

| 项目 | 主要用途 | JQ Factor Lab 的切入点 |
| --- | --- | --- |
| [Qlib](https://github.com/microsoft/qlib) | 机器学习驱动的量化投研 | 单机 DuckDB 上按统一契约评测日频截面和分钟 Bar |
| [RD-Agent](https://github.com/microsoft/RD-Agent) | Agent 驱动的研发流程 | 输入与输出契约明确的执行入口，提供 JSON、预检计划和退出码 |
| [VectorBT](https://github.com/polakowo/vectorbt) | 向量化组合回测 | 使用 Adaptive HAC、Bootstrap v2、TrialLedger 和 `calibration_status` 门禁评测因子，可接入 VectorBT 做策略回测 |
| [TradingAgents](https://github.com/TauricResearch/TradingAgents) | 多 Agent 交易分析与决策 | 按时点约束、净化切分和可追溯证据复现量化因子实验 |

### 内置防泄漏与执行护栏

- **自动因果门禁（Causal Gate）**：运行 `single-factor-analysis`、`time-series-analysis` 或独立 `verify-factor-causality` 时，自动执行时间截断与 PIT 抽样校验，拦截含未来函数的快路径因子。
- **统一执行时钟（`ExecutionScheduler`）**：单因子评测、多因子合成与组合回测共享同一套信号可用时刻到成交 Bar 的映射契约。
- **抗过拟合统计门禁**：支持 Adaptive HAC、配对 Moving-Block Bootstrap v2、`calibration_status` 适用性校准门禁与 `TrialLedger`（Holm / BH）多重检验追踪（详见 [评测引擎](docs/EVALUATION_ENGINE.md) 与 [评测政策](docs/EVAL_POLICY.md)）。
- **合约级组合执行**：VectorBT 适配器除股票截面外，还支持单账户期货合约执行（Session 边界换月、保证金率上限与整手取整）。

## 60 秒快速开始

使用 macOS 或 Linux 上的 Python 3.11–3.12。首次下载依赖可能需要更长时间。克隆仓库后，选择 pip 或 uv。

```bash
git clone https://github.com/GSX-Eis/jq-factor-lab.git
cd jq-factor-lab
```

**pip**

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install '.[synthesis,viz]'
```

**uv**

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install '.[synthesis,viz]'
```

PyPI 上架后，可改用 `python3 -m pip install 'jq-factor-lab[synthesis,viz]'` 或 `uv pip install 'jq-factor-lab[synthesis,viz]'`。基础包提供日频和 Bar 评测，`[synthesis]` 提供合成方法依赖，`[viz]` 提供看板，`[mcp]` 提供 stdio 服务，其余按需研究扩展见[端到端研究能力与按需安装](#端到端研究能力与按需安装)。`[all]` 会安装 Qlib 和 Torch 在内的全部扩展依赖，取决于平台 wheel 的可用性。

在可写目录运行以下命令。全部演示均使用合成数据，每次创建独立运行目录。

```bash
jq-factor-lab daily-demo > daily.json
jq-factor-lab bar-demo > bar.json
jq-factor-lab factor-synthesis-demo --method weighted_sum > synthesis.json
db="$(python3 -c 'import json; print(json.load(open("bar.json"))["metrics"]["results_db"]["path"])')"
jq-factor-lab read-bar-evaluations --results-db "$db" > bar-read.json
jq-factor-lab dashboard --results-db "$db" --dry-run
# 去掉 --dry-run 即可在 http://127.0.0.1:8501 打开只读看板。
```

以下是合成 Bar 演示的 1-Bar 周期诊断摘录。常收益使 Spearman IC 无法估计。

```json
{
  "ok": true,
  "exit_code": 0,
  "metrics": {
    "diagnostics_report": {
      "schema": "time_series_diagnostics.v1",
      "horizons": [
        {"horizon_bars": 1, "ic": {"sample_count": 39, "spearman_ic": null, "reason": "constant_return"}}
      ]
    }
  },
  "next_actions": ["Pass metrics.results_db.path to read-bar-evaluations --results-db."]
}
```

[AI 研究战报](docs/AI_WORKFLOW_SHOWCASE.md) 展示多个候选如何经历预检、校准与试验门禁、合成、示例回测及 DuckDB 回查。

如需直接体验单合约 `1min` 期货（`NQZ26` 搭配 `CMEGlobex_Equity` `ETH` / `RTH` 双视图）开箱样例，参见 [NQ 1min ETH/RTH 示例](examples/nq_intraday_futures/README.md)：

```bash
python3 -m tools.agent_harness.cli time-series-analysis --config examples/nq_intraday_futures/eval_eth.json --output-dir work/nq-eth-eval
python3 -m tools.agent_harness.cli time-series-analysis --config examples/nq_intraday_futures/eval_rth.json --output-dir work/nq-rth-eval
```

## 接入你的 AI Agent

### Agent Skill 或 Claude Code 插件

将完整的 [Factor Research Skill](skills/factor-research/SKILL.md) 目录连同参考文件复制到客户端的 skills 目录。使用 Claude Code 时，可在源码根目录执行 `claude --plugin-dir .` 加载[插件清单](.claude-plugin/plugin.json)，再调用 `/jq-factor-lab:factor-research`。项目约定见 [AGENTS.md](AGENTS.md) 和 [CLAUDE.md](CLAUDE.md)。

### Stdio MCP 服务

确保 MCP 客户端的 PATH 可找到 `uvx`。包上架 PyPI 后，将以下配置加入 Claude Desktop、Cursor、Claude Code 或 Windsurf 的 `mcpServers`。使用本地源码时，把 `--from` 的值改成源码目录的绝对路径并追加 `[mcp,synthesis]`。合成工具 `run_factor_synthesis_demo` 依赖 `[synthesis]`。

```json
{
  "mcpServers": {
    "jq-factor-lab": {
      "command": "uvx",
      "args": ["--python", "3.12", "--from", "jq-factor-lab[mcp,synthesis]", "jq-factor-lab-mcp"]
    }
  }
}
```

服务在本机运行。工具输入使用绝对路径。参见 [CLI 与 MCP 映射](skills/factor-research/references/cli_contract.md)。

### 可直接复制的研究提示词

```text
使用 Factor Research Skill，在新建的可写目录运行 AI 研究战报。
先锁定假设、候选、切分和筛选规则，只使用合成数据。逐项预检，
分开记录 stdout 和 stderr，并报告 IC、样本量、缺失原因、校准状态
和完整试验家族。仅将符合合成回测条件的候选纳入组合。核对 DuckDB
和看板启动计划，列出未通过的候选、产物路径和能否声称统计显著。
```

## 多市场数据与公开基准库

### 声明式多市场接入（`slot.yaml`）

无需编写 Python 适配器，通过一份版本化的 `slot.yaml` 即可接入本地 [CSV、Parquet 或 DuckDB](docs/DATA_IMPORT.md) 行情，同时也内置主流市场原生适配器：

| 市场类型 | 支持范围 | 接入方式 | 参考 |
| --- | --- | --- | --- |
| **A 股（日频 / 分钟）** | XSHG / XSHE，含 PIT 复权与微观结构规则 | 原生 `jq_ashare_local` 或 CSV / DuckDB | [本地流程](docs/LOCAL_WORKFLOW.md) |
| **美股 / 全球指数** | XNYS / XNAS，支持 PMC 核验日历与夏令时 | 声明式 `slot.yaml` + MarketDB Profile | [市场库指南](docs/MARKET_DATABASE_GUIDE.md) |
| **国内外期货** | CME / SHFE / INE / DCE / CZCE / CFFEX / GFEX（ETH/RTH 与 Session 换月） | 原生 `cn_futures` / `global_futures` 或声明式 `slot.yaml` | [NQ 1min ETH/RTH 示例](examples/nq_intraday_futures/README.md) · [交易日历](docs/TRADING_CALENDAR.md) |
| **Crypto 24/7 及自定义市场** | 任意时区、24/7 会话及资金费率等辅助特征 | 零代码声明式 `slot.yaml` | [Crypto 示例](examples/declarative_crypto/README.md) · [自定义市场与部署指南](docs/CUSTOM_MARKET_AND_DEPLOY_GUIDE.md) |

### 一键构建公开基准市场库（`[markets]`）

安装 `.[markets]` 即可直接抓取、SHA-256 冻结并重建三套公开基准 DuckDB，并附带版本化 `MarketDB` Profile（强制单库隔离，防止跨市场混算）：

- **Kenneth R. French**：49 行业组合 + FF3 日频序列（1963 年起）
- **Binance Vision**：10 个 USDT 现货交易对 `1min` 及派生 `1d` Bar
- **AKShare**：A 股宽基指数与申万一级行业日频指数

```bash
# 示例：抓取并重建 Binance 现货基准库（French / AKShare 重建命令见文档）
python3 -m data_pipeline.extractors.public_markets.binance \
  --work-dir work/binance --db work/binance/market.duckdb
python3 -m tools.agent_harness.cli market-db show --db work/binance/market.duckdb
```

完整重建步骤、快照校验与上游数据条款见 [公开数据市场库](docs/PUBLIC_MARKET_DATABASES.md)。

### 规模化 Bar 与因子持久化存储

当研究从单次 CSV 扩展到多因子、多频率规模时，可通过 `import-bars`、`build-bars`、`audit-bars`、`build-roll-schedule` 将行情与期货换月表写入受 Profile 校验的 DuckDB；大规模因子值可通过 `MarketFactorValueStore`（Hive 分区 Parquet + 只追加 JSONL 账本 + DuckDB 只读视图）进行跨频率写入、身份哈希回查与完整性核验（`write-factor-values` / `read-factor-values` / `verify-factor-values`）。详见 [时序与存储契约](docs/TIME_SERIES_CONTRACT.md) 与 [因子接口](docs/FACTOR_INTERFACE.md)。

## 端到端研究能力与按需安装

核心包保持精简，按需安装对应扩展即可启用完整研产链路（`python3 -m pip install 'jq-factor-lab[<extra>]'`）：

| 研究阶段 | 能力说明 | 可选 Extra | 文档 / 入口 |
| --- | --- | --- | --- |
| **单因子评测与诊断** | 日频截面 + 分钟 Bar 时序/截面评测、因果门禁与 DuckDB 存证 | 核心自带 | [评测引擎](docs/EVALUATION_ENGINE.md) · [评测政策](docs/EVAL_POLICY.md) |
| **符号因子挖掘** | 基于 PyTorch DSO（策略网络 + 结构先验）、DEAP GP 融合与 QuantaAlpha 提案结构门禁自动搜索并编译为独立 `FactorBase` 因子 | `[mining]` | [run_symbolic_factor_mining.py](run_symbolic_factor_mining.py) |
| **多因子合成与组合回测** | LASSO / ElasticNet / PCA / LightGBM / 动态混合 / 市值条件合成，衔接 VectorBT（含股票与单账户期货合约执行） | `[synthesis,backtest]` | [合成指南](docs/SYNTHESIS.md) |
| **策略打包与跨市场导出** | 将合成结果固化为不可变 `StrategyBundle`（含 SHA-256 清单），导出跨市场目标仓位与 `trade_orders.jsonl`（`signal_export`）或聚宽离线单文件（`jq_cloud`）并离线核验对齐 | 核心 / `[synthesis]` | [自定义市场与部署指南](docs/CUSTOM_MARKET_AND_DEPLOY_GUIDE.md) · [聚宽单文件打包](docs/JOINQUANT.md) |
| **公开基准数据集** | French / Binance Vision / AKShare 快照抓取与重建 | `[markets]` | [公开数据市场库](docs/PUBLIC_MARKET_DATABASES.md) |
| **可视化与 Agent 集成** | 本地只读 Streamlit 看板与 Stdio MCP 服务 | `[viz,mcp]` | [CLI 与 MCP 映射](skills/factor-research/references/cli_contract.md) |

> 其他按需扩展依赖：`[qlib]`（Qlib 适配器）、`[acceleration]`（Numba 加速）、`[dev]`（测试套件）、`[all]`（全量安装）。

```bash
# 从 StrategyBundle 导出并核验跨市场目标仓位与交易指令
python3 -m tools.agent_harness.cli export-strategy --target signal_export \
  --bundle-dir "$BUNDLE_DIR" --output-dir work/signal-candidate
python3 -m tools.agent_harness.cli verify-target-alignment --target signal_export \
  --bundle-dir "$BUNDLE_DIR" --artifact-dir work/signal-candidate
```

## CLI 入口一览

项目采用**双层 CLI 设计**，所有命令均输出严格 JSON 并支持 `--dry-run` 预检：

- **公开 Agent CLI（`jq-factor-lab`）**：面向日常研究与 MCP 的 7 个高频命令——演示（`daily-demo`、`bar-demo`、`factor-synthesis-demo`）、本地数据评测（`single-factor-analysis`、`time-series-analysis`）、DuckDB 回查（`read-bar-evaluations`）与看板（`dashboard`），支持 `--compact` 单行 JSON 输出。
- **完整研究后端（`python3 -m tools.agent_harness.cli`）**：面向外部编排器与多市场管线的 20 个子命令，同时提供三个独立主脚本（[run_analysis_main.py](run_analysis_main.py)、[run_factor_synthesis.py](run_factor_synthesis.py)、[run_symbolic_factor_mining.py](run_symbolic_factor_mining.py)）。详见 [公开范围](docs/PUBLIC_SCOPE.md)。

<details>
<summary>展开查看完整 Harness 后端 20 个子命令速查</summary>

| 分组 | 子命令 |
| --- | --- |
| **市场库与行情治理** | `market-db`（`init` / `adopt` / `show` / `migrate-profile`）、`import-bars`、`build-bars`、`audit-bars`、`build-roll-schedule` |
| **因子值存储** | `write-factor-values`、`read-factor-values`、`verify-factor-values` |
| **评测、因果门禁与回查** | `single-factor-analysis`、`time-series-analysis`、`verify-factor-causality`、`read-bar-evaluations` |
| **符号挖掘与筛选** | `symbolic-factor-mining`、`quanta-alpha-realize`、`parent-factor-screen` |
| **合成与部署对齐** | `factor-synthesis`、`export-strategy`、`verify-target-alignment`、`alignment-run` |
| **监控与运维** | `monitor-refresh`、`outputs-gc` |

</details>

## 试用与反馈

欢迎[提交 Issue](https://github.com/GSX-Eis/jq-factor-lab/issues/new/choose)报告问题、请求市场日历或数据契约，或分享 AI 工作流。请使用合成或脱敏样例。参见[贡献指南](CONTRIBUTING.md)、[安全说明](SECURITY.md)及[变更日志](CHANGELOG.md)。

采用 [Apache License 2.0](LICENSE)。归属信息见 [NOTICE](NOTICE)。
