# JQ Factor Lab — 聚宽与本地多因子研究框架

连接聚宽平台与本地多因子研究的工具框架。提供因子接口、DuckDB 数据访问、时点复权、前瞻收益计算、Alphalens 单因子评估，以及通用算法合成与因子预筛选。示例仅使用程序生成的合成行情。

## 快速开始

支持 Python 3.11–3.12、macOS / Linux。部分缓存锁使用 POSIX `fcntl`，暂不支持原生 Windows。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,synthesis]'
python -m examples.generate_data --output data/demo.csv
python -m tools.agent_harness.cli single-factor-analysis --config config/demo.json
python -m tools.agent_harness.cli factor-synthesis-demo --method weighted_sum --output-dir outputs/synthesis-weighted
python -m pytest -q
```

安装后也可使用 `jq-factor-lab` 命令调用相同 CLI；`python -m tools.agent_harness.cli` 仍然适用。

结果位于 `outputs/demo/`：`summary.json`、因子值、IC、IR 和分层收益 CSV。这些运行结果不会进入版本控制。重复运行不会覆盖已有结果目录，请指定新的 `--output-dir`。

## 本地数据导入

已有本地 CSV 可导入 DuckDB，入口和字段约定见 [数据导入](docs/DATA_IMPORT.md)。本地导入无需聚宽账号。批量下载器和认证工具不包含在本版中；聚宽运行时取数见下节。

## 聚宽运行与打包

提供行情、估值、因子库、资金流和行业 adapter，以及通用单文件构建器。生成示例：

```bash
python -m scripts.build_jq_strategy \
  --factor-files examples/momentum.py \
  --requirements config/factor_requirements.json \
  --cloud-api-mapping config/cloud_api_mapping.json \
  --single-file --out outputs/demo_jq_strategy.py
```

打包无需聚宽账号，也不会请求行情。生成文件需在聚宽环境运行，或注入模拟 API 做本地测试。示例入口只计算因子并记录日志。配置、依赖与能力边界见 [聚宽运行工具链](docs/JOINQUANT.md)。

## 数据与扩展

输入 CSV 每行是一只股票在一个交易日的记录，必填列为 `date, stock_code, close`；可提供正数复权字段 `factor`，缺省为 1。股票代码只接受字母、数字、点、下划线和短横线。日期与股票代码组合必须唯一，收盘价必须为有限正数。

继承 `utils.factor_base.FactorBase`，实现 `_get_raw_data_for_factor()` 和 `calculate_factor()`，即可接入自己的因子。`examples/momentum.py` 展示教学用的五期收益率。它不来自任何研究候选，也不代表策略收益承诺。只加载你信任的 Python 因子代码。

`config/demo.json` 中的路径相对配置文件解析（CLI 的 `--output-dir` 相对当前工作目录）；必填键、默认值、优先级和校验顺序见 [配置参考](docs/CONFIG.md)。公开 CLI 将 CSV 放入内存 DuckDB，并调用共享的因子基类与评估模块。底层库另提供 DuckDB 和版本化数据存储接口。

## 使用 agent

[AGENTS.md](AGENTS.md) 是统一入口；[CLAUDE.md](CLAUDE.md) 引用同一份索引。参考文档覆盖 [因子接口](docs/FACTOR_INTERFACE.md)、[无公式最小模板与合成测试](docs/FACTOR_TEMPLATE.md)、[数据语义](docs/SEMANTICS.md)、[本地导入](docs/DATA_IMPORT.md)、[算法合成](docs/SYNTHESIS.md) 和 [聚宽打包](docs/JOINQUANT.md)。这些说明可以直接作为 agent 上下文。

## 目录

- `backtest_and_analysis/`：因子加载、评估、收益计算和结果写入。
- `data_pipeline/loaders/`：本地行情、因子、辅助数据、指数成分和财务报表导入，不含在线下载。
- `factor_synthesis/`：合成算法、预筛选、预处理、交叉验证、诊断和权重接口。
- `scripts/build_jq_strategy.py`、`scripts/build/`：明确输入的本地单文件构建器。
- `utils/`：数据、缓存、股票池过滤机制及因子运行接口。
- `tools/agent_harness/`：独立的公开版 CLI。
- `examples/`、`config/`：合成数据生成器、教学因子和示例配置。
- `tests/`：使用合成输入与临时目录的回归测试。
- `docs/`：数据语义、公开范围与贡献方式。

## 当前边界

此版本包含通用合成算法和训练/选模代码，不包含原有实验配置、研究结果、原有因子库、专项研究脚本、真实数据、预训练模型、生产策略、自动上传工具或 dashboard。原根目录研究调度脚本不在本版中。

公开 CLI 可运行合成数据固定权重示例；其余六种方法需要生产方核验的标签退出上下文，见 [合成示例边界](docs/SYNTHESIS.md)。聚宽 adapter 和构建器已接入；真实云端运行与完整交易回测尚未验收。算法接口与依赖说明见 [合成算法](docs/SYNTHESIS.md)。

## 许可证

本项目采用 [Apache License 2.0](LICENSE)，版权声明见 [NOTICE](NOTICE)。第三方依赖保留各自的许可证。

贡献方式见 [CONTRIBUTING](CONTRIBUTING.md)，漏洞报告见 [SECURITY](SECURITY.md)，版本记录见 [CHANGELOG](CHANGELOG.md)。

完整的“CSV → DuckDB → 自定义因子 → 固定权重合成 → 价格收益标签 → 聚宽文件”示例见 [LOCAL_WORKFLOW](docs/LOCAL_WORKFLOW.md)，可运行 `python -m examples.local_factor_workflow --output-dir outputs/local-workflow`。

本地评估必须显式选择日历：`calendar_duckdb_path` 读取已导入的真实日历库，或 `calendar_file` 提供明确的日期列表与覆盖区间。`config/demo_calendar.json` 仅是虚构示例市场，不能用于真实 A 股研究。数据导入和覆盖边界见 [交易日历](docs/TRADING_CALENDAR.md)。
