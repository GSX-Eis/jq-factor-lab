# 公开范围

保留：本地 CSV→DuckDB 导入、单因子评估、因子扩展接口、DuckDB 存储、PIT 价格调整、显式股票池过滤机制、缓存和结果写入。

合成部分保留：LASSO、Ridge、ElasticNet、Quantile ElasticNet、PCA、LightGBM、固定权重、动态混合、市值条件合成；因子预筛选、特征与目标预处理、样本加权、交叉验证、诊断和组合权重接口。详见 [合成算法](SYNTHESIS.md)。

不包含：已有研究因子、个人合成实验配置、研究结果和已训练模型、专项实验脚本、真实行情、生产策略、自动登录上传及 dashboard。合成目录中绑定具体研究项目的脚本也不导出。

示例因子、合成数据和演示参数是公开版教学材料；保留测试中的小型合成输入与测试参数用于验证算法，不是个人实验配置。个人研究配置排除；公开 config 包含教学示例和需求/字段映射，以及通用 data_access.json、stock_pool.json、trading_calendar.json、calendar_overrides.json。

教学示例的执行范围为单因子 CLI 和固定权重合成。合成示例使用同日人工响应，其余六种方法缺少严格隔离所需的真实退出上下文，会在创建输出和进入 pipeline 前抛出 `LabelIsolationError`；不为该响应伪造下一交易日退出。所保留算法的回归测试与导出边界测试另行验收，不能把算法接口仍然存在解释为这六种教学调用已通过。Qlib 原生运行时、MPS 加速、真实数据接入、完整交易回测和云端执行没有在本版本中验收。

## 研究后端

公开版同时包含外部编排器（例如 Task Manager）调用的研究后端：

- `tools/agent_harness/cli.py`：结构化 JSON 包装，覆盖 `single-factor-analysis`、`time-series-analysis`、`factor-synthesis`、`symbolic-factor-mining`、`quanta-alpha-realize`、`parent-factor-screen`、`alignment-run`、`monitor-refresh` 和 `outputs-gc`。教学 CLI 位于 `tools/demo_cli.py`。
- `run_analysis_main.py`、`run_factor_synthesis.py`、`run_symbolic_factor_mining.py` 及其依赖的 `backtest_and_analysis`、`factor_synthesis`（含 VectorBT 适配）、`factor_mining/symbolic`。
- `scripts/alignment/` 的本地对齐与证据校验工具，以及其余审阅过的 `data_pipeline/loaders`。`alignment-run --factor` 使用 `scripts.build_jq_strategy` 打包，并以 `config/factor_requirements.json` 执行本地兼容性预检。独立的 `prepare_cloud_dumper` 工具需要通过 `--source` 传入自备的云端因子生产脚本。 源文件须在模块顶层或 `main()` 函数体中直接用字符串字面量各赋值一次 `start_date` 和 `end_date`。指定 `--target-date` 时，生成文件的两个日期均设为该日。
- 后端的默认门槛和参数按源码原样公开。

后端需要调用者自备 DuckDB 数据库、因子文件和运行配置；数据库默认路径为 `./data/duckdb/`。VectorBT 回测与 DSO 符号挖掘分别需要可选依赖 `backtest` 与 `mining`。对齐流程的浏览器上传、下载辅助脚本不在本仓库中。Regime 研究命令不公开。

数据导入部分见 [本地数据导入](DATA_IMPORT.md)。只保留经过审阅的 loaders；批量下载器、登录、JQData 本地 SDK 和旧在线复权类不包含在公开版本中。

## Harness 输出布局

`time-series-analysis` 默认在 `outputs/agent_harness/time-series-analysis/<run-id>/attempts/<execution-id>/` 保存每次调用。`--output-root` 改变根目录，重复 run-id 仍生成独立 attempt。attempt 的 `summary.json` 是包装器摘要，run 目录的同名文件是最近一次调用的导航。

Bar 评估默认写入 attempt 下的 `evaluation/`：`per_bar_labels.csv`、`ic_by_horizon.csv`、`quantile_stats.csv` 与 `summary.json`。评估的 `summary.json` 最后原子写入，作为四文件完整发布的完成标记；失败时旧 CSV 不能单独作为成功证据。`--output-dir` 或配置中的 `output_dir` 可指定评估目录，但不能覆盖包装器摘要。具体路径规则、哈希和失败契约见 [分钟研究契约](TIME_SERIES_CONTRACT.md)。教学 CLI 的独立布局见 [配置参考](CONFIG.md)。

公开控制台 `jq-factor-lab time-series-analysis` 和 `jq-factor-lab read-bar-evaluations` 以进程的当前工作目录解析相对路径，包括配置文件及其中的 Bar 输入、输出目录和 `results_db`。写库后可用相同的相对路径回查。使用 `--output-root` 指定 Harness 输出、临时文件和缓存的根目录，默认使用当前工作目录下的 `outputs/agent_harness`。显式评估目录已存在时需要 `--overwrite`。

分钟评估从代码根目录自己的 Git 元数据读取版本。无 Git 元数据时，依次使用 `QUANTATION_CODE_VERSION`、已安装的 `jq-factor-lab` 分发包版本或 `unknown`。通过 `result_identity.code_version_source` 核对所用来源。

聚宽运行工具链保留 JQFactorAdapter 和单文件构建器，以及共享的因子执行、价格 PIT 和信号计算模块。运行时可以调用聚宽 API；本地打包无需联网。基础映射仅使用原始公开字段名，不附带个人资金流衍生参数。详见 [聚宽运行](JOINQUANT.md)。
