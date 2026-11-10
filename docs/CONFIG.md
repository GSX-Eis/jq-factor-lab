# single-factor-analysis 配置参考

命令：

```bash
python -m tools.agent_harness.cli single-factor-analysis --config config/demo.json --output-dir outputs/my-analysis
```

配置采用 UTF-8 JSON 对象。相对 `--config` 路径从当前工作目录解析；配置内所有路径相对该配置文件所在目录解析。绝对路径保持其含义。

## 必填键

| 键 | 类型与含义 |
| --- | --- |
| `input_csv` | 非空字符串，日频行情 CSV；必含 date、stock_code、close |
| `factor_file` | 非空字符串，可信 Python 因子文件路径 |
| `factor_class` | 非空字符串，精确选择一个因子类 |
| `output_dir` | 非空字符串，新输出目录；提供 CLI `--output-dir` 时可省略，配置中的该值不再参与校验 |
| `calendar_file` 或 `calendar_duckdb_path` | 必须且只能设置其中一个有效路径。前者为含 sessions、start_date、end_date、calendar_id 的日历 JSON；后者为已导入的交易日历 DuckDB |

## 可选键与默认值

| 键 | 默认值 | 校验 |
| --- | --- | --- |
| `periods` | `[1, 5, 10]` | 非空列表，每项必须为正整数，布尔值不接受 |
| `execution_lag` | `1` | 非负整数，布尔值不接受 |
| `n_quantiles` | `5` | 至少为 2 的整数，布尔值不接受 |

未列出的键不会被此 CLI 使用，也没有严格未知键拒绝；请检查拼写。此配置不接受研究调度器的任意训练参数。

CLI `--output-dir` 优先于配置 `output_dir`，并相对**当前工作目录**解析；配置 `output_dir` 相对**配置文件目录**解析。已有输出目录会被拒绝，即使为空。覆盖参数可以替代缺失或无效的配置 `output_dir`，不会覆盖现有目录。

## 校验与执行顺序

1. 解析 CLI 参数，读取配置文件，解析 JSON 并要求顶层为对象。
2. 按 input_csv、factor_file、factor_class、output_dir 顺序检查非空字符串；有 CLI 输出参数时跳过 output_dir 的配置检查。
3. 依次检查 periods、execution_lag、n_quantiles 的类型和值。
4. 解析最终输出目录并拒绝已存在路径；该步骤早于日历和输入文件读取。
5. 要求恰好一个日历来源，加载并验证日历（行情自身不能定义交易日）。示例日历只是虚构市场。
6. 读取行情，依次校验必填列、非空、日期（无空值、无时区、午夜）、股票键、重复键以及 close/factor 为有限正数。缺省 factor=1。随后验证日历覆盖。
7. 加载可信因子文件并精确选择一个类。在内存 DuckDB 中计算因子，然后评估前向收益 `price[t + lag + period] / price[t + lag] - 1`。
8. 要求 IC 结果可用，最后创建输出目录并写入结果。失败不意味着云端或交易验收成功；发生写入中断时应保留目录检查，不能当作可覆盖的成功输出。

常见配置/文件/计算错误由 CLI 输出 `Analysis failed: ...`，非零退出；JSON 成功摘要在 stdout。构建器和版本化读取示例的错误合同分别见 [聚宽构建](JOINQUANT.md) 和 [版本化读取](CANONICAL_DATA_READS.md)。
