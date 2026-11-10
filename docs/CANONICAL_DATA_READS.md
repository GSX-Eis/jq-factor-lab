# 显式版本读取的最小验证

已有版本化输入/输出库时，可以用 `examples.canonical_data_reads` 检查三条读取链路：当前行情版本、冻结行情版本、冻结因子结果。无需复制整库，也不需要旧数据库路径别名。

这不是普通 CSV/DuckDB 导入的必需步骤。输入库需要 `meta.datasets` 版本目录，输出库需要已发布的因子产物；版本引用必须来自自己的目录或运行回执，不能猜测版本号。

配置形状如下（替换路径、版本、因子名和股票后使用）：

```json
{
  "input_db": "/absolute/path/input.duckdb",
  "output_db": "/absolute/path/output.duckdb",
  "price_versions": {
    "current": {"role": "input", "table": "main.price_data_daily", "version": "CURRENT_VERSION_HASH"},
    "frozen": {"role": "input", "table": "main.price_data_daily", "version": "FROZEN_VERSION_HASH"}
  },
  "factor_version": {"role": "output", "table": "main.factor_values", "version": "FACTOR_ARTIFACT_ID"},
  "factor_name": "MyFactor",
  "stocks": ["000001.XSHE", "000002.XSHE"],
  "start_date": "2019-01-02",
  "end_date": "2019-01-10"
}
```

```bash
python -m examples.canonical_data_reads --config config.json --receipt receipt.json
```

选一个短日期区间、少量股票和已有因子。`current` 是显式选择的版本标签，运行时不会自动追踪最新版本。两个行情版本由独立连接读取，避免同名表的版本互相覆盖。所有连接只读，忽略环境中的 `QUANT_DATA_BINDINGS`；此入口拒绝 `sources` 和外部 `data_bindings` 配置。

验证内容：

- 用 `DuckDBAdapter` 和 `FactorReader` 读取样本，再与 `read_reference` 的指定版本逐值比较。
- 检查连接只附加配置中的输入/输出库。
- 将冻结行情的一日价格变化与已有因子按日期/股票对齐，计算截面排名均值，检查最小数据处理流程能完成。
- 输出所选引用、样本行数、内容哈希和流程结果哈希；不导出原始样本数据。

缺失版本、空样本或没有有效重叠样本都会失败；回执已存在也会拒绝覆盖。这不是完整实验复现、策略回测或交易验证，不训练模型、不发布因子、不产生交易。原实验只有通过对应的完整回放验收后才可标记为“已验证可复现”。

配置必须是 JSON 对象，包含 `input_db`、`output_db`、`price_versions`（恰含 current/frozen 两个 DataRef）、`factor_version`、`factor_name`、非空唯一字符串列表 `stocks`、`start_date`、`end_date`。所有 DataRef 必须是对象；数据库路径与日期、因子名必须是非空字符串。配置错误在 CLI 中以 stderr JSON 返回，错误码 `canonical_data_reads_config_error`、退出码 2，不生成成功回执；库函数仍将异常交给调用方。已有回执拒绝覆盖。
