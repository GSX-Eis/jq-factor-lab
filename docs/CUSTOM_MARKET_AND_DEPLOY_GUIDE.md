# 自定义市场 slot.yaml 接入与 DeployTarget 导出指南

## 声明并导入行情

以 [Crypto 24/7 样例](../examples/declarative_crypto/slot.yaml) 为起点。
相对路径按配置文件目录解析。配置应覆盖研究所需 Session 和标的，
并声明时间戳口径和数据版本。

~~~yaml
schema_version: 1
slot_id: crypto_example
calendar:
  provider: custom_sessions
  options:
    calendar_id: crypto_utc
    start: "2026-01-03"
    end: "2026-01-31"
    custom_sessions:
      timezone: UTC
      weekdays: [0, 1, 2, 3, 4, 5, 6]
      segments:
        - {open: "00:00", close: "00:00", open_day_offset: 0, close_day_offset: 1}
bar_feed:
  provider: csv_files
  options:
    paths: [bars.csv]
    frequency: 1h
    timestamp_anchor: bar_end
    source_timezone: UTC
    volume_mode: interval
    data_version: sample-v1
    column_mapping: {symbol: instrument_id, time: timestamp, open: open, high: high, low: low, close: close, volume: volume}
universe:
  provider: static
  options:
    instruments:
      BTC-USD: {asset_class: crypto, currency: USD}
      ETH-USD: {asset_class: crypto, currency: USD}
    universes:
      spot: [BTC-USD, ETH-USD]
aux_features: {provider: none}
microstructure: {provider: open}
execution_profile:
  entry_price_field: open
  exit_price_field: close
  min_entry_offset_bars: 1
  exit_adjustment: 0
  return_unit: decimal
  gap_rule: endpoints_only
  boundary_policy: cross_session
  allow_short: true
~~~

准备包含配置列名的本地 `bars.csv`。样例仓库的两根 Bar 可演示导入。完整评测和合成需要覆盖多个 Session 的行情及有效因子值。

~~~sh
python3 -m tools.agent_harness.cli import-bars --manifest slot.yaml --db work/crypto-bars.duckdb
~~~

检查导入返回的覆盖和缺口计数。声明式 Slot 按自身的 `bar_feed` 读取输入。
导入的 DuckDB 可以用于审计。若希望 Slot 从 DuckDB 取数，在配置中显式选择
`duckdb_table` 数据源并声明路径和表。

## 计算、评测与合成

将现有 `FactorBase` 子类绑定到同一 Slot，然后写入方案 B 存储。示例中的
`Momentum` 和 `Reversal` 代表项目中已实现的两个因子类。

~~~python
from slots import build_slot_from_config
from utils.market_factor import MarketFactorValueStore

slot = build_slot_from_config("slot.yaml")
store = MarketFactorValueStore("work/crypto-factors")
for factor_type, name in [(Momentum, "momentum"), (Reversal, "reversal")]:
    factor = factor_type(
        None, name, "2026-01-03", "2026-01-31", slot=slot,
        stock_pool=["BTC-USD", "ETH-USD"], data_frequency="1h",
        analysis_mode="time_series",
    )
    print(factor.save_bar_factor_values(store).factor_identity_hash)
~~~

在 `evaluation.json` 设置相同的 Slot、日期和因子身份。时间序列模式按标的
评测。截面模式要求每个信号时刻覆盖足够的可用标的。

~~~json
{
  "slot_config": "slot.yaml",
  "slot_id": "crypto_example",
  "factor_store_root": "work/crypto-factors",
  "factor_id": "momentum",
  "start_date": "2026-01-03",
  "end_date": "2026-01-31",
  "frequency": "1h",
  "analysis_mode": "time_series",
  "horizons": [1, 6],
  "n_quantiles": 2
}
~~~

~~~sh
python3 -m tools.agent_harness.cli time-series-analysis \
  --config evaluation.json --output-dir work/crypto-evaluation
~~~

存在多个已登记身份时，评测用 `factor_identity_hash` 固定选择。检查
`bar_evaluation.v2` 的有效计数、剔除原因和统计量，再配置合成。
以下 `synthesis.json` 使用同一日期范围和两个因子。

~~~json
{
  "analysis_period": {"start_date": "2026-01-03", "end_date": "2026-01-31"},
  "factor_groups": {"all": {"factors": ["momentum", "reversal"]}},
  "synthesis_methods": {
    "mix": {"method": "weighted_sum", "factor_group": "all",
            "weights": {"momentum": 0.7, "reversal": 0.3},
            "preprocess": {"ALL": "raw_zscore"}}
  },
  "performance_evaluation": {
    "top_k_strategy": {"enabled": true, "top_k": 1, "holding_period": 6, "rebalance_freq": "bar"}
  },
  "output_settings": {"composite_factor_prefix": "crypto"}
}
~~~

~~~sh
python3 -m tools.agent_harness.cli factor-synthesis \
  --config synthesis.json --slot-config slot.yaml --slot-id crypto_example \
  --factor-values-store work/crypto-factors --frequency 1h --run-id crypto
~~~

从成功结果的 `metrics.strategy_bundle.bundle_dir` 取得 bundle 路径。
若配置生成多个候选，从 `metrics.strategy_bundles` 按评测和试验身份选择。
检查 VectorBT 交易数、净值和回测收益，以及 bundle 中锁定的因子身份和执行配置。

## 导出信号

`target_positions` 包含调仓决策时间 `timestamp_utc`、对应 `session_id`、
`instrument_id` 和 `target_weight`，以 Parquet 或 JSONL 提供。需要按当前
持仓计算首期差额时，传入 `--current-positions`，内容为标的到当前权重的
JSON 对象。若 bundle 已绑定目标仓位文件，可省略 `--target-positions`。

从 VectorBT 调仓审计生成开盘仓位时，使用 `target_instructions.csv` 的
`decision_target_weight`。该字段用指令金额除以执行前按开盘价标记的模拟组合净值，
缺少在仓标的的开盘价或净值无效时留空。旧字段 `target_weight` 用于收盘估值诊断。
导出时间使用执行 Bar 的 UTC 开盘时间，跨夜期货使用所属 Session 的开盘时间。

~~~sh
python3 -m tools.agent_harness.cli export-strategy --target signal_export \
  --bundle-dir "$BUNDLE" --output-dir work/signal-candidate --target-positions positions.parquet

python3 -m tools.agent_harness.cli verify-target-alignment --target signal_export \
  --bundle-dir "$BUNDLE" --artifact-dir work/signal-candidate --reference-positions positions.parquet
~~~

用 `deploy_manifest.json` 核对仓位 Parquet、JSONL 和 `trade_orders.jsonl` 的
SHA-256。提供参考仓位可额外比较目标权重。
两条 Harness 命令均输出 `execution_result.v1`。在 `metrics.deploy_manifest`
查看导出清单，在 `metrics.alignment_report` 查看核验项和差异。

## 导出聚宽候选

用 A 股 `jq_ashare_local` 的独立 `weighted_sum` bundle，且执行配置仅做多。
因子源和特征必须能映射到当前聚宽离线编译契约。必要时可为导出重复指定
`--factor-file` 并提供 `--requirements` JSON。

~~~sh
python3 -m tools.agent_harness.cli export-strategy --target jq_cloud \
  --bundle-dir "$ASHARE_BUNDLE" --output-dir work/jq-candidate

python3 -m tools.agent_harness.cli verify-target-alignment --target jq_cloud \
  --bundle-dir "$ASHARE_BUNDLE" --artifact-dir work/jq-candidate \
  --trusted-manifest-hash "$TRUSTED_MANIFEST_HASH" \
  --replay-inputs replay.json --reference-factors factors.parquet --reference-positions weights.json
~~~

`replay.json` 包含 `stocks`、ISO `asof` 和按聚宽周期代码索引的
`bars_by_frequency`（例如 `daily` 对应含 `date`、`code` 和所需行情列的记录数组）。
参考因子 Parquet 按标的代码索引，每个因子占一列。
`TRUSTED_MANIFEST_HASH` 取自本地可信编译的导出结果
`metrics.deploy_manifest.manifest_hash`。把该值单独保存在可信运行记录中，
再用于核验收到的候选目录。候选目录内的副本只用于校验文件完整性。
虚拟重放会执行已认证的候选 Python，编译时应只选可信的因子源码。
`weights.json` 是标的代码到目标权重的对象。结果给出候选语法、加载、
运行预算和虚拟云端因子与持仓检查。核验后由人工复核交接。
