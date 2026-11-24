# 新增一个市场库

## 1. 编写 YAML 声明

从仓库内的模板复制一份配置。为每个物理库选择唯一的 `db_id`，按交易规则和日历
选择 `market_domain`。填写允许的 Slot provider、资产类型、交易所及别名、带交易所
后缀的标的格式、时区、日历来源及本地日期覆盖、币种、频率、交易时段视图和市场规则。
配置文件属于该库的版本化输入，`profile_hash` 由规范化内容计算。

- A 股、国内期货、全球期货和美股模板见 [config/market_databases](../config/market_databases/)。
- French、Binance 和 AKShare 的独立配置及重建命令见 [公开数据市场库](PUBLIC_MARKET_DATABASES.md)。
- 以下示例使用美股模板的 `us_equity_fixture_v1`、`XNYS`、`America/New_York` 和 `USD`。

## 2. 初始化新库或接管已有库

命令在仓库根目录执行。为新库选择支持原子排他重命名的目录，并预留唯一文件名。

```bash
mkdir -p work/market-db-example
db=work/market-db-example/us_equity.duckdb
profile=config/market_databases/us_equity.yaml
python3 -m tools.agent_harness.cli market-db init --profile "$profile" --db "$db" --dry-run
python3 -m tools.agent_harness.cli market-db init --profile "$profile" --db "$db"
```

现有 A 股主库采用只读接管。先记录原库文件的 SHA-256、尺寸和修改时间，再执行以下
命令。设 `existing_db` 为现有 DuckDB 的路径，随后重新核对同一组指纹。

```bash
python3 -m tools.agent_harness.cli market-db adopt --sidecar --db "$existing_db" \
  --profile config/market_databases/cn_equity.yaml
python3 -m tools.agent_harness.cli market-db show --db "$existing_db"
```

接管仅在同目录创建 `<database>.market_profile.json`，因此目录须允许写入 sidecar。
旧库以只读方式使用。源库指纹变化、缺少 profile 或两份声明不一致时，读取会失败。

## 3. 绑定库 ID

为当前 Shell 设置 JSON 绑定。已有其他库的绑定时，向现有 `market_dbs` 对象添加条目。

```bash
export QUANT_DATA_BINDINGS='{"market_dbs":{"us_equity_fixture_v1":"work/market-db-example/us_equity.duckdb"}}'
python3 -m tools.agent_harness.cli market-db show --market-db us_equity_fixture_v1
```

跨工作目录执行时使用稳定的绝对路径。核对返回的 `db_id`、`market_domain`、
`profile_hash` 和 `source_kind`。

## 4. 导入 Bar

CSV 使用 UTC Bar 起点、规范频率及带后缀的 instrument ID。元数据 JSON 数组逐个声明
`instrument_id`、`asset_class`、`exchange`、`timezone`、`calendar_id`、
`rth_reference_calendar` 和 `price_adjustment`。例如 `AAPL.XNAS` 使用
`XNAS` 交易所和 `XNYS` 日历。期货需额外声明具体合约属性，CME 标的须显式提供
`calendar_id`。

```bash
python3 -m tools.agent_harness.cli import-bars --db "$db" \
  --bars work/market-db-example/bars.csv \
  --metadata work/market-db-example/instruments.json --dry-run
python3 -m tools.agent_harness.cli import-bars --db "$db" \
  --bars work/market-db-example/bars.csv \
  --metadata work/market-db-example/instruments.json
```

导入先核对库配置、标的和交易网格，然后在同一事务中写入元数据、Bar 和目录记录。
已经有符合契约的声明式输入时，也可用 `import-bars --manifest <slot.yaml>`，
并用独立的新库作为导入目标。

## 5. 绑定 Slot 并运行研究

读取时核对 `market_rules` 中的 `st_filter`、`price_limit`、`t_plus` 是否与
所选 provider 的执行能力一致。出现差异时，`MARKET_DB_PROFILE_INVALID` 会列出对应
字段、profile 声明值和实现值。A 股为 `{true, true, 1}`，国内期货和全球期货为
`{false, false, 0}`，顺序与上述字段一致。声明式 Slot 的 `price_limit` 和
`t_plus` 分别为 `false` 和 `0`。若 `aux_features` 经列映射后提供
`is_st`，则声明 `st_filter: true`，其余情况声明 `false`。如配置辅助表或文件，
绑定时须能读取其列结构。无辅助源时使用 `aux_features: {provider: none}`，如 XNYS
基础示例，对应默认模板的 `st_filter: false`。
期货模板的 `price_limit: false` 表示当前执行层未实现期货涨跌停约束。
这三项用于核对声明。VectorBT 日内年化通过 `bars_per_session` 读取库配置。
[市场库日历、规则及可用性契约](TIME_SERIES_CONTRACT.md#市场库日历和数据可用性)

A 股原生 Slot 可直接用 `jq_ashare_local`。美股日线使用以下声明式 Slot。将
`start` 和 `end` 设置成已导入的交易日期范围，静态标的应与导入文件一致。
在新库内对 `market_bars_1d` 创建读取视图。声明式适配器使用这个列投影，规范
Bar 表继续用于存储和审计。

```sql
CREATE VIEW us_daily_inputs AS
SELECT instrument_id, timestamp_utc, open, high, low, close, volume FROM market_bars_1d;
```

`--market-db` 从绑定补齐 bar feed 的 `path`。将下面的配置保存为
`work/market-db-example/slot.yaml`。

```yaml
schema_version: 1
slot_id: us_daily
market_domain: us_equity
calendar:
  provider: pmc
  options: {calendar_id: XNYS, start: "2026-03-05", end: "2026-03-13"}
bar_feed:
  provider: duckdb_table
  options:
    table: us_daily_inputs
    frequency: 1d
    timestamp_anchor: bar_start
    source_timezone: America/New_York
    volume_mode: interval
    data_version: sample-v1
    column_mapping: {timestamp_utc: timestamp, instrument_id: instrument_id, open: open, high: high, low: low, close: close, volume: volume}
universe:
  provider: static
  options:
    instruments:
      AAPL.XNAS: {asset_class: equity, currency: USD}
      MSFT.XNAS: {asset_class: equity, currency: USD}
      JPM.XNYS: {asset_class: equity, currency: USD}
aux_features: {provider: none}
microstructure: {provider: open}
execution_profile:
  entry_price_field: close
  exit_price_field: close
  min_entry_offset_bars: 0
  exit_adjustment: 0
  return_unit: decimal
  gap_rule: endpoints_only
  boundary_policy: cross_session
```

Python 因子计算通过 `open_research_slot(db_id, slot_config, base_dir)` 取得该库的 Slot。
在其上下文中构造因子，向 `FactorBase` 传入 `slot`、日期和标的列表。调用
`get_factor_values()` 后用 `save_bar_factor_values(store)` 写入独立的
`MarketFactorValueStore`。研究配置内填写同一仓库的 `factor_store_root`、因子 ID、
频率、日期区间、评测模式与 horizon。已计算的因子可用以下入口评测。

```bash
python3 -m tools.agent_harness.cli single-factor-analysis \
  --config work/market-db-example/evaluation.json \
  --slot-config work/market-db-example/slot.yaml --market-db us_equity_fixture_v1 \
  --results-db work/market-db-example/evaluations.duckdb
```

`time-series-analysis` 接受相同的库选择参数。`factor-synthesis` 补充合成配置、
`--factor-values-store`、`--frequency` 和同一个 `--market-db`。检查运行摘要、
方案 B 因子身份及评测产物内的 `db_id` 和 `profile_hash`。
[可复现的 XNYS 导入、因子计算及评测样例](../tests/test_stage7_us_equity_e2e.py)

## 6. 迁移与审计

对于拥有内置配置表的自管库，修改 YAML 后先预演再提交迁移。

```bash
python3 -m tools.agent_harness.cli market-db migrate-profile \
  --db "$db" --profile work/market-db-example/profile-next.yaml --dry-run
python3 -m tools.agent_harness.cli market-db migrate-profile \
  --db "$db" --profile work/market-db-example/profile-next.yaml
python3 -m tools.agent_harness.cli market-db show --db "$db"
```

迁移保留 `db_id` 与 `market_domain`，在同一事务内将旧配置和旧哈希写入
`market_db_profile_history` 并登记新配置。摘要报告 `previous_profile_hash`、
当前 `profile_hash` 和 `profile_changed`，相同哈希重复执行是幂等操作。
新哈希用于后续研究，历史因子及评测保留原版本。采用 sidecar 的库在只读接管
流程下管理。
