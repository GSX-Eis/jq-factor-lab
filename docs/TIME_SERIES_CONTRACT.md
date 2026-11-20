# 时序研究契约

## 阶段 6.6：Bar 评估结果入库（`bar_evaluation_*`、`--results-db`、只读查询与监控）

由模块 [`utils/bar_evaluation_store.py`](../utils/bar_evaluation_store.py) 与只读监控页 `utils/bar_evaluation_app.py`（私有监控入口） 实现。零引入 SQLite，零改动日频 `factor_analysis_results`、`factor_values`、`price_data_daily`、`price_data_5m` 及既有缓存键。

### 表结构与双模式兼容性

结果写入显式指定的 DuckDB。首次写入时创建以下五张 Bar 表，其他表保留原样。

1. **`bar_evaluation_runs`**（每次评估一行，主键 `identity_hash VARCHAR PRIMARY KEY`，即 `result_identity.identity_hash`）：
   - 模式与网格：`analysis_mode`、`frequency`（单频率存单值字符串如 `'1min'`；多频率存规范 JSON 数组）、`view`、`boundary_policy`、`bucket_strategy`、`calendar_id`、`calendar_hash`；
   - 身份与版本：`config_hash`、`input_file_hashes`（JSON）、`data_identity`（JSON，如实记录来源、`data_versions` 及已存在的 `build_id`/`build_ids`/`roll_build_ids`/`continuous_futures`，不编造缺失字段）、`code_version`、`code_version_source`、`result_schema_version`；
   - 统计与产物存证：`exclusion_summary`（JSON）、`summary_path`、`summary_sha256`、`per_bar_labels_path`、`per_bar_labels_sha256`、`ic_by_horizon_path`、`ic_by_horizon_sha256`、`quantile_stats_path`、`quantile_stats_sha256`、`run_id`、`ingested_at_utc`（`TIMESTAMPTZ`）。
   - v2 新增可空列 `has_split BOOLEAN` 表示是否使用 train/OOS，
     `split_config_json VARCHAR` 保存各段除 `horizons` 外的元数据，
     `trial_ledger_json VARCHAR` 保存候选账本，
     `calibration_summary_json VARCHAR` 保存标准表的 `interval_usability`。只读查询将 JSON 文本解码。
   - `per_bar_labels.csv` **不入库**，仅在 `bar_evaluation_runs` 记录路径与 SHA256。
2. **`bar_evaluation_ic`**（主键 `(identity_hash, row_index)`）：
   - 采用**列取并集 + 可空 + `analysis_mode` 列**结构，同时容纳 `time_series`（`TS_IC_COLUMNS`）与 `cross_sectional`（`CS_IC_COLUMNS`）两种模式；非当前模式的专属列存为 `NULL`。
3. **`bar_evaluation_quantiles`**（主键 `(identity_hash, row_index)`）：
   - 采用**列取并集 + 可空 + `analysis_mode` 列**结构，同时容纳 `TS_QUANTILE_COLUMNS` 与 `CS_QUANTILE_COLUMNS`；非当前模式的专属列存为 `NULL`。
4. **`bar_evaluation_cross_sections`**（主键 `(identity_hash, row_index)`）：
   - 对应 `CS_COLUMNS`，仅 `cross_sectional` 模式有行（`signal_timestamp TIMESTAMPTZ NOT NULL`），`time_series` 模式写入 0 行。

5. **`bar_evaluation_split_metrics`**（主键由 `identity_hash VARCHAR` 和 `row_index BIGINT` 组成）
   - 每行对应 `summary.json` 中一个 split 的一条 horizon 统计。`split`、`frequency`、
     `interval_method`、`calibration_status` 为非空 `VARCHAR`，`instrument_id VARCHAR`
     在截面行为空。`horizon_bars`、`split_purged_count`、`n_observations` 为非空 `BIGINT`。
   - `ic`、`n_dependence_adjusted`、`interval_lower`、`interval_upper` 为可空 `DOUBLE`。
     `ic_reason`、`p_value_method` 为可空 `VARCHAR`。`descriptive_stats_usable` 和
     `significance_interpretation_usable` 为非空 `BOOLEAN`。`p_value`、`adjusted_p_value`、
     `inference_adjusted_p_value` 为可空 `DOUBLE`，`significant` 为可空 `BOOLEAN`。
   - `fit_identity_json VARCHAR` 保存可用的训练拟合身份，`metrics_json VARCHAR NOT NULL`
     保存该行完整统计和校准证据。两列以 JSON 文本存储。候选能唯一对应评估流时，
     在候选选择段保存其校正 p 值和显著性判定。轻量评估在此表写入 0 行。

### 旧 v1 DuckDB 的显式迁移

写入仅接受已发布的 `bar_evaluation.v2` 产物。写入前核对 DuckDB 主 schema 的必需表、
列类型和主键。v1 库缺少 v2 列或表时，写入事务回滚，Harness 返回退出码 `2`
并提示迁移。包含原四张表的 v1 库仍可只读查询，新增运行列以 `null` 返回，推断行为空。

1. 暂停结果库写入，关闭连接，并保留已关闭的旧库快照。
2. 在显式指定的现有库或其工作副本上执行迁移。

   ```bash
   python -c 'import sys; from utils.bar_evaluation_store import migrate_bar_evaluation_db; migrate_bar_evaluation_db(sys.argv[1])' work/bar-evaluation/results.duckdb
   ```

3. 用 `read-bar-evaluations` 复核历史 identity 和行，用 `monitor-refresh` 检查
   `bar_evaluation_split_metrics` 已存在，并核对旧运行新增的列为 `null`。验收后恢复 v2 写入。

迁移在同一库的独占写锁下验证原四表及主键，单事务添加可空列和 split 表，且可重复执行。
旧记录的版本与产物散列保持原值，历史行不会回填推断。校验或 DDL 失败时事务回滚。
迁移已提交后如需回到旧库，关闭连接后从迁移前快照恢复。先保留已迁移库以保存任何
后续 v2 写入，避免恢复旧快照时丢失该批证据。

### 写入规则、锁选择与幂等/冲突语义

- **写锁选择理由**：写入串行锁沿用 `utils.duckdb_results_writer.acquire_exclusive_write_conn`（基于 `<db>.write.lock` 的 `fcntl.flock` + DuckDB 文件锁重试 + `_reject_sqlite_file`），而不使用 `utils.data_store.write_store`，因为 `write_store` 会创建并更新 `meta.*` 编目与缓存版本表，违背阶段 6.6“只管理 `bar_evaluation_*` 表、不碰库内其他任何表”的隔离边界。库被其他进程占用时沿用该重试/等待机制，不静默跳过。
- **发布校验**：入库前只读取已发布产物（`summary.json` 及三个 CSV 均须存在）。调用方必须通过 `artifacts` 或 `expected_hashes` 提供 Harness 发布的全部四份 SHA256，缺失或格式非法立即拒绝；重新计算四文件 `sha256_file` 并与其核对一致，校验 `ResultIdentity.identity_hash()` 复算一致，并核对 CSV 与 `summary.json` 的统计、计数和身份一致，同时校验切分网格、区间和候选账本。取得写锁后再次核验四文件未在解析或等待期间改变；任一核对失败立即拒绝。
- **持久化目标校验**：已有目标必须是带原生 DuckDB 文件头（偏移 8 的 `DUCK` 标识）的普通文件；CSV、JSON、空文件及其他非原生格式在开库前以退出码 `2` 拒绝，原文件不变。扩展名不作为判断依据。取得写连接后，必须由 `current_database()` 与 `PRAGMA database_list` 确认当前活动库对应显式指定的持久化文件，拒绝内存库或其他库的连接，再开始写入事务。普通目标前置条件失败仍撤销评估完成标记。
- **单事务原子性**：建表、幂等/冲突检查与五张表的 `INSERT` 在同一个 DuckDB `BEGIN TRANSACTION ... COMMIT` 事务内完成；任一步异常立即 `ROLLBACK`，不留任何残留行。不提供任何 `UPDATE`/`DELETE` 接口。
- **幂等与冲突检测**：比较时对 `DOUBLE` 的 `NaN`/`NULL` 统一归一（`-0.0` 归为 `+0.0`，按 IEEE-754 位模式比对），JSON 字段按解析后的结构比对。同一 `identity_hash` 下语义字段（含 4 份 SHA256 与配置/数据/日历身份）及 `bar_evaluation_ic`、`bar_evaluation_quantiles`、`bar_evaluation_cross_sections` 和 `bar_evaluation_split_metrics` 全部行完全一致时为纯 no-op（`status='noop'`，各表行数不变，`ingested_at_utc` 与 `run_id` 保留首次记录）；任一字段或指标存在差异即抛出 `BarEvaluationConflictError(ValueError)` 并回滚。

### CLI 参数优先级、退出码与页面启动

- **`--results-db` 参数与优先级**：
  - `time-series-analysis` 与分流至 Bar 路径的 `single-factor-analysis` 均支持可选 `--results-db <path>`，也可在配置 JSON（顶层或 `time_series_analysis` 块）中写 `results_db`。
  - 两者同时存在时，**命令行 `--results-db` 优先**（按调用方工作目录解析），配置 `results_db` 按仓库根目录解析。`results_db` 在传给子进程 `tools/agent_harness/time_series_analysis.py` 前从 `resolved.json` 剥离，不进入 `config_hash` 计算；未传 `--results-db` 且配置未声明 `results_db` 时，照常发布 `bar_evaluation.v2` 产物，不创建 DuckDB 文件。
  - 若 `single-factor-analysis` 未分流到 Bar 路径（即保留日频 `run_analysis_main.py`）却显式传入 `--results-db`，直接以退出码 `2` 拒绝。
  - 子进程评估成功后由父进程入库：成功时 Harness `metrics["results_db"]` 记录 `path`、`identity_hash` 与 `status`（`inserted` 或 `noop`）。
  - **输出路径隔离**：在任何完成标记清理、子进程发布、Harness/JSON 镜像写入之前，拒绝结果库与四份评估产物、Harness attempt/run summary、`config/resolved.json`、`output_bindings.json`、`--json-out` 重合（包括符号链接、硬链接和文件/目录祖先冲突）。命令行及配置指定的库均受保护；dry-run、fake-result、只读查询以及日频拒绝路径也不能通过镜像覆盖库。路径冲突返回退出码 `2`，仅向 stdout 返回失败 JSON，**不写错误镜像、Harness 文件，也不删除完成标记**。
  - 写库失败时按现有规则撤销 `evaluation/summary.json`，并按 `EVAL_POLICY.md` 返回区分写库原因的失败结果：
    - 退出码 `2`（`bar_evaluation_store_precondition_failed`）：路径非法、旧库 schema、非 DuckDB 文件或发布产物/SHA256 校验失败；
    - 退出码 `4`（`bar_evaluation_store_conflict`）：同 `identity_hash` 内容冲突（`BarEvaluationConflictError`）；
    - 退出码 `3`（`bar_evaluation_store_write_failed`）：其他 DuckDB 写入执行异常。
- **只读查询与监控**：
  - `read-bar-evaluations --results-db <path>` 支持 `--identity-hash`、`--analysis-mode`、`--frequency`、`--calendar-id`、`--start-utc`、`--end-utc`（按 `ingested_at_utc` 闭区间过滤，接受 `YYYY-MM-DD` 或带时区 UTC 时间）与 `--json-out`。以 DuckDB `read_only=True` 打开。库不存在、为非 DuckDB 文件或缺少原四张必需表时返回退出码 `2`。
  - 推断筛选支持 `--split {full,train,oos}`、`--calibration-status {uncalibrated,calibrated_pass,calibrated_fail}`，
    以及 `--descriptive-stats-usable`、`--significance-interpretation-usable`、`--has-trial-ledger`
    和对应的 `--no-...` 形式。匹配的运行返回标准表和筛选后的 split 行。
    没有推断行的轻量运行可按标准表的校准可用性匹配，指定 split 时只匹配推断行。
  - `monitor-refresh --database-path <path>` 在 `metrics.bar_evaluation` 报告五张表各自的
    `exists` 和 `row_count`，以及 `run_count`、`latest_ingested_at_utc`。
    `all_tables_exist` 检查原四表，因此兼容旧库。空库或没有 Bar 表的库仍可返回
    退出码 `0`，此时 `any_table_exists=false`。
- **只读 Streamlit 页面启动方式**：
  ```bash
  streamlit run utils/bar_evaluation_app.py -- --results-db work/NERE-749/results.duckdb
  # 或通过环境变量（命令行 --results-db 优先于环境变量）：
  QUANT_BAR_RESULTS_DB=work/NERE-749/results.duckdb streamlit run utils/bar_evaluation_app.py
  ```
  页面按评估模式、频率、日历和运行筛选，展示原始统计、身份、候选账本和按 split
  分组的 IC、区间、校准与显著性门控。页面及查询以 `read_only=True` 打开库，
  导入模块不触发数据库连接。

## 阶段 6.5b：分钟因子值入库（`market_factor_values`，方案 B）

默认存储方案（大古于 6.5a/NERE-747 选定）由模块 [`utils/market_factor_store.py`](../utils/market_factor_store.py) 实现，采用 **Hive 分区 Parquet + `fcntl.flock` 保护的只追加 JSONL 账本 + DuckDB 只读视图**。零引入 SQLite（符合 `DUCKDB_ONLY.md`），零改动日频 `factor_values`、`factor_analysis_results`、`price_data_daily`、`price_data_5m` 及既有 Parquet 导出。

> **POSIX 锁说明**：写入串行锁使用 `fcntl.flock`（`.ledger.lock`），仅支持 POSIX 宿主（macOS / Linux）。只读端（`read_factor_values`、`list_identities`、`create_view`、`verify_store`）不获取写锁，写进程持锁期间读进程可立即无锁读取已提交分片。

### 物理布局与表结构

`MarketFactorValueStore(root)` 必须显式传入 `root` 目录，不设默认路径：

- 分区数据：`parquet_data/factor_id=<factor_id>/frequency=<frequency>/year_month=<YYYY-MM>/part-*.parquet`（ZSTD 压缩，写入前按 `(factor_identity_hash, instrument_id, timestamp_utc)` 排序，默认 `row_group_size=65536`）。
- 身份账本：`identities.jsonl`（只追加，对应 `factor_value_identities`）。
- 批次账本：`batches.jsonl`（只追加，记录 `status='committed'` 批次及其分片清单与 Session 计数）。
- 逻辑列（13 个核心列）：
  - `factor_id`（`VARCHAR NOT NULL`，非空单段安全路径标识）、`factor_identity_hash`（64 位小写 SHA256）、`frequency`（`Frequency.canonical`）、`session_view`（`ETH`/`RTH`）、`instrument_id`、`timestamp_utc`（UTC Bar 开始时间，SQL 类型为 `TIMESTAMP_NS`）、`session_id`（`YYYY-MM-DD`，须与 `SessionCalendar.session_at` 一致）、`factor_value`（`DOUBLE`，允许 `NaN`，严格拒绝 `±inf`）、`source_build_id`（非空）、`calendar_hash`（64 位 SHA256）、`code_version`、`created_at`（UTC 创建时间，SQL 类型为 `TIMESTAMP_NS`）、`batch_id`；视图另含 `year_month`。

### DuckDB SQL 视图升级与时间比较

`market_factor_values` 视图的 `timestamp_utc` 和 `created_at` 均以 `TIMESTAMP_NS` 表示 UTC 时间，空视图也使用这两个类型。DuckDB 1.5.3 的 `TIMESTAMPTZ` 只有微秒精度。`TIMESTAMP_NS` 本身没有时区字段，直接与 `TIMESTAMPTZ` 比较会受会话 `TimeZone` 影响。Python 的 `read_factor_values` 返回带 UTC 时区的 `datetime64[ns, UTC]`，闭区间上下界均按纳秒比较。

升级已有持久视图时，先更新使用旧 `TIMESTAMPTZ` 谓词或连接条件的查询，再以可写连接刷新保存该视图的 DuckDB 文件。传入原因子库根目录和原视图名。刷新以 `CREATE OR REPLACE VIEW` 重新绑定当前已提交分片，可在刷新后使用 `DESCRIBE market_factor_values` 核对类型。Parquet 和账本可原样使用。后续有新批次时，再运行同一刷新调用即可让持久视图看到新增分片。

```python
import duckdb
from utils.market_factor_store import MarketFactorValueStore

store = MarketFactorValueStore("/path/to/factor-store")
with duckdb.connect("/path/to/views.duckdb") as con:
    store.create_view(con, view_name="market_factor_values")
```

SQL 的比较和连接可统一使用 UTC epoch 纳秒，适用于任意会话时区。

```sql
-- 带时区参数精确到微秒时，绑定含 UTC 偏移的值，例如 2026-01-04T20:30:00-05:00。
SELECT * FROM market_factor_values
WHERE epoch_ns(timestamp_utc) = epoch_ns(CAST(? AS TIMESTAMPTZ));

-- 纳秒参数绑定整数 UTC epoch 纳秒，例如 1767576600000000001。
SELECT * FROM market_factor_values
WHERE epoch_ns(timestamp_utc) = CAST(? AS BIGINT);

-- 行情表的 Bar 起点是 TIMESTAMPTZ，按同一品种和同一时间点连接。
SELECT f.* FROM market_factor_values AS f
JOIN market_bars AS b
  ON f.instrument_id = b.instrument_id
 AND f.frequency = b.frequency
 AND epoch_ns(f.timestamp_utc) = epoch_ns(b.timestamp_utc);
```

闭区间可用 `BETWEEN` 比较同样归一后的两个上下界。固定的 UTC 纳秒时间也可直接使用同类型的字面量，例如 `TIMESTAMP_NS '2026-01-05 01:30:00.000000001'`。现有 Bar 列的微秒精度决定了可精确连接的时刻，整分钟 Bar 只命中整分钟因子键。

### 身份计算与冻结粒度约定

`compute_factor_identity_hash(...)` 对包含以下字段的规范 JSON（`sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False`）取 UTF-8 SHA256（schema `"market_factor_identity.v1"`）：

1. `factor_id`、`factor_class`（全限定名 + MRO/桥接/类源码 `source_sha256`）、`params`（`_get_cache_inputs()` 严格 JSON 参数及辅助 DataFrame 摘要）；
2. `frequency`、`output_frequency`、`session_view`、`analysis_mode`、`bucket_strategy`；
3. `calendar_hash`（单值或逐标的 `{instrument_id: hash}`）与可选 `rth_reference_calendar`；
4. `source_identity`（数据源元信息、`data_version`、`source_build_id`，连续合约另含 `adjust` 与 `roll_build_ids`）；
5. `code_version`（`resolve_code_version()`）。

**冻结粒度约定（6.5b）：**
1. **增量追加时保持身份不变的字段**：只要因子定义（`factor_id`、`factor_class`、`source_sha256`、`params`）、网格参数（`frequency`、`output_frequency`、`session_view`、`analysis_mode`、`bucket_strategy`）、代码版本（`code_version`）、日历区间（`calendar_hash`）与数据源版本（`source_dataset_version` / 显式 `data_version`）保持不变，在 `identity_scope="incremental"` 下按 Session 增量追加新交易日将保持同一 `factor_identity_hash`，同时每行与 `batches.jsonl` 仍忠实记录逐 `(instrument_id, session_id)` 的 `source_build_id`。
2. **上游 `build-bars` 重建导致 `source_build_id` 变化**：
   - 在默认 `identity_scope="exact"` 下，逐 Session `source_build_id` 直接锁入 `factor_identity_hash`，上游重建会自动生成新版本 `factor_identity_hash`（与旧版本在同一 `factor_id` 下共存）。
   - 在 `identity_scope="incremental"` 下，对**已写入的同一 `(instrument_id, session_id)`**，若上游 `source_build_id` 发生变化，写入器以 `strict_source_build_id=True` **拒绝写入**（即使因子数值未变），防止同一身份下静默混入不同构建来源；调用方须以新身份版本重算。
3. **日历区间扩大时的 `calendar_hash` 处理**：`SessionCalendar.calendar_hash` 包含声明区间 `[start, end]`，扩大区间必然改变 `calendar_hash` 并产生新 `factor_identity_hash`。增量追加流水线必须在初始化时固定覆盖整个研究周期的 `SessionCalendar` 区间（如 `XSHG` 默认区间或显式年度/生命周期区间），不得逐日扩大日历边界；若确需扩大区间，按新版本身份处理。

### 写前校验、按位幂等/冲突与发布可见性

- **无 `UPDATE`/`DELETE` 接口**：`MarketFactorValueStore` 仅暴露 `append_factor_values(...) -> FactorWriteReceipt`，不提供任何 `update`/`delete`/`overwrite` 接口。
- **写前校验**：单批只允许唯一 `(factor_identity_hash, factor_id, frequency, session_view)`；批内主键 `(factor_identity_hash, instrument_id, timestamp_utc)` 重复即拒绝（`DuplicateKeyError`）；`±inf`、naive 或非 UTC 时间戳、非法频率/视图均拒绝；每次非空写入均强制传入 `calendar`（CLI 强制 `--calendar`），逐条校验 `session_id == SessionCalendar.session_at(timestamp_utc, session_view).session_date`；首次登记的 `factor_identity_hash` 必须附带可复核的 `identity_record`（对 `identity_payload_json` 规范 JSON 重算 SHA256 与 `factor_identity_hash` 一致），缺失或校验不通过即拒绝，绝不写入占位身份。
- **浮点按位比较与 DuckDB `NaN`/`NULL` 归一**：`canonical_float_bits` 将所有 `NaN`/`NULL` 统一映射为规范缺失哨兵（`0x7FF8000000000000`），将 `-0.0` 规范为 `+0.0`，并对 `1-ULP` 差异严格敏感；幂等与冲突检测严格按逻辑主键 `(factor_identity_hash, instrument_id, timestamp_utc)` 对齐（按时间戳范围/分区扫描候选分片，不以 `session_id` 集合交集跳过冲突检查）；DuckDB SQL 对比同步使用 `sql_canonical_float`（`NULLIF(..., CAST('nan' AS DOUBLE))`）。同身份同值重写入为纯 no-op（`inserted=0`）；任意键发生值冲突（含 `1-ULP`、`NaN`↔有限数）整批拒绝（`FactorValueConflictError`，含 `factor_id`、`factor_identity_hash`、`conflict_count` 及至多 5 条示例），异常路径同步清理本次未入账的目标分片，零部分写入。
- **原子发布与只读可见性**：数据先写入 `.staging-*`，再 `os.replace` 进入 `parquet_data/`，随后在 `LOCK_EX` 下依次追加 `identities.jsonl` 与 `batches.jsonl` 提交记录（若账本末尾存在未以换行结束的半行，读端视为尚未提交并照常返回已提交行，后续写入在持锁下截断该未提交尾残余后继续追加，中段损坏行仍直接报错）。`read_factor_values` 与 `create_view` **只读取 `batches.jsonl` 中已提交的分片路径列表**（使用 `hive_types={'factor_id': 'VARCHAR', 'frequency': 'VARCHAR', 'year_month': 'VARCHAR'}`），绝不通配扫描未提交分片或 `.staging-*`。只按 `factor_identity_hash` 查询时，先通过账本反查 `(factor_id, frequency)` 并注入分区过滤条件，仅扫描目标分区。`verify_store()` 只读核对账本与磁盘分片，报告任何未提交分片、`.staging` 残留、缺片、行数不一致、未登记身份或损坏/未换行残留账本行。
- **FactorBase 无侵入集成**：`save_bar_factor_values(factor, store, ...)` 不改动 `utils/factor_base.py`。不假定 `factor.market_bars.attrs['build_ids']` 存在，优先读显式入参，其次 `attrs`，否则主动只读查询 `bar_build_log` / `bar_catalog.build_id`；均不可得时明确报错，绝不静默写空串。
- **后续月度分片合并接口约束**：若后续增加已封闭月份碎分片压缩合并，只能读取同 `(factor_id, frequency, year_month)` 下已提交分片生成合并后新分片并原子更新账本记录，不得改变任何逻辑行内容或覆盖/丢失批审计记录（6.5b 不实施合并）。

### CLI 入口

```bash
python3 -m tools.agent_harness.cli write-factor-values --factor-values-store work/NERE-748/store \
  --input work/NERE-748/factors.csv --factor-id alpha_001 --frequency 15min \
  --analysis-mode time_series --bucket-strategy segment_anchored --calendar XSHG \
  --calendar-start 2026-01-05 --calendar-end 2026-01-05 --source-build-id build-1
python3 -m tools.agent_harness.cli read-factor-values --factor-values-store work/NERE-748/store \
  --factor-identity-hash <sha256> --list-identities
python3 -m tools.agent_harness.cli verify-factor-values --factor-values-store work/NERE-748/store
```

三者均强制传 `--factor-values-store <dir>`，支持 `--dry-run` / `--json-out`，遵循 `EVAL_POLICY.md` 退出码（`0` 成功/dry-run，`2` 前置校验失败，`3` 执行失败，`4` 冲突拒绝或 `verify_store()` 校验未通过）。

## 阶段 6.4b：国内期货分桶推荐与本地验证

按阶段 6.4b 批准的方案 A，`BarSchedule` 保留“国内期货未显式指定 `bucket_strategy` 就报错”的行为不变，
同时在 `utils.time_axis`（`utils/time_axis/futures.py`）写入项目级推荐值
`RECOMMENDED_DOMESTIC_FUTURES_BUCKET_STRATEGY = 'segment_anchored'` 与辅助接口
`recommended_bucket_strategy(product_root=None)`，供调用方显式传入（`config/futures_calendar_map.json`
保持纯粹的品种日历映射结构不变；股票与 CME 行为不变）。

- **三种口径说明**：项目内采用 `segment_anchored`（推荐，桶不跨休息、可用时间清晰、与股票一致）
  与 `session_clock`（显式可选，除尾桶外等长）；天勤 `3600.parquet` 使用的自然整点切分记为第三种外部口径，
  仅用于 `work/NERE-744/` 交叉核对，不进入生产分桶。
- **聚宽 `1m` 时间戳语义与本地验证**：对只读外接盘 `2025-04-02 .. 2025-05-09`（每品种 24 Session）
  实证聚宽 `1m` 的 `datetime` 为上海时区 **Bar 结束时间**；6 品种（`CU/AU/M/TA/IF/SC`）144/144 Session
  根数与 `expected_bar_opens` 精确匹配，周末/节前/AU・SC 跨午夜归属 32/32 通过，两套策略 `build-bars`
  冒烟 48/48 组完整率 `1.0`；`TA/M` 非换月日 `1d` 与天勤 `86400.parquet` 在 `high/low/close/volume/open_interest`
  上 45/45 精确一致（天勤无 `money` 列；部分日期仅 `open` 有集合竞价口径差），按自然整点脚本重聚合可对齐全部
  316 个同交易日 60m 桶边界。`CU/AU/IF/SC` 无第二数据源，仅完成自洽检查。
- **未来聚宽对齐约定**：以后如需将国内期货信号部署到聚宽或与聚宽回测对齐，须按
  `quantation-alignment` 流程单独核对聚宽原生 Bar 口径。

## 阶段 6.4a：国内期货连续合约

分频率 `market_bars_{1min,5min,15min,60min,1d}` 在写入时幂等执行
`ALTER TABLE ... ADD COLUMN IF NOT EXISTS open_interest DOUBLE`，允许 NULL。
importer 的该列可选，缺省写 NULL；提供时必须为有限、非负实数或 NULL。
离线构建取桶内按时间排序的最后一根源 Bar 的持仓量，最后值 NULL 时仍为 NULL，
不会回找先前非空值。只读入口不迁移：旧表无该列时读取结果也不含该列。
`price_data_daily`、`price_data_5m`、阶段 5 单表 `market_bars` 的结构与写入契约不变。
原始导入仍拒绝 CONT、MAIN、888、000 等连续/主力代码，只存具体合约。

### 离线换月决定

```bash
python3 -m tools.agent_harness.cli build-roll-schedule --db work/NERE-743/input.duckdb \
  --product-roots CU,AU --start 2026-01-06 --end 2026-01-30 \
  --rule max_oi --confirm-days 1 --forward-only
```

必须显式传库路径，只写该库的 `futures_roll_schedule`，不修改原始 Bar。
`--no-forward-only` 允许切回近月。`confirm_days` 为正整数；默认规则如下：

1. 每个有效 Session t 的决定，只读 t-1 及更早的 Session 收盘。
   新合约须在截至 t-1 的连续 N 个交易 Session 收盘均为最大持仓量，才在 t 开盘切换。
   N 默认 1；当日持仓量交叉不在当日生效。
2. 优先选已登记的 `market_bars_1d`。没有该日频数据集时，选已登记的最细分钟频率，
   取该 Session 的最后一个应有槽位；缺少收盘槽位不能使用更早的报价。
   已登记日频但缺行/NULL 时直接失败，不用分钟源掩盖缺口。
   每个决定 Session、每个合约采用的源表写入 `rule_params.oi_sources`。
3. 持仓量相同按 `contract_month` 升序（近月优先），同月再按 `instrument_id` 升序。
   `forward_only=True` 默认只向严格更远月份切换，领先者回到近月时保留现合约。
4. 品种已注册、未到期的具体合约组成候选全集，必须有 `contract_month`，共享日历。
   不从缺失报价推断尚未上市。决定所需持仓量 NULL、缺 Session、收盘桶不完整均报错，
   指明 `product_root`、Session、合约；批次校验后统一事务写入。
   初始状态取首次确认窗口中最早 Session 的领先者，再应用 N 日确认与向远月限制；
   后续增量从已有上一 Session 的 `to_contract` 延续。

表每天保存一条决定（无切换也保存），主键 `(product_root, effective_session_id)`：

| 列 | 类型/含义 |
| --- | --- |
| product_root | VARCHAR，品种根 |
| effective_session_id | VARCHAR，生效交易日 YYYY-MM-DD |
| decision_session_id | VARCHAR，最后决定交易日，数据库约束必须早于生效日 |
| from_contract / to_contract | VARCHAR，前一状态/当前合约；初始 from 为 NULL |
| rule / rule_params | VARCHAR / JSON，规则、N、向远月开关、平局规则及逐合约源表 |
| from_open_interest / to_open_interest | DOUBLE，决定时两个合约持仓量；初始 from 为 NULL |
| source_hash | VARCHAR，决定窗口持仓量、源版本、日历、月份、参数和先前合约的 SHA256 |
| version / build_id | BIGINT / VARCHAR，该品种 × Session 的版本和构建身份 |
| code_version / code_version_source | VARCHAR，复用 `utils.code_version.resolve_code_version` |

相同输入不重写，输入变化后版本加一并产生新 build_id。重建较早区间时也重算已保存的
后续决定，避免旧状态残留；源输入或前一合约未变的后续行保留版本与身份。
补充新的未到期合约后须重建所需区间，不能沿用新 Session 缺失的持仓量。

### 连续视图、调整与标签

`utils.continuous_bars.ContinuousBarSource` 借用连接或只读打开显式库，
`read(product_root, frequency, start, end, view='ETH', *, adjust)` 按每日决定拼接具体合约。
日期界限指交易 Session，UTC 界限指 Bar 开始时点，均含端点；不聚合、不降频回退、
不写回任何表。缺频率、缺决定、所选合约整 Session 无 Bar 抛 `MissingBarDataError`，
消息含 frequency；部分桶仍按原质量规则返回并警告。
`instrument_id` 为品种根，另含 `contract_id`、`session_id`、`is_roll_session`。
切换 Session 的所有行标 True。质量告警按实际合约汇总。同次输入使用同一分桶策略；
国内期货分钟默认策略仍待 6.4b 核对。

`adjust` 必须显式指定，缺失或非法值报错，无默认值：

- `none`：原价拼接。
- `ratio_back_adjust`：以决定日两合约收盘之比 `close_new / close_old` 逐次乘到此前 OHLC。
- `diff_back_adjust`：以决定日两收盘之差 `close_new - close_old` 逐次加到此前 OHLC。

收盘源遵循上述日频优先/分钟收盘规则。只调 open/high/low/close，
volume、money、open_interest、factor 不变。锚定本次请求 end，只使用到 end 已生效的
切换；end 之后的决定、价格与切换不能影响请求内任何值。这与日频 PIT 的端点锚定原则
一致，未授权全历史锚定。回归用增删未来切换证明逐值不变。

`BarReturnLabeler(..., continuous_source=source)` 检测 `contract_id` 后，
按每个信号的起点合约重新读取原始价格，整个前瞻窗口保持该合约；既不使用调整后价格，
也不使用下一合约补洞。原合约窗口内缺 Bar/不完整桶为 `missing_bar`。
缺少原始 source 时拒绝给连续视图打标签。默认仍是 `same_session`；连续入口可显式选
`boundary_policy='cross_session'` 检查跨 Session、跨换月窗口，旧入口不接受此选项。

因子使用 `factor_bridge.continuous_factor_input(source, configuration)`，配置须含
`product_root/frequency/start/end/adjust`。时序 Harness 接受以下配置替代 bars_csv，
只读开库、用同一原始合约标签管线评估，把库文件 SHA256 绑定到结果身份：

```json
{
  "analysis_mode": "time_series",
  "continuous_futures": {
    "db": "work/NERE-743/input.duckdb", "product_root": "CU", "frequency": "5min",
    "start": "2026-01-06", "end": "2026-01-30", "view": "ETH", "adjust": "none"
  },
  "factors_csv": "work/NERE-743/factors.csv", "horizons": [5],
  "boundary_policy": "same_session"
}
```

旧原始具体合约 CSV/FactorBase 入口沿用 `price_adjustment='none'` 元数据契约。
本次仅更新 legacy cache 映射中 `bar_catalog.py` 的校验值：该文件唯一变化是写入时增列，
旧目录解析与旧价格读取相同；baseline 身份不变，旧日频缓存字节契约由回归验证。

本契约定义阶段 2–5 共用的数据结构与结构校验，入口为
`backtest_and_analysis.time_series`。契约与 CSV 层依赖标准库、NumPy 与 pandas，
日历由 `utils.session_calendar.SessionCalendar`（PMC + 版本化修正）提供。
CSV 读取仅加载调用方指定文件，不执行研究任务、不修改数据库或研究状态。

## 新旧模块边界

`AlphalensRunner`、`TimeSeriesEvaluator` 与 `BarCrossSectionalEvaluator` 是
[统一评测引擎](EVALUATION_ENGINE.md)的兼容入口。`ForwardReturnsCalculator`
在统一标签层选择日频收盘到收盘约定，Bar 使用开盘到收盘约定。股票日频
`TradingCalendar` 沿用原接口与冻结 fingerprint，见[交易日历](TRADING_CALENDAR.md)。
评测入口显式指定 `analysis_mode`。时序研究逐品种沿时间计算 IC，截面研究
在同一信号时刻沿资产计算 IC。

## Bar、因子值与时间

`Bar` 为 frozen dataclass，字段为 `instrument_id, timestamp_utc, frequency,
open, high, low, close, volume, source, data_version`。
`timestamp_utc` 始终表示 **Bar 开始时间**；无日历的 `Bar.bar_end_utc` 为名义宽度。
频率统一由 `utils.time_axis.Frequency` 解析，接受 `1/5/15/30/60min` 及其
`m` 别名、`1h`、`daily/1d`。原始拼写保留在 Bar 键中。
实际结束和可知时间必须取 `BarSchedule`，尤其是跨休息、段末不足整桶与 Session 日频。
旧 Bar 结构契约仍通过 `Frequency.contract_duration` 接受任意正 Timedelta（如 `1ns`）；
这不表示对应宽度支持交易网格，网格解析仍使用上述显式别名集合。
资产、来源和数据版本必须为非空字符串；OHLCV 必须为实数（不接受布尔值或数字字符串）；
成交量必须有限且非负。结构错误抛 `ContractValidationError(ValueError)`。

标量时间接受 `datetime` 或 `pd.Timestamp`，必须携带 UTC 时区；
接受 UTC 及其显式别名 `Etc/UTC`。naive、NaT、时间字符串、非 UTC 时区均拒绝，
即使当地时间恰好是零偏移也不自动转换。
表级时间列须已解析为 `datetime64[任意精度, UTC]`（同样接受 `Etc/UTC`），且无 NaT。
CSV 读取器负责显式解析输入，契约校验函数不隐式转换任何列。

价格异常在构造与表级结构校验时保留，通过 `Bar.has_valid_prices` 和
`invalid_price_mask(df)` 暴露。异常包括非有限值、任意价格 ≤ 0、
`high < max(open, close)`、`low > min(open, close)` 或 `low > high`。
下游将这些异常映射为 `invalid_price`，不通过删除原始行来隐藏异常。
`invalid_price_mask` 返回与输入行索引一致的布尔 Series，True 表示异常。

`FactorValue(timestamp_utc, instrument_id, factor_value)` 也是 frozen dataclass。
因子值允许 NaN；缺行或 NaN 由后续标签模块记为 `factor_missing`。±inf 与非实数类型拒绝。

## 表级校验

| 函数 | 必须包含的列 | 唯一键与排序顺序 |
|---|---|---|
| `validate_bar_frame` | `Bar` 的全部字段 | `(instrument_id, timestamp_utc, frequency)` |
| `validate_factor_frame` | `FactorValue` 的全部字段 | `(timestamp_utc, instrument_id)` |

两者检查完整列、字段类型、UTC 时间与唯一键。重复键抛
`DuplicateKeyError(ContractValidationError)`，错误信息列出重复键值。
频率是 Bar 键的一部分，原始频率字符串保留，不进行单位或拼写归一化。
未排序输入按键稳定排序，返回独立副本；值、行数、列、原行索引及额外列均保留，
不修改调用方 DataFrame。实现使用列级向量运算，不逐行构造记录。
空表须保留正确列类型；数值列采用实数数值 dtype，
价格/因子列支持 nullable 数值缺失，成交量不允许缺失。

## 标准 CSV、视图与 Bar 可用性

`read_bar_csv(path)` / `read_factor_csv(path)` 从包入口导出，同时支持单标的与多标的
分钟级输入，可供 `time_series` / `cross_sectional` 两种分析模式共用。
CSV 必须包含上述表级必需列；额外列保留。时间文本采用 ISO 日期及时间，显式携带
`Z`、`+00:00` 或 `+0000`，支持不同写法混合及纳秒精度。naive、非零偏移、
空值和非法日期统一抛 `ContractValidationError`，不将非 UTC 输入自动换算。
解析结果固定为 `datetime64[ns, UTC]`，始终表示 Bar 开始时间。
数值列由 CSV 解析器识别类型，布尔值不能当数值使用；资产代码等文本列按字符串读取，
保留 `000001` 的前导零。空表的必需数值列为 `float64`，文本列为 string。
随后复用 `validate_bar_frame` / `validate_factor_frame`，按唯一键稳定排序，
重置 `RangeIndex(0, len(frame))`。重复键报错列出具体键值；坏价格与 NaN 因子原样保留。

`filter_bars_by_view(bars, calendar, view='ETH')` 按 `SessionCalendar.session_at`
筛选 Bar 开始时间，ETH/RTH 分段完全来自日历。传入的日历必须适用于表内全部标的，
跨市场输入先按日历分组。返回表重新排序/重置索引，保留调用方 attrs，不修改原表。
读取 Bar CSV 时（或无元数据的表首次筛选时）即记录原始覆盖，**先于视图筛选**：

| attrs 键 | 含义 |
|---|---|
| `raw_coverage_start_utc` / `raw_coverage_end_utc` | 原始表最早/最晚 Bar 开始时间，空表为 NaT |
| `raw_coverage_by_instrument` | 标的 → `(最早时间, 最晚时间)` |
| `raw_coverage_by_key` | `(标的, 原始频率字符串)` → `(最早时间, 最晚时间)` |
| `session_view` / `calendar_hash` | 已筛选的视图及日历身份 |

覆盖取时间极值，与文件物理行顺序无关；RTH 尾部或全部 Bar 被筛掉时仍保留原覆盖。
重复同视图筛选保留元数据；更换视图或日历须从原始表开始。改变原始数据行后，调用方
必须重新读取或清除旧 attrs，不能把旧覆盖边界当作新数据集边界。

`BarAvailability(bars, calendar, view='ETH')` 快照观测键与原始覆盖，
可接受原始表或同视图/日历的筛选表。供后续标签器调用
`status(instrument_id, timestamp_utc, frequency)`：

| 返回值 | 判定 |
|---|---|
| `market_closed` | 日历所选视图无有效时段，无应有 Bar（含维护、午休、节假日、提前收盘后） |
| `observed` | 该日历网格的 Bar 已存在，价格异常不影响存在性 |
| `missing_bar` | 应有 Bar 未观测到，时间 ≤ 该标的/频率的原始覆盖末端 |
| `insufficient_history` | 应有 Bar 超出原始覆盖末端，或该标的/频率完全无输入 |

查询时间必须是 UTC；时段内不在 `expected_bar_opens` 网格的时间抛
`ContractValidationError`，日历覆盖越界继续抛 `CalendarCoverageError`。
先判休市再判数据尾部，不从实际行情反推会话，也不补价、插值或顺延。
此接口只报告单根 Bar 的可用性；标签跨会话和多原因优先级仍由阶段 3 处理。

## DuckDB 分钟数据接入（阶段 5.1）

`data_pipeline.loaders.market_bar_importer.import_market_bars(connection, bars, metadata)`
仅使用调用方显式传入的连接，不打开默认数据库。元数据和 Bar 物理表在一个
`import_transaction` 中建表、插入并推进缓存 revision；重复 Bar 主键或任一步失败
全部回滚。既有品种元数据须完全一致，不允许改变历史 Bar 的解释；新增批次为追加，
不覆盖旧键。价格异常保留给下游 `invalid_price` 判断。

阶段 6.2 的存储、目录与读取规则见下节。阶段 5 的 `market_bars` 保留为兼容快照，
以支持既有调用方直接 SQL 修改旧表；新管线不依赖该快照，旧快照上的修改不传播到新表。
旧 Python importer 的 `legacy_compatibility=True` 保留此行为；国内期货旧调用未指定
策略时只写旧表和元数据，不写分频率表、目录或质量表。新 CLI 固定使用
`legacy_compatibility=False`，不写旧快照，缺少期货策略即报错；`derive` 也强制策略。

`instrument_metadata` 主键为 `instrument_id`，包含 `asset_class, exchange, timezone,
contract_expiry, price_adjustment, point_value, rth_reference_calendar`。
另有可空列 `calendar_id, product_root, contract_month`（月份为 `YYYY-MM`）。
导入器在显式连接的事务中用幂等 `ALTER TABLE ADD COLUMN IF NOT EXISTS` 补列；
旧行保持 null，reader 也兼容尚未迁移的旧表。国内期货按显式 `calendar_id` 或
`config/futures_calendar_map.json` 的品种根解析，未知品种/日历报错，不能回退交易所日盘。
股票 `point_value` 缺列时默认为 1；期货须声明到期月份/日期、正乘数，复权只能为
`none`，Bar 的 `factor` 仅可为 null 或 1。原始导入不接收连续/主力代码；
连续派生读取见阶段 6.4a。
`market_bars` 的主键为 `(instrument_id, timestamp_utc, frequency)`，时间以
TIMESTAMPTZ 存储 UTC Bar 开始时间；除标准 Bar 字段外保留 `money, factor`。

`ExternalBarSourceSpec` 与 `adapt_external_bars(data, spec, metadata=..., frequency=...,
source=..., data_version=..., instrument_id=...)` 接受 DataFrame 或 CSV。以下语义
缺失即报错：`timestamp_anchor`（bar_start/bar_end）、`source_timezone`（IANA）、
`volume_mode`（interval/cumulative_session）、源列名到标准列名的 `column_mapping`。
映射必须覆盖 `timestamp, open, high, low, close, volume`；品种可映射为
`instrument_id` 或以参数指定。bar_end 减一个频率；naive 仅按显式非 UTC 时区
本地化，DST 歧义/不存在时间拒绝。声明 UTC 时输入必须显式带零偏移；带偏移时间
须与声明时区在该时点的偏移一致。累计量仅在 `(instrument_id, session_id)` 内
按时间差分，首根取自身，时段内倒退、缺量或网格缺口均拒绝；跨自然日夜盘仍属
同一 session。转换结果 attrs 标记 `external_bar_conversion=True` 及规范化语义，
重复转换标记和已满足完整 Bar schema 的输入均拒绝。导出/重读自定义 CSV 时
调用方须保存转换语义；attrs 不是 CSV 格式的一部分。

`utils.market_bar_reader.MarketBarReader(connection=...)` 借用连接；
`MarketBarReader(duckdb_path=...)` 只读打开文件并支持 context manager。
`read_bars(instrument_ids, frequency, start_utc=None, end_utc=None, view='ETH',
calendar=None)` 的时间上下界均包含端点，按 UTC Bar 开始时间筛选。
默认从品种元数据推导日历（CME → CME Globex Equity，XSHE → XSHG，其他 exchange
使用同名 PMC 日历），参考市场来自 `rth_reference_calendar`；其他产品可显式传日历。
推导范围按原始数据的当地日期前补 1 天、后补 7 天，覆盖夜盘归属与标签边界。
`get_calendars(bars)` 从 raw coverage 复建相同的日历；`get_instrument_metadata(ids)`
提供原始元数据查询。

查询后先记录四个 `raw_coverage_*` attrs，再按各品种日历筛选 ETH/RTH。
相同日历使用单一 `calendar_hash`，多日历时该属性为 `{instrument_id: hash}`，
标签器按各品种分别验证，不能把跨市场数据绑定至单一错误日历。
`build_availability(bars, view=...)` 返回单品种的 `BarAvailability`，多品种返回
`{instrument_id: BarAvailability}`。`read_market_bars` 为便捷读取函数。

旧表入口为 `read_legacy_price_table(source_table, instrument_ids, ...)`，或
`read_bars(..., source_table=...)`，均只查询、不迁移、不改结构、不复权。
`price_data_5m` 的清晰兼容预设为 `legacy_tz='Asia/Shanghai'`、
`legacy_timestamp_anchor='bar_end'`；其他来源必须按真实语义覆盖参数。
`price_data_daily` 以 `1d` 标准结构返回，UTC 时间代表本地午夜的日期标识，
不作日内 ETH/RTH 筛选，不用于分钟标签器。两者保留 `money/factor` 等额外列；
`data_version` 默认 `legacy_unversioned`，冻结研究须显式传入版本。
现有 `DuckDBAdapter.fetch_price_data` 不变；FactorBase 双模式接口见下文阶段 5.2。

## 阶段 6.2：频率目录、直取、离线构建与质量

`Frequency.table_name` 唯一决定 `market_bars_1min/5min/15min/30min/60min/1d` 名称。
新 CLI 向各频率物理表写入；Python 兼容入口在策略允许时同时保存上述阶段 5 快照。
派生写入仅影响目标频率表。
标准列外增加 `n_source_bars, n_expected_bars, is_complete`。原始导入每根为 1/1/true；
派生计数按源频率槽位，短于名义宽度但符合日历的段末桶仍可 complete。

`utils.bar_catalog.BarCatalog` 读取配置及库内 `bar_catalog`，初始化和解析均无 DDL。
配置 `price_tables` 自动声明股票原生 daily/5m；daily 时间语义为当地午夜日期标识与
PIT 复权列，5m 为上海时区 bar_end。登记不要求创建数据库或改写价格表。
数据库条目按数据集、频率、表、品种保存 native/derived、源频率/版本、策略、
calendar_hash、build_id、代码版本、覆盖 Session 范围与扫描状态。同频率原生优先，
同级多表报歧义；股票配置不作用于期货。旧表只有实际存在时才参与连接上的解析。

`utils.bar_source.BarSource(connection=... | duckdb_path=...)` 借用连接或只读打开显式文件。
`read(instruments, frequency, start, end, view)` 接受含端点 UTC 开始时间，或含端点的
Session 日期。按目录直接 SQL 取表，旧 5m 只做时间列平移；不读取更细频率、不聚合。
频率不存在或覆盖不足抛 `MissingBarDataError(ValueError)`，消息包含 frequency、
可用频率与 `build-bars --db ...` 建议。`attrs['build_ids']` 按品种与 Session 绑定构建身份。
旧 daily 保留当地午夜日期标识，不解释为可执行的日内开始时间。
start/end 可各自省略；旧表的每个日期边界独立按上海当地日期核验覆盖。
跨品种合并时保留视图筛选前的四个 raw_coverage 属性，RTH 全空也不丢失原始覆盖；
真正没有原始行时全局边界为 NaT。重建日历和标签使用同一份原始覆盖。

```bash
python3 -m tools.agent_harness.cli build-bars --db work/NERE-728/input.duckdb \
  --source 5min --targets 15min,60min --instruments A --start 2026-01-06 --end 2026-01-08
python3 -m tools.agent_harness.cli audit-bars --db work/NERE-728/input.duckdb
python3 -m tools.agent_harness.cli import-bars --db work/NERE-728/input.duckdb \
  --bars canonical.csv --metadata instruments.json --derive 15min,60min
```

`import-bars` 的 metadata 为 JSON 记录数组；CSV 必须已符合标准 UTC Bar 契约。
Python importer 也接受 `derive=(...)`；先提交原始导入，再执行离线构建，派生失败不撤销
已成功的原始导入。build/audit/import 全部要求显式 `--db`，不选择默认生产库。
国内期货必须加 `--bucket-strategy segment_anchored|session_clock`，不猜默认策略。
所有源槽位和目标边界均取 `BarSchedule`，每个目标边界必须落在源边界上。
SQL GROUP BY 完成 OHLC、volume/money 和 Session factor 归约，Session factor 不一致报错。
原生目标拒绝派生，`--check-only` 只读输出收盘差、成交量差和缺失日，不写数据表。

`bar_build_log` 以频率、品种、Session 保存源内容哈希、映射/策略/代码/日历身份、版本
及 build_id；完全相同的重跑不写入，源变化只重建相关 Session、版本加一。
不同 Session 可以来自不同构建，研究使用读出 attrs 的逐 Session build_id 绑定。
当前表保存最新构建，旧 build_id 是追踪身份，不提供历史版本 Bar 的自动回放。

`bar_quality` 键为 `(frequency, instrument_id, session_id)`，含应有/实有数量、
`complete|partial|missing_session`、不完整桶数及 UTC Session 包络。
importer 扫描受影响 Session，build 扫描新构建 Session，audit 可扫描旧表；
旧日频表审计支持 start/end 日期子区间：先筛选范围，再映射开市 Session。
范围内若有非交易日记录，审计在写入前明确报错；范围外的记录不参与该次审计。
数据缺口不补行、不补价。只读登记的旧表为 `unscanned`，每张表仅记一次 info。
每次 `BarSource.read` 覆盖质量标记范围时发一条合并的 `BarDataQualityWarning`
并记录 warning 日志，列出频率、品种、缺失 Session、不完整桶数和示例日期。
`BarAvailability(..., bucket_strategy=...)` 将 `is_complete=False` 当作 `missing_bar`。
Bar 评估要求内嵌因子的当前信号桶完整，包括 FactorBase 从同批 Bar 计算的因子。
连续合约的内嵌信号也读取输入快照的完整性，持有窗口仍检查具体合约行情。
独立因子由调用方按实际发布时间对齐，可使用不完整的行情桶作为信号键。
信号之后到退出为止的缺桶或不完整桶只影响所涉及的持有期标签。
没有完整性列时沿用观测到的 Bar。
所有命令仅获准在隔离的合成库中验证；生产库、旧原生价格表不在写入范围。

## 剔除原因与固定优先级

`ExclusionReason` 是字符串 Enum，固定优先级如下：

| 优先级 | 枚举值 | 口径 |
|---|---|---|
| 1 | `factor_missing` | 因子缺行或值为 NaN |
| 2 | `insufficient_history` | 标签所需的未来 Bar 超出数据覆盖末端，即“末尾数据不足” |
| 3 | `session_boundary` | 标签违反所选会话/视图边界 |
| 4 | `missing_bar` | 数据覆盖范围内缺少应有 Bar |
| 5 | `invalid_price` | 参与计算的 Bar 存在价格异常 |

数据覆盖范围内的缺口必须记为 `missing_bar`，不记为 `insufficient_history`。
Bar 缺失/覆盖判断由上述接口提供，标签级原因合并由 `BarReturnLabeler` 完成。
每条记录只能计一个主原因，多个条件同时为真时取最高优先级。

`primary_exclusion_reason(flags)` 接受原因到布尔值的映射，未提供的原因视为 False，
返回 `ExclusionReason` 或 None。
`primary_exclusion_reasons(flags_df)` 要求五个以枚举值命名的布尔列（支持 Enum 列名），
不允许缺列、空值、整数代替布尔值；返回名为 `exclusion_reason` 的 Series，
保留索引，每行是枚举值或 None。`EXCLUSION_PRIORITY` 暴露固定顺序。

## 标签记录与单位

`LabelRecord` 为 frozen dataclass，包含以下字段：

| 字段 | 含义 |
|---|---|
| `instrument_id, frequency` | 资产与 Bar 频率 |
| `entry_offset_bars, horizon_bars` | 入场偏移 k ≥ 1，持有长度 h ≥ 1，均为整数 |
| `signal_timestamp` | 信号 Bar t 的开始时间 |
| `signal_available_at` | 信号 Bar t 的结束时间，即可知时刻 |
| `entry_timestamp` | Bar t+k 的开始时间 |
| `exit_bar_timestamp` | Bar t+k+h−1 的开始时间 |
| `exit_timestamp` | 实际退出时间，为退出 Bar 的结束时间 |
| `entry_price, exit_price` | 入场、退出价格 |
| `price_change_points` | 退出价 − 入场价，单位为价格点 |
| `return_pct` | `(exit_price / entry_price − 1) × 100`，单位为百分比 |
| `status, exclusion_reason` | `valid` 或 `excluded`，以及唯一主剔除原因 |

必须满足 `valid ⇔ exclusion_reason is None`；excluded 的原因必须是
`ExclusionReason` 实例。所有已提供时间必须为 UTC，并满足以下顺序。
无日历的构造保留原名义宽度等式；使用 `LabelRecord.from_schedule(schedule, **fields)` 时，
开始时间须命中槽位，两个结束时间改为校验槽位真实结束时间。
`schedule` 仅为该构造方法的校验参数，不进入 dataclass 字段；标签器自动传入相同网格。

```text
signal_available_at == signal_timestamp + frequency
signal_available_at <= entry_timestamp <= exit_bar_timestamp < exit_timestamp
exit_timestamp == exit_bar_timestamp + frequency
```

valid 记录要求全部时间、价格与收益字段齐全；价格有限且 > 0，
点差及百分比收益均须与价格算术一致，绝对容差 1e-9，相对容差 0。
例如信号 14:30Z、k=1、h=5，入场 14:31Z open=101，退出 Bar 14:35Z close=106，
实际退出 14:36Z：点差是 5，`return_pct` 是 **4.9504950495…**。

excluded 记录仍须提供信号开始时间和可知时刻；入场/退出时间、价格、点差与收益
可以逐字段为 None。已提供的时间仍校验 UTC 和能够判定的顺序/等式；
已提供价格仍检查实数类型，但允许 NaN、非正值来保留异常证据。
k/h 对交易时段中应有 Bar 序号的映射由 `BarReturnLabeler` 完成，不能简单用墙上时间
乘法替代会话日历；有日历时由构造方法校验真实槽位结束。

## 按 Bar 前瞻标签

```python
from backtest_and_analysis.time_series import BarReturnLabeler

labeler = BarReturnLabeler(calendar, entry_offset_bars=1,
                          horizons=(1, 5, 15, 60), view='ETH')
per_bar_labels = labeler.label(bars, factors)
records = labeler.label_records(bars, factors, signal_timestamps=['2026-12-01T14:30Z'])
```

`calendar` 接受单个 `SessionCalendar` 或 `instrument_id -> SessionCalendar` 映射。
每个 `(instrument_id, frequency)` 独立绑定所属日历、所选视图和原始覆盖末端。
同一信号截面内其他标的的缺口、坏价格、因子缺失与尾部不足不会影响本标的；
输出可按 `instrument_id` 交给时序评估，或按 `signal_timestamp` 交给分钟级截面评估。
标签器不推断评估模式，也不计算 IC 或分组统计。

`entry_offset_bars=k` 为正整数；`horizons` 接受单个正整数或非空、不重复的正整数序列，
均拒绝布尔值。支持 `view='ETH'|'RTH'`；具体合约入口只接受
`boundary_policy='same_session'`，连续入口的显式跨 Session 模式见阶段 6.4a。
`bucket_strategy` 传给 `BarSchedule`，国内期货必须显式指定。
因子表通过 `(timestamp_utc, instrument_id)` 关联，缺行或 NaN 记 `factor_missing`。
不传因子表时读取 Bar 的 `factor_value` 列，两者均未提供时报错。

仅原始行情中位于所选视图应有网格上的 Bar 作为候选；可用 `signal_timestamps`
进一步过滤。该参数接受 UTC 时间序列或带 UTC 的时间文本，不接受 naive/NaT。
视图外信号不进入候选长表，也不计为剔除；时段内偏离频率网格的 Bar 报错。
按日历应有 Bar 序列定位入场 `t+k`、退出 Bar `t+k+h-1`，不对行情行号做 shift。
改变 k 仅平移入场/退出位置，不改变持有 h 根 Bar 的长度。
午休、同一交易日会话内的中断不计 Bar；ETH 可跨越 RTH 开收盘，不能跨交易日会话。

尾部不足通过底层完整 ETH 网格的 `t+k+h-1` 与各标的/频率原始覆盖末端比较，
超出日历网格也记尾部不足；随后按所选视图判断是否跨会话。故 RTH 收盘后仍有原始
行情时可以是 `session_boundary`，而非 `insufficient_history`。
行情内 `[t+1, t+k+h-1]` 任一应有 Bar 缺失即触发 `missing_bar`；
持有窗口 `[t+k, t+k+h-1]` 任一 OHLC 异常即触发 `invalid_price`，包括中间 Bar。
缺失不补价、不缩短持有期、不跳过缺口顺延；并发原因按上述固定优先级合并。

实现按标的/频率、会话和持有长度分组，候选窗口用索引映射与前缀和向量计算。
`label` 返回全部 15 个 `LabelRecord` 字段的长表，按标的、频率、信号时间、持有长度排序。
剔除行保留信号与可知时间，入场/退出时间置 `NaT`，价格/点差/收益置 `NaN`；
`label_records` 仅在序列化时逐条构造契约记录，将这些空值转为 `None`。
输入表及 attrs 均不修改；同视图/日历已筛选表须保留 `raw_coverage_by_key`，
更换视图或日历必须从原始行情重新开始。

## 按 Bar 评估与 CLI

`TimeSeriesEvaluator(analysis_mode='time_series')` 按标的、频率、持有长度独立计算
Spearman IC（平均秩处理同值），再汇总各资产 IC 的均值和中位数。
标准 IC 表的 `non_overlapping_ic` 先按候选信号时间排序、每隔 h 根取样，再筛选有效记录；
不能先删除剔除行再取步长。默认 `min_ic_samples=3`，常数因子、常数收益及样本不足
分别报告 `constant_factor`、`constant_return`、`insufficient_samples`，IC 为空。
全样本 `qcut(..., duplicates='drop')` 不打散同值，报告实际组数，分组固定标为
`quantile_scope='full_sample_descriptive_only', descriptive_only=true`，不能作为可交易分组规则。

`BarCrossSectionalEvaluator(analysis_mode='cross_sectional')` 在同一
`(frequency, horizon_bars, signal_timestamp)` 上只用 `status='valid'` 标的算截面 IC，
沿时间等权汇总 `mean_ic, median_ic, std_ic, icir, positive_ic_ratio`。
标准差使用 `ddof=1`，少于两个有效截面或标准差为零时 ICIR 为空。
默认 `min_cross_section_size=3`；标的不足、常数因子或常数收益报告对应不可算状态。
每截面的候选/有效/剔除数与状态保存在评估摘要的 `cross_sections`。
分位组按截面先计算、再等权汇总，范围为 `cross_sectional_time_aggregated`。
重复值导致某些截面组数减少时，高分位组只汇总实际存在该组的截面；消费时同时检查
`cross_section_count`、`min_actual_quantiles`、`max_actual_quantiles`，不能假设各组覆盖相同。
轻量与推断运行的标准表均报告上述全样本描述性统计。启用推断后，同次摘要也报告应有网格上的统计与候选证据。

```bash
python3 -m tools.agent_harness.cli time-series-analysis --config work/minute-bars/nq.json
# 同一 Bar 配置也可经 single-factor-analysis --config 分流
```

配置为 JSON 对象，可直接使用顶层字段，或放在 `time_series_analysis` 对象中
（块内同名字段优先；块内选择 `calendar` 或 `calendar_by_instrument` 时也替换顶层的
另一种日历配置）。NQ 合成行情示例：

```json
{
  "analysis_mode": "time_series",
  "bars_csv": "work/minute-bars/nq.csv",
  "calendar": "NQ",
  "start_date": "2026-12-01",
  "end_date": "2026-12-02",
  "view": "ETH",
  "horizons": [1, 5, 15, 60],
  "entry_offset_bars": 1,
  "n_quantiles": 5,
  "min_ic_samples": 3
}
```

行情内提供 `factor_value` 即可；独立因子文件用 `factors_csv` 指定，按
`(timestamp_utc, instrument_id)` 关联，也接受 `(timestamp, symbol)` 别名。
同时提供独立因子文件时以该文件为准，缺因子不回填行情内的值。
每个外部值须在其对应信号 Bar 的实际结束时刻已经可得。评估入口拒绝携带
`signal_available_at[_utc]`、`available_at[_utc]`、`published_at[_utc]` 或
`high_tf_bar_end` 的待对齐因子表。按品种将原始特征的 Bar 结束和发布时间提供给
`backtest_and_analysis.evaluation.split.align_released_features`，以下游日历
的实际信号 Bar 结束时间查询可用值，再按信号 Bar 开始时间和品种提交因子值。
横截面将 `analysis_mode` 改为 `cross_sectional`，可配置 `min_cross_section_size`。
多市场可用 `"calendar_by_instrument": {"NQZ26": "NQ", "000001": "XSHG"}`
代替 `calendar`；每个标的必须有映射，共用声明的 `start_date/end_date` 覆盖。
NQ/ES 要求显式覆盖日期；XSHG 未声明时沿用其已验证默认覆盖。
ETH/RTH 由日历决定，视图外信号不进入候选或剔除计数。日历越界拒绝执行。

`--bars-csv, --factors-csv, --analysis-mode, --view, --calendar, --horizons 1,5,15,60,
--entry-offset-bars, --n-quantiles` 覆盖配置；`--calendar` 覆盖整个标的映射。
CSV 路径及配置 `output_dir` 相对仓库根目录解析（不相对配置文件目录）；
命令行 `--output-dir` 相对调用方工作目录解析。输入仅限调用方指定的本地 CSV。
`single-factor-analysis` 在配置中出现 `bars_csv`、存在 `time_series_analysis` 块、
顶层显式声明 `analysis_mode='time_series'`，或命令行指定 `--bars-csv` /
`--analysis-mode time_series` 时分流。缺少 CSV 的显式 Bar 请求在该分支报前置校验失败；
空块或非法块也由 Bar 分支校验，不转交日频执行。未声明这些 Bar 意图的配置继续执行
原日频入口，包括仅含 `analysis_mode='cross_sectional'` 的配置。
Bar 模式始终必须显式指定；缺失/非法分别返回 `missing_analysis_mode` /
`invalid_analysis_mode`（退出码 2），绝不按标的数推断。

复用 Harness 的 `--run-id, --output-root, --json-out, --dry-run, --fake-result,
--timeout-sec`。dry-run/fake 不读 CSV、不清理评估产物；真实执行受子进程超时约束。
默认评估目录为本次 attempt 的 `evaluation/`；`--output-dir`（或配置 `output_dir`）
可覆盖，若与 Harness attempt 或其运行级目录相同，则仍使用 attempt 的 `evaluation/`，
避免覆盖任一级包装器摘要。目标为普通文件或完成标记为目录时返回结构化前置错误
（退出码 2），不删除用户文件或递归删除目录；清理异常记录在 `error.details` 中。
预演、模拟及实际执行均在写摘要前校验 `--json-out`。指向评估产物时返回退出码 2，
保留已有产物与完成标记。`diagnostics_report.csv`、`diagnostics_report.md` 和
已退役的旁路文件名均作为保留路径参与碰撞校验。若输出目录含旧旁路文件，选用新目录
重跑，以保留原有历史证据。
默认 Harness 摘要和运行级导航遵循 `docs/PUBLIC_SCOPE.md 的 Harness 输出布局`。

| 标准文件 | 内容 / Harness artifact_type |
|---|---|
| `per_bar_labels.csv` | 全部候选 LabelRecord 字段和因子值 / `per_bar_labels` |
| `ic_by_horizon.csv` | 时序逐资产及汇总 IC，或沿时间汇总的截面 IC / `ic_by_horizon` |
| `quantile_stats.csv` | 分位组样本数、平均/中位收益、正收益比例及实际组数 / `quantile_stats` |
| `summary.json` | 配置、身份、标准统计、切分推断、门控与候选账本 / `evaluation_summary` |

CSV 时间为 ISO8601 UTC `Z`，JSON 不可算数值为 null。四文件各自登记绝对路径与
SHA256，核心摘要复制到 Harness `metrics`。评估 `summary.json` 最后原子发布，作为唯一
完成标记；与 Harness 顶层 `summary.json` 分离。通过路径校验后，真实重试在读取 CSV 前
清理显式目标的旧完成标记。即使配置解析失败，也会清理 `--output-dir` 显式指定的标记。
执行失败或超时会撤销完成标记。旧 CSV 可留作诊断，须配合当前完成标记消费。

### 时序深度诊断报告

使用 `time-series-analysis --config <path> --diagnostics-report`，或设置 JSON 布尔字段
`"diagnostics_report": true`。该选项适用于 `time_series`，诊断直接复用当次评估的应有网格统计。
全样本报告采用 `time_series_diagnostics.v1`。配置训练/OOS 切分后采用
`time_series_diagnostics.v2`。评估 `summary.json` 和 Harness
`metrics.diagnostics_report` 携带同一报告，切分诊断 CSV 增加
`split`、`split_purged_count` 两列，Markdown 分别展示训练和 OOS。
候选和剔除计数基于清除前的标签，清除数单列报告。`diagnostics_report.csv`
和 `diagnostics_report.md` 分别以 `diagnostics_report_csv` 和
`diagnostics_report_markdown` 登记 SHA256，在完成标记之前发布。
CSV 的 `scope` 为 `overall`、`bucket` 或 `offset`，
`index` 依次为空、从 1 开始的桶号或从 0 开始的网格偏移。
JSON 用 null、CSV 用空字段、Markdown 用破折号表示不可计算的数值。
显式 `--results-db` 持久化主契约标准统计及推断字段，诊断报告作为发布产物保留。

对每组标的和频率，所选视图的应有网格从首个到末个候选信号，包含中间缺 Bar 的槽位。
各持有期在该网格上报告全部 `0..h-1` 个偏移的有效样本数、Spearman IC 与状态，
并核对实际持仓区间在各偏移内互不重叠。候选及剔除计数沿用标准 LabelRecord 分类。
块 bootstrap 在网格上重采样，按有效配对重新排名。缺省自动选块，各持有期以自身的 `h` 为下界。
可通过 `bootstrap.block_length` 显式配置，解释区间时以报告逐持有期记录的
`bootstrap.block_length` 实际值为准。未配置切分时，即使指定了块长，全样本
诊断仍采用 `time_series_diagnostics.v1`。其他默认值为种子 0、999 次抽样、
至少 900 次有效重复和双侧 95% 区间。
全局及各描述性分位桶均报告方向命中数、分母、上涨基准率、提升，
以及条件收益 P5/P25/P50/P75/P95 和四个尾部均值，收益单位为百分数。
全样本 v1 报告使用描述性分桶。训练/OOS 验收见[统一评测引擎第 16 节](EVALUATION_ENGINE.md#16-s7-时序诊断与-nq-1min-验收)。
S7 的 NQ 1min 报告标记 `real_market_data_used=false`。真实 NQ 1min 在外置盘挂载后补跑。

### 统一引擎参数

[`evaluate_bars()`](../backtest_and_analysis/time_series/evaluator.py) 接收原始 Bar、
可选因子长表、`BarReturnLabeler` 和显式 `analysis_mode`。通过标签器和交易日历构造
应有网格、信号可用时刻，以及逐持有期的 `entry_timestamps` 和 `exit_timestamps`，
再交给 [`EvaluationEngine.evaluate()`](../backtest_and_analysis/evaluation/engine.py)。
推断按时序的 `(frequency, instrument_id)` 或截面的 `frequency` 分组，
每组只调用一次引擎，标准表与切分统计由该次返回值组装。轻量运行可直接基于观察样本
出表；预计算标签可交给对应模式的描述性评估器。

| 参数 | Python API 与 JSON 配置语义 |
| --- | --- |
| `split` | `train_end` 与 `oos_start` 两个带时区的边界。Python API 也接受 `SplitConfig`。 |
| `bootstrap` | 仅时序。支持 `block_length`（缺省 `"auto"`，可指定正整数）、`n_boot`（默认 999）、`seed`、`min_valid_replicates`（默认 900）和区间 `alpha`。 |
| `hac_lag` | 仅截面。缺省 `"auto"`，可指定非负整数以固定 Bartlett HAC 窗口。可估区间的各期滞后至少为 `h - 1`。 |
| `direction`、`standardization` | 方向缺省为 1。可选标准化仅用于时序训练/OOS 切分。 |
| `calibration_scenario` | 显式声明依赖场景，供区间按方法匹配已登记的校准依据。自适应 HAC 的独立场景名为 `iid`，固定 HAC 使用 `independent_null`。 |
| `trial_selection` | 配置候选族的 `method` 和 `alpha`，默认 `holm`、0.05。 |
| `factor_id` | 账本中的候选因子标识，默认为 `factor_value`。单独设置此字段只指定候选标识。 |

配置中的 `split`、`bootstrap`、`hac_lag`、`direction`、`standardization`、
`calibration_scenario`、`trial_selection` 中任一项为非 null 即请求应有网格推断和账本。
有效的空对象（如 `"trial_selection": {}`）采用该项默认设置。Python 调用方也可传入
预登记的 `trial_family`。仅请求诊断（Python `diagnostics=True` 或 Harness
`diagnostics_report`）时有 `full` 推断统计，`trial_ledger` 为 `null`。

截面按频率和信号实际可用时刻对齐，于每个截面使用 `per_cross_section` 分桶。
时序全样本使用 `full_sample_descriptive_only`。时序切分采用 `fitted_on_train`，
每个频率和标的在训练期冻结分位边界、方向及可选标准化参数，训练和 OOS
复用同一 `fit_identity`。样本外极端值进入端点桶，空桶保留编号。

切分按日历上的实际信号可用时刻对齐。不完整的当前 Bar 上的内嵌因子以缺失值输出，
并排除在训练拟合和截面分桶之外。当前可得的因子即使缺少某个未来持有期的收益，
仍参与信号分桶和训练拟合。训练候选的计划退出晚于首个 OOS 应有信号的
`signal_available_at` 时按持有期清除。退出恰好等于边界时保留。即便退出价缺失，
跨界的计划退出仍计入清除数。各评估面板的训练与 OOS 段都须包含应有信号。
时序区间使用持仓时间验证各非重叠相位，块重采样保留网格缺口并在每次复制中重排
有效配对。自动块长须覆盖当前持有期的 `h`。显式整数块长须满足所有持有期下界。切分与推断细节见
[统一评测引擎](EVALUATION_ENGINE.md#14-训练与样本外评测接口)。

### bar_evaluation.v2 产物契约

[`inference.py`](../backtest_and_analysis/time_series/inference.py) 组装推断统计和候选族。
[`artifacts.py`](../backtest_and_analysis/time_series/artifacts.py) 始终将
`result_schema_version="bar_evaluation.v2"` 写入评估 `summary.json` 和 `result_identity`。
Harness 复制摘要字段至 `metrics`（配置见冻结的 `config/resolved.json`）。

- 顶层含 `analysis_mode`、`frequency="bar"`、`return_unit="percent"`、
  `config`、`data_identity`、`calendar_identity`、`result_identity`、
  `ic_by_horizon`、`quantile_stats`、`cross_sections`、`exclusion_summary`。
  `interval_usability` 是按频率、持有期和标的标注的标准 IC 表可用性。请求诊断时
  摘要还包含 `diagnostics_report`。
- `evaluate_bars()` 在 `calendar_identity` 中记录分桶策略，以及单个或逐标的日历的
  覆盖日期和 RTH 参考日历。即使 Python 发布时省略 `config`，入库仍可在日历哈希
  匹配后从标签重建应有网格，复算与缺口有关的 HAC 统计和校准观测。
  入库校验和运行表共用同一个分桶策略。旧产物未记录策略时按发布配置回退，
  与产物或其他配置项冲突的显式策略会导致拒收。
  产物记录 `null` 而发布配置显式指定策略时，入库按已核验日历和标签的实际频率
  确认 `BarSchedule` 的默认策略，并仅在配置与默认值一致时记录该值。
  标签为空时，使用摘要 `data_identity` 中通过产物哈希校验的原始输入标的与频率。
- 轻量评估的 `splits={}`、`trial_ledger=null`。推断的 `splits` 为 `full`
  或 `train` 与 `oos`。每段提供 `train_end`、
  `oos_start`、`quantile_scheme`、`fit_identity`、逐持有期 `split_purged_count`
  和 `horizons` 统计行。每行记录实际 Bar 频率、持有期和 UTC 网格边界，以及
  对应的 IC、分桶、方向和区间。时序行还含 `intervals_verified` 和各非重叠相位，
  `instrument_ic_summaries` 汇总各频率及持有期的多标的 IC 和逐相位 IC。
- `hac` 或 `bootstrap` 中的 `reason` 说明能否估计。
  `calibration_status` 默认为 `uncalibrated`。当声明的场景、估计方法、实际
  窗口、应有样本数和标称覆盖率匹配已登记依据时，报告 `calibrated_pass` 或
  `calibrated_fail` 及 `calibration_evidence`。每行的 `interval_usability`
  独立报告 `descriptive_stats_usable` 与 `significance_interpretation_usable`。
  后者还须满足区间可估计及登记校准的样本完整性、复制数等适用条件。
  证据与门控见[评测政策](EVAL_POLICY.md#unified-evaluation-engine)和
  [校准登记](EVALUATION_ENGINE.md#61-一次性合成校准)。
- UTC 时间序列化为 ISO8601，无法计算的数值以 JSON `null` 表示，保留相应原因码。

### TrialLedger 候选登记与校正

请求带账本的推断时，在评估前登记配置中的全部 `(factor_id, horizon)` 候选，切分时使用
训练期统计，全样本运行则使用 `full`。Python 调用方可在查看结果前创建包含
全部因子与持有期的 `BarTrialFamily`，再将同一实例作为 `trial_family` 随每个
因子的 `factor_id` 传入。每次只记录该因子本次已登记的持有期。共享族首次成功
记录时绑定全样本或训练窗口。复用时训练边界须按 UTC 一致。手工记录应使用独立候选族。

`trial_ledger` 提供 `method`、`alpha`、`split`、`attempt_count`、
`selection_complete` 和 `best`。逐候选记录统计量、观测数 `n_observations`
（`n_effective` 为同值旧字段）、区间、原始 `p_value`、审计校正
`adjusted_p_value`、推断校正 `inference_adjusted_p_value`、`significant` 和
`reason`。单流截面候选的原始 p 值由 HAC 区间的分布计算。`p_value_method` 记录
`two_sided_student_t_hac`（自动窗口，有效自由度）或 `two_sided_normal_hac`（固定窗口）。
显著性解释仍受校准门控。自适应 HAC 依据方法版本和可核验的网格、缺失模式、截面宽度等观测匹配
已登记适用域。自适应 Bootstrap 当前没有已登记适用域，保持 `uncalibrated`。
时序区间不提供原始 p 值，同一候选涉及多个评估流时也不自动合并原始 p 值。
未评测和缺原始 p 值的候选分别列入 `pending_candidates` 和
`missing_p_value_candidates`。整族均有原始 p 值时才执行 Holm 或 BH 校正并报告
最优候选。详见[尝试账本与多重检验](EVALUATION_ENGINE.md#62-择优的尝试账本与多重检验)。

### 身份绑定

`ResultIdentity` 为 frozen dataclass，字段如下：

| 字段 | 含义 |
|---|---|
| `config_hash` | 冻结配置的 SHA256 |
| `input_file_hashes` | 输入文件名/路径 → SHA256 映射 |
| `calendar_hash` | 冻结日历的 SHA256 |
| `code_version` | 代码版本，非空字符串 |
| `result_schema_version` | 结果结构版本，非空字符串 |

哈希字段采用 64 位小写十六进制字符串。文件映射在构造时复制为只读映射，
避免外部字典变化导致身份漂移；路径按调用方提供的原始字符串绑定。
`sha256_file(path)` 分块读取原始文件字节，无写入。
`identity_hash()` 将全部字段转为 JSON（递归键排序、紧凑分隔符、ASCII 转义），
对 UTF-8 字节计算 SHA256；输入映射插入顺序不影响结果，任一字段变化会改变身份。
CLI 的 `config_hash` 绑定合并命令行覆盖后的配置，递归排序、紧凑 JSON 编码；
`input_file_hashes` 使用解析后的绝对 CSV 路径，并在评估前后复查源文件未变化。
`code_version` 为 Git HEAD，已跟踪的计算代码有未提交改动时附加 `-dirty`；正式证据应使用
已提交版本。`calendar_hash` 来自真实日历身份（含覆盖、依赖版本和修正表），
多市场不同日历时绑定整个映射，`result_schema_version` 使用评估器当前格式版本。

## 验证

`tests/test_time_series_contracts.py` 验证输入、排序、重复键诊断、
全部 32 种剔除原因组合、标签时间、百分比单位及身份稳定性。
CSV 和标签分别由 `tests/test_time_series_csv_reader.py` 与
`tests/test_bar_return_labeler.py` 覆盖。
CLI 双模式合成演示和失败重试见 `tests/test_time_series_cli_end_to_end.py`。
入库、旧库迁移和校准门控的往返见 `tests/test_bar_evaluation_store.py`。
冻结合成数据集见 `tests/fixtures/time_series/README.md`。

## 验收测试清单

以下测试位于 `tests/test_time_series_acceptance.py`，只通过
`tests/time_series_acceptance_support.py` 调用待实现能力。
阶段 2 的 19 例、阶段 3 的 14 例及阶段 4 的 2 例均已接线并移除 xfail，真实通过。
手算预期为 `tests/fixtures/time_series/expected_*.json`，经本阶段验收后冻结；
阶段 2–4 只接线适配层并移除标记，不得修改预期，修改须经大古同意。

| 测试名（参数化场景） | 场景 | 归属阶段 | 当前状态 |
|---|---|---|---|
| `test_session_membership`（9例） | RTH开收盘、ETH维护休市及23:00下一交易日 | 2 | passed |
| `test_expected_bar_grid`（4例） | 半开小窗口、RTH开盘、ETH维护无应有Bar | 2 | passed |
| `test_eth_rth_view`（2例） | 同一输入的ETH/RTH候选集合 | 2 | passed |
| `test_csv_duplicate_keys`（2例） | Bar与因子重复键必须拒绝 | 2 | passed |
| `test_csv_unsorted_matches_sorted`（2例） | Bar与因子乱序稳定排序、逐列值及UTC解析一致 | 2 | passed |
| `test_label_scenarios[valid_five-RTH]` | 101→106、五Bar、百分比及实际退出时间 | 3 | passed |
| `test_label_scenarios[rth_boundary-RTH]` | 跨21:00 RTH收盘 | 3 | passed |
| `test_label_scenarios[eth_boundary-ETH]` | 跨22:00–23:00维护休市 | 3 | passed |
| `test_label_scenarios[missing_bar-RTH]` | 会话内缺一根，不补价/缩短/顺延 | 3 | passed |
| `test_label_scenarios[invalid_zero-RTH/invalid_nan-RTH]`（2例） | 持有期中间Bar收盘为0或NaN | 3 | passed |
| `test_label_scenarios[insufficient_history-RTH]` | 最后四根信号均超过覆盖末端 | 3 | passed |
| `test_label_scenarios[factor_nan-RTH/factor_absent-RTH]`（2例） | 价格完整但因子NaN或缺行 | 3 | passed |
| `test_label_scenarios[factor_and_gap-RTH]` | 因子缺失与缺Bar并发，仅记factor_missing | 3 | passed |
| `test_label_scenarios[tail_and_boundary-ETH]` | 末尾不足与跨会话并发，仅记insufficient_history | 3 | passed |
| `test_label_scenarios[view_difference-ETH/view_difference-RTH]`（2例） | 14:27信号ETH有效、RTH不出现在长表 | 3 | passed |
| `test_unsorted_labels_match_sorted` | 乱序输入标签逐行等于排序输入 | 3 | passed |
| `test_end_to_end_counts`（2例） | 同一夹具ETH为4有效/5剔除，RTH为1有效/5剔除 | 4 | passed |

共35个参数化验收用例：35 passed，0 xfailed（阶段2为19、阶段3为14、阶段4为2）。
夹具自检 `tests/test_time_series_fixtures.py` 的57个用例现在必须通过，不标记xfail；
它们校验SYNTHETIC身份、全部29条LabelRecord、手算收益、主原因优先级与原始异常证据。
验收使用合成行情与显式覆盖的 PMC 日历；阶段 4 已实现双模式评估与 CLI，
保持现有日频横截面模块独立。

## FactorBase 双模式接入（阶段 5.2）

阶段 6.3 的写法分流只由 `FactorFrequencyRoute.resolve` 决定，解析唯一来自
`Frequency.parse`；`data_frequency` 始终保留调用方原值。构造不触发价格读取。

| 写法类别 | 示例 | 构造 / 云端 | 本地读取 |
| --- | --- | --- | --- |
| canonical 分钟 | `1min/5min/15min/30min/60min` | 必须显式 `analysis_mode`；不加载本地配置 | BarSource，UTC Bar + Session 网格 |
| canonical Session | `1d` | 必须显式 `analysis_mode` | BarSource 的 Session panel，保留阶段 5 的 daily/PIT 计算输出 |
| legacy | `daily/1m/5m/15m/30m/60m/1h` | 不因频率校验报错；云端原写法不变 | BarSource 目录解析，保留旧 panel 的列、SQL、时区、PIT 和缓存 |
| legacy 显式模式特例 | `5m` + `analysis_mode` | 新 Bar 接口 | 与 canonical `5min` 的网格路径相同 |
| invalid | `5M/None/空串/2min` | 构造保留原值，不提前拒绝 | 读取时报包含 `frequency` 的 `ValueError` |

`1h` 属于 legacy，`30min` 属于 canonical；模式不能从标的数推断。
`DataAccessConfig.price(frequency)` 对外仍只接受配置中的旧键，未扩大别名范围。
FactorBase 的价格读取通过 `BarSource.read_panel`（兼容 panel）或 `BarSource.read`
（分钟网格）进入统一来源边界；没有读取时聚合或较细频率回退。
`read_panel` 委托原 adapter 的 SQL/Parquet/缓存查询，并且每次读取检查已登记的质量。
阶段 5 的 `market_bars` 快照可独立于新表编辑；FactorBase 显式选择
`legacy_snapshot=True` 时按标的检查目标 canonical 频率是否确有快照行，且目录项不能为
`derived`。仅命中的标的读取快照；派生表、未写快照的标的和只有旧别名行的情况
仍按目录直取规范分频率表。混合来源合并时保留各标的原始覆盖和日历属性。
普通 BarSource 调用不会启用此兼容选择，也不会绕过期货缺少分桶策略的错误。

非 Parquet panel 读取会核验配置物理表实际存在，并查询库内目录；配置中的旧表名
不遮蔽新导入的同频率表。Parquet 数据集仍按配置与原 adapter 的源策略读取。
panel 读取先校验日期和字段；目录 Bar 投影沿用 borrowed panel 的键字段剔除与去重
规则。空池返回日期类型明确的空 panel，无连接则给出包含 frequency 的连接错误。
分钟金额归一化使用 Frequency
的种类/分钟数映射到原 PIT helper 支持的频率，保留显式 paused/零金额语义；
`utils/pit.py` 与冻结云端输入不变。

旧日频缓存的源码身份在 `utils/legacy_cache_identity.json` 记录一次严格等价映射：
只有当前 FactorBase 与抽出的读取/频率/窗口依赖全部匹配已审计 SHA256 时，才映射到
已审计基线的源码摘要。其他源码改动恢复正常失效，原数据、参数、配置、PIT、
日历指纹与子类源码仍参与原 v3 键。未重写既有缓存文件或历史哈希。
聚宽构建使用 `scripts/build/cloud_runtime/` 中冻结的两份云端源码。
本地重构不会自动改变云端部署产物。

`_bars_per_day` 通过 XSHG 完整参考 Session（2026-01-05）的 `BarSchedule` 和
`Lookback(sessions=1)` 得到 240/48/16/8/4。旧 `requirements.daily.min_days` 与
`requirements.5m.min_bars` 经 `Lookback` 校验，再按同一参考 Session 换算为日数。
这是旧股票配置的完整交易日单位，不用于估计国内期货或提前收市日。

CSV 评估版本固定从代码根目录取 `git -C <root> rev-parse HEAD`，只接受根目录自身的
Git 元数据（含 worktree `.git` 文件），不向父目录搜仓库。无 Git 时依次使用
`QUANTATION_CODE_VERSION`、已安装分发包 `jq-factor-lab` 的版本、`unknown`。
结果 `result_identity.code_version_source` 记录 `git/env/package/unknown`；Git 已跟踪文件
有差异时版本附 `-dirty`。该字段作为来源说明，不改变已有 ResultIdentity 哈希字段。

`FactorBase` 新增 `analysis_mode, session_view, output_frequency, label_config,
data_version, source_revision, session_calendar` 参数。旧 `daily`、未声明模式的
`5m` 调用保留原日频因子路径；`1d` 是旧日频数据入口的别名，新接口要求显式模式。
`1min`、`5min` 等正整数分钟频率必须声明 `time_series` 或 `cross_sectional`，
显式声明模式的 `5m` 接入分钟路径并读取 `5min`。模式绝不根据标的数量推断。
`session_view` 默认为 `ETH`；分钟输出频率须匹配输入，或以 `daily`/`1d`
要求 Session 日频截面输出；不隐式重采样成另一分钟频率。

分钟子类实现 `calculate_factor(data)`，使用 `_fetch_market_bars()` 提供的完整应有
网格；此路径自动读取 `market_bars`，不调用旧 `_get_raw_data_for_factor`。
网格包含 UTC `timestamp_utc`、`instrument_id`、交易日字符串 `session_id`、OHLCV
和 `observed`。`start_date/end_date` 指含端点的 **Session 日期**，不是 UTC 自然日。
日历可显式绑定单个 `SessionCalendar` 或标的字典，否则按元数据构造。
`data_version` 非空时校验读取的每条 Bar 版本，混用版本直接报错。
分钟接入保留源价格，不套用旧日频 PIT 调整；股票应由冻结来源明确复权语义，
期货仍仅支持 `price_adjustment='none'` 的单一到期合约。

```python
class MinuteMean(FactorBase):
    def calculate_factor(self, data):
        values = self.session_rolling(data, 'close', 5)
        return data[['timestamp_utc', 'instrument_id']].assign(factor_value=values)

    def _get_cache_inputs(self):
        return {'window': 5}

factor = MinuteMean(connection, 'minute_mean', '2026-09-01', '2026-09-02',
                    stock_pool=['600000.XSHG'], data_frequency='1min',
                    analysis_mode='time_series', session_view='ETH',
                    label_config={'horizons': [1, 5, 15, 60]}, data_version='v1')
long = factor.get_bar_factor_frame()
result = factor.evaluate_bar_factor(n_quantiles=5)
```

`session_rolling(data, column, window, min_periods=None, within_session=True,
warmup_sessions=0, warmup_bars=0, aggregation='mean')` 在桥接模块中也可直接导入。
`min_periods=None` 等于完整窗口。必须传完整网格，不传压缩后的观测 Bar。
网格来自 `expected_bar_opens`，缺 Bar、整段缺失会话、缺失收盘槽位都保留 NaN；
不 ffill、不插值。日内滚动按 `(instrument_id, session_id)` 隔离，跨自然日夜盘
仍在同一组；午休/维护休市没有应有 Bar，因此不占缺失槽位。
`within_session=False` 允许跨 Session 滚动，但仍按应有槽位计算，完整窗口有缺口
即输出 NaN。预热按每个标的从所请求范围首个 Session 开始计数：前
`warmup_sessions` 个交易时段、前 `warmup_bars` 个应有槽位均置 NaN，二者取并集。
需要评估起点前预热时，应把请求范围前移足够交易时段，再显式裁剪正式评估区间；
框架不自动猜测历史长度，也不按自然日或实际观测行数计数。

| 模式 | 接口与输出 | 评估接入 |
|---|---|---|
| A 分钟时序 | `get_bar_factor_frame()` → `(timestamp_utc, instrument_id, factor_value)` | `BarReturnLabeler` + `TimeSeriesEvaluator` |
| B 分钟截面 | 同一长表，显式 `cross_sectional` | `BarReturnLabeler` + `BarCrossSectionalEvaluator` |
| C Session 日频截面 | `output_frequency='1d'` 后 `get_factor_values()` → `DatetimeIndex(name='date') × stock_code` | `AlphalensRunner.run(daily_factor, factor.get_session_daily_pricing())` |

分钟 `get_factor_values()` 返回 UTC 宽表（索引名 `date` 保留旧缓存约定），
`bar_factor_frame(wide)` 无损转成长表，NaN 不丢失。长表输出通过
`validate_factor_frame`，重复键、非 UTC、无穷值报错。缺省输出覆盖整个网格，
缺行情槽位因子固定为 NaN。
`get_bar_factor_frame()` 也将当前源 Bar 不完整的派生值置为 NaN，供 Python 和
三列 CSV 回放共用。标签候选仍仅为观测 Bar。
`evaluate_bar_factor` 从同一次读取快照生成因子、标签和评估；`label_config`
直接传给标签器（`horizons, entry_offset_bars, boundary_policy`），视图来自
`session_view`。若走 Harness CLI，写出 `market_bars` 和长表 CSV，使用既有
`time-series-analysis --config ...`；产物目录仍由 Harness 绑定。

日频聚合默认取每个 Session 的最后应有槽位，不用 `groupby.last` 跳过 NaN。
`session_daily_frame(..., aggregation=...)` 可显式选择日内均值等聚合。
`get_session_daily_pricing()` 从同一网格取 Session 收盘，返回
`MultiIndex(asset, date)` 的 Series 并绑定共享 `TradingCalendar`；不同日历的
资产拒绝混成一个日频横截面，应分别评估。日频标签/IC/分层规则全部沿用现有
`AlphalensRunner` 和 `ForwardReturnsCalculator`。

缓存继续使用显式 `_get_cache_inputs()`，未声明完整依赖则禁用缓存。身份绑定
模式、原始输入/输出频率字符串、视图、全部资产和品种元数据摘要、数据版本、
源 revision、逐标的 calendar hash / RTH 参考日历、标签参数、因子自身参数与额外
输入、原始网格内容及实现源码。旧因子缺少新增属性时按 None 处理；不能跨视图、
频率、模式、版本或标签定义复用。同样的底层数据仅切换 ETH/RTH 也必须重算。

## 后续挖掘、合成与训练扩展边界

以下为后续实现必须遵守的接口规范，本阶段不启动挖掘、训练或 OOS 研究。

1. **因子挖掘（Symbolic / DSO Mining）**：消费唯一键为
   `(timestamp_utc, instrument_id)` 的完整网格、`session_id`、逐标的日历和
   `observed` 掩码；算子输出标准因子长表。`shift`、rolling、滞后算子必须明确
   是否允许跨 Session，并按应有网格操作，禁止在观测行上盲目 shift。
   适应度明确指定模式：时序逐标的报告 full / non-overlapping IC，优先使用冻结
   非重叠口径；截面按每个 timestamp 计算跨标的 IC、ICIR 与分层覆盖。
   训练内选择方向、算子参数和适应度阈值，保留全部试验与缺失原因计数。
2. **因子合成（Factor Synthesis）**：分钟表采用
   `(timestamp_utc, instrument_id, factor_name)`，日频采用
   `(date, instrument_id, factor_name)`；先校验模式、Session、视图、频率、版本和
   标签身份一致，再按键对齐。缺失保留并报告，不用 inner join 隐藏缺口。
   截面去极值/标准化仅在同一 timestamp（或 Session 日期）跨标的进行；
   时序仅按单标的已知历史窗口 trailing 估计，不得使用全样本均值/方差/分位点。
   权重和变换参数只能在训练集拟合，验证与测试复用冻结参数。
3. **模型训练（Model Training）**：输入特征键、可知时间与 `LabelRecord` 的
   `signal_timestamp, signal_available_at, entry_timestamp, exit_timestamp,
   horizon_bars, status` 同时保留。excluded 记录不进入监督目标。特征必须在
   `signal_available_at` 已知；绝不能只按 `signal_timestamp` 切训练/验证/测试。

**Purging（重叠清除）**：把样本信息区间视为
`[signal_available_at, exit_timestamp)`。验证/测试保护区间从首个应有信号
的可用时刻开始，结束于该折最大实际退出时间。清除信息区间与保护区间相交
的训练标签。对于验证段之前的训练样本，逐持有期保留
`train.exit_timestamp <= val_first_signal_available_at` 的有效标签。
若交叉验证允许验证段之后的训练样本，也须检查它们与上述保护区间的相交。
使用 UTC 实际退出时间，跨午休或维护时段时按交易日历取退出 Bar 的结束时刻。

**Embargo（隔离带）**：验证/测试最后信号之后，先等待该折最大实际
`exit_timestamp`，再施加不少于配置 `embargo_bars` 的应有 Bar 隔离，或明确配置
的 `embargo_timedelta`。之后才允许样本重新进入后续训练；边界检查使用
`signal_available_at`。Bar 隔离按各资产所选视图的 `SessionCalendar` 推进，不计
午休/维护/周末为应有 Bar；多资产共享模型采取覆盖所有资产退出和隔离截止的
保守共同截止。先 Purge 再 Embargo，两者不互相替代。每折保存边界、标签配置、
日历/数据身份、剔除样本键与原因，审查测试集在调参期间保持隔离。

分钟 DuckDB 接口和本地评估不打包到聚宽云端单文件策略；构建器沿用本地读取隔离规则，
将这些方法替换为明确报错，既有日频与旧 `5m` 聚宽适配路径保持原行为。
