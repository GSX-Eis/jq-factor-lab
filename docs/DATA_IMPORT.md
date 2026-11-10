# 本地数据导入

本节的导入器只读取已经保存到本地的文件，写入调用方指定的 DuckDB。导入器没有聚宽下载器、登录逻辑、SDK 依赖或在线复权接口；不需要连接聚宽账号。保留聚宽常用字段和证券代码格式，方便导入已有导出文件。

## 入口

从项目根目录执行，路径由使用者提供。以下 data/import 目录需要先放入自己的文件；仓库不附带行情或研究数据。

```bash
# 日线、5 分钟行情：输入根目录下分别使用 daily/ 和 5m/
python -m data_pipeline.loaders.price_data_to_duckdb --data-dir ./data/import/price_data --db-path ./data/market.duckdb

# 已有因子 CSV：每个证券一个文件
python -m data_pipeline.loaders.factor_data_to_duckdb --csv-dir ./data/import/factors --db-path ./data/market.duckdb

# 辅助数据：示例为证券基本信息；--help 列出其他数据集
python -m data_pipeline.loaders.portfolio_data_to_duckdb --dataset security_master --source ./data/import/security_master --db ./data/market.duckdb

# 指数成分
python -m data_pipeline.loaders.import_index_constituents_duckdb --csv ./data/import/index_constituents.csv --db ./data/market.duckdb

# 财务报表：调用方提供字段契约
python -m data_pipeline.loaders.import_raw_financial_statements_duckdb --statement income --source ./data/import/financials --schema-contract ./data/import/financial_schema.json --db ./data/market.duckdb
```

## 输入约定

| 导入器 | 输入 | 目标表 |
| --- | --- | --- |
| 行情 | daily/、5m/ 下 `<stock_code>.csv`；字段 date、open、close、high、low、volume、money，复权字段 factor；日线另含 paused、high_limit、low_limit、pre_close | price_data_daily、price_data_5m |
| 因子 | `<stock_code>.csv`，date 加各因子列；证券代码取自文件名 | factor_data |
| 辅助数据 | 按 `DATASETS` 定义的文件名和字段；包括基本信息、估值、风格、行业、指数行情及事件数据 | 对应数据集的表 |
| 指数成分 | date、stock_code、index_code | index_constituents |
| 财务报表 | income/、balance/、cashflow/ 下 raw_<statement>_statement_YYYY.csv；字段由 JSON 契约声明 | raw_<statement>_statement_events |

财务字段契约示例：`{"income": {"columns": ["id", "code", "pub_date", "end_date", "revenue"]}}`。其中 id 是主键，pub_date 是公告日，code 是证券代码，end_date 是报告期末日。导入器按字段类别转换类型，未保留私人实验的日期截断。

行情与因子 CSV 可使用 `000001.XSHE.csv` 等原有命名。公开版不提供实际证券数据；测试使用虚构证券和合成数值。

## 重复导入与错误处理

行情与因子目录没有可处理的 CSV 时命令非零退出；行情的已完成文件全部被断点续传跳过仍是有效操作。每个待导入文件必须独立包含必填表头，不能由同批其他文件补齐。指数成分的 date 必须可解析，stock_code 和 index_code 去除首尾空白后不能为空；表头正确但没有任何记录的文件也不能用于清空旧数据。

行情导入沿用按“频率/文件名”记录的恢复机制：已记录的文件会跳过，不会因为文件内容更新自动重导。要重建修订后的整份行情，使用新数据库重新导入；不要把这种恢复机制当成自动增量更新服务。因子和指数成分按主键更新；辅助数据按各数据集的主键或去重规则合并。

行情按频率目录、因子按一次目录操作、辅助数据按数据集、指数成分按一次文件操作使用事务；出现向调用方报告的导入异常时回滚该操作。整个多数据集命令不是一个总事务，前面已经成功完成的操作会保留。`--replace` 仅在支持它的导入器中显式清空目标表后重导，其失败不会留下已清空的表。

财报每张报表的建表与导入在同一事务内；写入前校验已有表的列名和类型，列顺序可不同，按列名写入。失败不会残留本次新建的表。若数据库已留有旧错误结构，重试会报告 schema mismatch 并保留原数据，应先备份并核对字段后迁移，或使用新数据库重新导入。

CSV 解析错误不再设置为静默忽略；各数据集仍保留原字段类型转换和缺失主键/公告日期过滤规则。导入成功不代表已经完成所有数据质量审计，应核对导入统计与源文件。

## 使用边界

这次加入的是 DuckDB 本地导入器，保留既有表结构供分析与合成库读取。单因子 CSV 演示和合成算法演示仍是独立入口，不会自动转用刚导入的数据库；实际分析调用方需要显式指定数据库路径或数据绑定。[完整本地研究示例](LOCAL_WORKFLOW.md) 已演示导入后的 DuckDB 如何接入因子、合成和打包。

下载器、私人路径、账号凭证、实验配置、实际数据和专项冻结回填流程均未导出。运行日志、数据库与 Parquet 输出应放在忽略的 data/ 或 outputs/ 下。

## 数据库格式

本地运行仅支持 DuckDB；Parquet 由 DuckDB 读取。不会自动读取、转换或回退到 SQLite。已有 SQLite 文件需先用外部工具导出 CSV/Parquet，再导入新的 DuckDB 文件；改文件后缀不能转换格式。数据绑定模式不会在空查询后切换到其他数据源。

## 适配器的数据源选择

`get_data_adapter()` 和直接构造 `DuckDBAdapter()` 使用同一套规则：

| 参数 | 行为 |
| --- | --- |
| `data_bindings=None`（默认） | 从当前进程的 `QUANT_DATA_BINDINGS` 读取绑定；未配置环境绑定时使用普通路径。 |
| `data_bindings={}` | 明确关闭环境绑定，使用传入的普通路径。 |
| 非空 `data_bindings` | 使用指定的 `input_db` 及数据版本；查询时需要有效的 `inputs` 表版本。 |
| 绑定 + `duckdb_path` | 路径必须对应绑定声明的 `input_db`；未知路径报错，不会被静默覆盖。非空 `sources` 旧别名契约已停用。 |
| 绑定 + `parquet_dir` | 报 `ValueError`。删除 `parquet_dir` 来使用绑定，或明确传 `{}` 来使用普通路径。 |

例如，明确读取自己的数据库或 Parquet，不继承运行环境中的绑定：

```python
from utils.duckdb_adapter import get_data_adapter

adapter = get_data_adapter(duckdb_path="data/prices.duckdb", data_bindings={})
# Parquet 模式：get_data_adapter(parquet_dir="data/parquet", data_bindings={})
try:
    prices = adapter.fetch_price_data(["000001.XSHE"], "2026-01-05", "2026-01-06", ["close"])
finally:
    adapter.close()
```

Parquet 目录沿用现有分区布局：`<parquet_dir>/<表名>/year=YYYY/month=M/*.parquet`。无绑定且同时配置 DuckDB 和 Parquet 时，必须显式指定 `source_policy="duckdb_then_parquet_on_empty"`，否则构造时报错。指定该策略后，行情、因子和通用表查询先查 DuckDB，只有查询成功但结果为空时才尝试 Parquet；需要唯一来源时只传一个路径。DuckDB 缺表、缺字段等执行失败会直接抛出异常，不会触发 Parquet 回退；Parquet 文件损坏或查询失败同样会抛出异常。原先依赖 DuckDB 缺表后回退的调用，应明确选择 Parquet 来源。

工厂函数在无绑定、无路径时返回 `None`；直接构造适配器则要求至少一个来源。两种本地读取模式都依赖 DuckDB，工厂即使无来源也会先检查该依赖。

已有版本化存储时，可按 [显式版本读取的最小验证](CANONICAL_DATA_READS.md) 检查当前行情、冻结行情与冻结因子三条链路；无需旧路径别名或整库复制。

绑定模式只接受契约声明的物理库路径和明确的表版本。`FactorReader` 的 `database_path` 与 `results_database_path` 都必须对应 `output_db`。复现历史结果时保留原 `table/version`，把读取路径改为契约声明的物理库，并移除 `sources`；不要直接改成最新版本。一个表的不同版本使用独立绑定或明确的表别名。普通无绑定 DuckDB 文件读取仍受支持，文件名不要求叫 `input.duckdb` 或 `output.duckdb`。

## DuckDB 连接资源

普通文件、版本绑定和 Parquet 模式共用一套连接预算，默认内存限制为 `3GB`、线程数为 `2`。在创建连接前，可用以下环境变量覆盖：

```bash
export QUANT_DATA_STORE_MEMORY_LIMIT=4GB
export QUANT_DATA_STORE_THREADS=2
```

旧的 `QUANTATION_DUCKDB_MEMORY_LIMIT` 不再生效，请改用 `QUANT_DATA_STORE_MEMORY_LIMIT`。内存限制是 DuckDB 查询预算，不会预分配对应内存，也不是整个 Python 进程的内存上限。

适配器首次查询才建立连接，后续复用；修改环境变量不会重配已有连接。`close()` 同时关闭连接并清空查询缓存，可重复调用；下一次查询会重新连接并读取。此前关闭后保留缓存的行为已移除。

缓存键保留完整的股票和字段元组，并包含表名、日期范围及输入绑定摘要；股票顺序不影响命中，字段顺序影响命中。受支持的导入提交会更新数据库代号；读取器在命中前检查，变化时清空查询和股票池缓存。Parquet 内置导出有发布标记，未管理目录不复用跨调用缓存。跨进程写入前仍需释放只读连接。详见 [缓存失效与 harness 边界](CACHE_INVALIDATION.md)。需要每次执行查询时，构造适配器可设置 `use_cache=False`。

关闭重读会继续使用原来指定的输入版本，不会自动切换到最新版本；需要另一版本时应创建带有对应绑定的新适配器。

资源配置或版本初始化失败会抛出错误、关闭临时连接，且不会保存未初始化完成的连接；修正配置后可使用同一适配器重试。

## 通用 SQL 入口

`DuckDBAdapter.execute_sql(sql, params=None)` 在当前连接上执行底层 SQL，支持位置参数和命名参数：

```python
df = adapter.execute_sql(
    "SELECT stock_code, close FROM price_data_daily WHERE stock_code = ?",
    [stock_code],
)
df = adapter.execute_sql("SELECT $stock AS stock_code", {"stock": stock_code})
```

参数用于传递值，不能替代表名或列名。结果保留 SQL 的原始列名，合法零行查询仍保留列结构；连接、查询和参数错误直接抛出，不再转换为空表。

此入口不做面板配置映射、日期/股票键规范化、结果缓存或自动来源回退；Parquet 文件需在 SQL 中显式读取。它复用适配器连接，支持 `PRAGMA` 和 `SET`，会话设置会影响后续调用。输入数据库文件只读不代表连接会话不可修改。

## 按日查询范围与 SQL 参数

`fetch_price_data()`、`fetch_factor_data()`、`fetch_table_data()` 的日期参数使用 `YYYY-MM-DD`，范围包含起止日全天：开始日零点及之后、结束日次日零点之前。不再根据 `price_data_5m` 这个特定名称选择日期规则；DuckDB 中 `price_data_5m` 和 `main.price_data_5m` 的时间范围一致。

支持 DATE、TIMESTAMP，以及导入器使用的规范 ISO 日期字符串列；不会自动修复非规范文本日期。查询值（股票、日期和 Parquet 文件路径）使用参数绑定，DuckDB 表名与列名正确引用转义。股票值含单引号、字段名含双引号时也按字面内容查询。

单独使用 DuckDB 时，成功查询无匹配行仍返回空表。读取 Parquet 或显式回退到 Parquet 时，对应表目录不存在会报错；目录存在但文件读取/解析失败同样会报错。

## 行情读取的频率与空结果

`fetch_price_data()` 的 `frequency` 必须匹配读取配置中的键。默认配置提供精确的 `daily` 和 `5m`；默认情况下 `1m`、`5M`、空字符串、`None` 会明确报错，即使股票列表为空也不会按日线处理。FactorBase 保留传入的原始频率，本地行情读取同样校验，不再提前把未知频率改为日线；聚宽适配器原有的额外分钟频率范围不在此次收窄范围内。

`fields=None` 默认读取 `open, close, high, low, volume, money` 六列，不会自动加入 `factor` 等字段。`fields=[]` 只读取日期和股票键。行情入口统一返回 `date, code` 及请求的业务列；空股票和无匹配记录的空结果也保留这个列结构；缺失 Parquet 表目录会报错，其中 `date` 为日期时间类型。空业务列的 dtype 不保证与有数据时一致。

有数据时保留按 code/date 排序与缓存副本语义；只计算一次缓存键，空结果仍不缓存。


## 本地面板读取配置

`DuckDBAdapter`、`get_data_adapter` 和 `FactorBase` 接受 `data_config_path`。
不指定时加载随源码和安装包提供的 `config/data_access.json`，不依赖当前工作目录。
自定义文件必须提供完整配置；不会与默认文件隐式合并。

配置定义价格频率到表名的映射、价格别名、因子输入表、每张表的日期/股票键、Parquet 子目录和默认字段。
`price_aliases` 必须显式提供；旧自定义配置需要补上该字段，不需要别名时填写 `{}`。不会隐式继承默认别名。
数据库路径仍由 `duckdb_path` 提供；冻结数据版本仍由 `data_bindings` 提供。
配置不应含账号、实验参数、研究结果或私人特征清单。自定义配置由用户自行保存，不进入框架发布文件。

```python
from utils.duckdb_adapter import get_data_adapter

reader = get_data_adapter(
    duckdb_path="data/input.duckdb",
    data_config_path="config/data_access.json",
    data_bindings={},
)
try:
    frame = reader.fetch_factor_data(
        ["DEMO.A"], "2026-01-01", "2026-01-31", fields=["my_feature"]
    )
finally:
    reader.close()
```

字段规则：`fields=None` 使用该表配置的默认值；没有默认值则报错。
默认 `factor_data` 不预设财务字段，因为其列由导入文件决定。
`fields=[]` 只读取日期和股票键；显式字段优先于配置，缺字段报错。
三个面板入口（价格、因子、通用表）都返回 `date/code/请求字段`，按股票、日期排序；空结果也保持相同列结构。
自定义输入键会统一重命名为 `date/code`，其他业务列不能与这两个输出名冲突。

只有一个来源时直接读取该来源。若同时传入 DuckDB 和 Parquet，必须明确传入
`source_policy="duckdb_then_parquet_on_empty"`：仅 DuckDB 成功返回零行才回退，不拼接部分结果，查询错误直接传播。
非空绑定仍不能与 Parquet 同时使用。缺表、缺字段、缺目录、空数据目录、文件损坏和反向日期范围均属于错误。
合法查询没有匹配记录或月份分区时返回空表。Parquet 记录日期必须与 `year/month` 分区一致；查询本身不执行全数据集一致性扫描。
显式请求的 `year/month` 列会保留。

读取配置在适配器创建时固定；修改文件后需要新建适配器。可用 `reader.data_config.as_dict()` 保存生效配置，
复现时应保留这份内容而不只是配置路径。`FactorBase` 计算缓存包含生效配置。
会话内的结果缓存不会自动检测外部数据修改；更新数据后先 `close()` 再读，或关闭查询缓存。
启用查询缓存后，普通查询自动缓存容量范围内的非空结果；无需单独预加载。缓存按完整查询键匹配，不自动复用股票、日期或字段子集。

此配置适用于本地面板读取。股票池筛选的表配置仍由其 `fundamental_table/price_table` 参数控制；
聚宽云端通过 API 读取，不需要本地 JSON。已有因子显式声明的字段、窗口与 PIT 参数不会被默认读取字段覆盖。


## 可用股票发现

`reader.get_available_stocks(start_date, end_date, min_records=10)` 返回配置中 `daily` 行情数据集在日期范围内达到记录数量门槛的股票，按代码排序。表名、日期列、股票列、Parquet 子目录与价格面板使用同一份配置；支持 DuckDB、冻结绑定与 Parquet。结束日期包含全天，反向范围或非法日期报错。

`min_records` 必须为正整数，统计记录行数，不是不同交易日数量。查询成功但没有股票达到门槛时返回 `[]`；缺表、缺字段、文件错误等向上传播。双来源仅在 DuckDB 对该日期范围查询成功零行时回退：若主源已有记录、只是各股票未达到门槛，则返回空列表，不改读 Parquet，也不合并来源。此入口不缓存结果。


## 指数成分股读取

`reader.fetch_index_constituents(index_code, date)` 查询精确日期的快照，返回排序、去重后的股票代码列表。
不会自动使用此前最近日期的成分股。指数标识和日期使用 SQL 参数绑定；标识必须是非空字符串，日期必须是有效的 `YYYY-MM-DD`。
查询成功但没有匹配记录返回 `[]`；缺表、缺字段、缺目录、空目录、坏文件或绑定错误均抛出异常。

默认读取配置包含：

```json
"index_constituents": {
  "table": "index_constituents",
  "index_column": "index_code"
}
```

此处的 `table` 引用 `tables` 中的定义，日期列、股票列和 Parquet 子目录来自该表配置。
更换物理表或列名时修改这些配置，无需改函数。只使用行情/因子面板的自定义配置可以省略整个
`index_constituents` 段；省略后调用指数读取会明确要求补充配置，不会猜测表名。

读取支持 DuckDB、冻结版本绑定和 Parquet。Parquet 支持导入器导出的
`index_constituents/data.parquet`，也支持其下的分区文件；使用递归文件查询与日期条件，
不假设一定存在 year/month 分区。双来源沿用显式 `duckdb_then_parquet_on_empty` 策略，
只有 DuckDB 成功零行才回退，不在查询错误后切换来源，也不拼接部分结果。

动态股票池使用同一指数读取逻辑，可接收 `data_config_path`；旧的 `index_constituents_table`
显式参数仍可指定表名。月度、周度及指定日期入口都执行指数约束：合法空快照产生空股票池，
不会返回未过滤的股票列表；指数读取异常会传到调用方。指定日期筛选先应用指数约束，再限制股票数量。
目前一个筛选请求只支持一个 `index_code == "指数标识"` 条件；不支持的运算符、空值和重复指数条件会报错。
已有基本面取数日期/滞后策略保持原接口，指数查询匹配各入口实际传入的快照日期。

指数读取接口自身不缓存。股票池管理器只缓存成功且非空的成分股集合，返回副本；失败和空快照不会污染该缓存。
管理器的其他股票池缓存仍按原会话规则运行；更新输入数据后请新建管理器。

## 分析读取器的配置边界

`FactorLoader.load_factor_from_file()` 与 `load_all_factors()` 接受额外关键字参数，并原样传给因子构造函数，包括 `duckdb_path`、`parquet_dir`、`data_config_path` 和 `source_policy`。双来源仍需显式指定策略。

收益标签使用的 `PriceFetcher` 只读取 DuckDB，保留的 `parquet_dir` 参数不参与读取，也不提供 Parquet 回退。省略 `data_config_path` 时同样加载默认 `config/data_access.json`，显式传路径则使用该文件。价格表及日期、证券键由 `DataAccessConfig.resolve_price()` 按生效配置解析：先查 `price_aliases` 得到频率，再通过 `price_tables` 找实际表；非别名按字面表名选择 dataset，并从 `price_tables` 确定频率。默认别名如下（这是完整配置中的一个字段）：

```json
"price_aliases": {
  "price_data": "daily",
  "price_data_daily": "daily",
  "price_data_5m": "5m"
}
```

别名优先于同名旧表，因此改动 daily/5m 对应的实际表后仍会读取新表。别名不能指向未配置的频率；同一实际表不能登记多个频率；别名不能覆盖另一频率的实际表名。未登记频率的字面表保留按日取最后收盘价的原有行为。输出仍为 `(asset, date)` 的 close Series，日内数据仍按日取最后收盘价。必须在 DuckDB 中准备标签所需的行情。

动态股票池同时收到借用连接和显式数据源（路径、绑定或来源策略）时，以显式数据源创建读取器，完整执行来源校验与回退规则。仅传连接时仍使用该连接；管理器不关闭调用方的借用连接。

股票分块探针不通过时，基类恢复完整股票范围并执行完整时间窗口计算，不再自动切换时间分块。适配器的复权基准查询仍与行情采用相同的表、日期/证券键、冻结版本和来源策略，支持 Parquet；基准在 DuckDB 内聚合，空行情不提供基准，已观测行情的非法复权因子会抛错。

`PriceFetcher` 的数据库参数统一为必填 `duckdb_path`，接受普通文件路径字符串或 `Path`，不再接受 `db_conn_str`、`db_backend` 或 `duckdb:///...` 连接 URI。示例：

```python
reader = PriceFetcher(duckdb_path="data/input.duckdb", price_table="price_data_daily")
try:
    prices = reader.fetch(["000001.XSHE"], "2024-01-01", "2024-01-31")
finally:
    reader.close()
```

## Trading calendar

Import an independently verified calendar into the input DuckDB before using stock-pool schedules, forward returns or Qlib exports. Price observations do not define exchange sessions. See [Trading calendar](TRADING_CALENDAR.md) for the import receipt, coverage, version bindings and Parquet-only usage.

## 已退役的滚动复权中间列接口

2026-10-26 移除 `utils.pit_adjustment.add_rolling_pit_columns()` 及其导出，不保留兼容包装。
当前仓库业务入口未调用这个函数；它原先仅生成 `price × factor` 中间列，不负责滚动窗口、
计算截止日或逐窗口复权。旧调用会直接导入失败，需要显式修改。

常规因子继续使用 `FactorBase` 现有复权链路。独立代码若需要相同的窗口复权，可审阅
`utils.pit.apply_pit_adjustment()`：它会修改副本中的价格和成交量，默认以每只股票窗口内
最后一个有效复权因子作为基准，不是生成 `pit_*` 列的直接替代品。调用前须提供未复权行情、
合法复权因子，并将窗口截断到计算截止日；不得对框架已经复权的价格再次执行。

同日进一步移除整个旧 `utils.pit_adjustment` 模块，包括策略基类、DuckDB/JQ
旧调整器、异常类、策略常量及 `create_pit_adjuster()` 工厂；不保留兼容导入。

合成读取器的 `get_pricing_data()`、`get_open_price_data()` 和
`get_complete_price_data()` 同时移除 `pit_adjuster`、`pit_base_date`、
`pit_base_date_mode` 参数和旧辅助调用。复权开关统一使用 `pit_enabled`，
仍按现有 `factor_end` 方式计算。`get_pricing_data()` 的 `future_periods`
必须以关键字传入，避免旧位置参数被误解释为前瞻交易日数量。

## 合成价格的严格 PIT 契约

`factor_synthesis.core.factor_data_loader` 中的两个 `apply_pit_adjustment_factor_end*`
接口仍使用 `price × factor / factor_end`。每只股票的基准取窗口内日期最晚的
有价格记录；输入顺序不影响基准，输出保留价格的索引顺序。调用方应提供
未复权价格和对应因子，不得重复复权。

- 有价格的位置必须有数值型、有限且大于零的 factor；不再补 1、不接受数字字符串或布尔值。
- 无价格的位置保持缺失，不要求 factor，也不用于选基准。合法 factor 本身可以为 1。
- Series 使用具名 `(asset, date)` 或 `(stock_code, date)` 两层索引；层顺序可调整。
  宽表使用日期索引和股票列。日期必须是无时区、非缺失的 DatetimeIndex，股票/日期键不得重复。
- 非法输入或数值溢出抛出 `PITValidationError`，错误包含坏值的股票、日期样例；不会返回原价格。

读取器在日内聚合之前校验有价格记录，防止聚合跳过坏因子；缺失价格记录的因子
不参与每日最后值选择。`get_pricing_data()`、`get_open_price_data()`、
`get_complete_price_data()` 和训练收益计算启用 PIT 后，缺少 factor 列、数据库
查询失败或任一批次失败均向上传递错误；不再自动关闭 PIT 或返回部分批次。
查询成功但没有价格观测时仍返回原接口约定的空结果。

显式设置 `pit_enabled=False` 仍读取原始价格，并保留原有非 PIT 回退行为。
遇到数据错误时应修复输入数据，而不是靠关闭 PIT 让任务继续。

## 股票池字段查询与缓存

`DynamicStockPoolManager` 仅缓存成功读取的表字段。缺表、失效连接或其他结构查询
错误会保留表名和原始异常，不再变成已缓存的空字段列表；修复数据源后，同一实例
可以重试。`validate_filters()` 保留返回布尔值的接口，结构查询失败时返回 `False`
并记录错误；选股入口读取必要表结构失败时抛错，不缓存此次股票池结果。

基本面表提供候选股票；只有筛选字段或排序字段不在基本面表中时，才读取行情表结构。
仅基本面/指数条件的选股不要求额外建立无用的行情表。指数成分仍由独立入口验证。
成功字段缓存随导入代号失效；字段、成分和最终股票池一起刷新。直接手工 SQL 修改后须显式失效或创建新管理器。切换来源或绑定仍应创建新实例。

## 导入事务与目标列约定

行情、指数成分及 Portfolio 的 INSERT 使用显式目标列，表的物理列序不再决定字段落点；
Portfolio 的整行 EXCEPT 去重按同一列集、同一顺序比较。现有表的主键和去重口径不变，
不会迁移或改写已落盘数据。使用旧数据库前，应核对表结构与数据集合同。

行情导入不再吞掉 factor 列迁移、读取或写入 import_progress 的错误。已有大小写变体
`Factor` 不会重复建列；CSV 的大小写扩展名均可识别，重复非空表头明确拒绝。
`import_directory` 按一次目录调用提交数据、进度与修订号，失败会回滚；直接调用
`import_csv_file` 的调用方仍必须显式开启并提交或回滚事务。
首次使用现有行情库时，检查进度表含 file_name、imported、record_count、import_date 四列。

日因子更新区分“源文件没有该列”和“源文件明确写 NULL”：缺列保留旧值，显式 NULL
仍清空对应值。每个文件按自己的列集更新；不借助批次 union 后的 NULL 猜测缺列。
因子 Parquet 导出支持含单引号的路径。

Portfolio 每个数据集独立提交，`--dataset all` 不是整体事务。法定现金请求数据集
`statutory_cash_claim_events` 的默认源为数据根目录下的
`dividend_lowvol_wide/statutory_cash_claims`，执行 all 时不再因缺默认路径映射而中断。
资源配置出错也会关闭本次创建的连接。

原始财报每张报表单独提交：数据、修订号、两条索引及摘要查询现在都在同一个事务里。
索引或摘要失败，以及中断异常，都会回滚这一张报表；已成功提交的其他报表不回滚。
上述修复不自动修复历史数据，也不改变既有日期窗口、类别枚举或 CSV 额外字段的合同。
