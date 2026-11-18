# 本地交易日历

## 阶段 6.4a：Session 边界换月

`build-roll-schedule` 以 Session 标签而非自然日做决定和生效：t-1 收盘决定，
t 的第一交易段开盘生效。周五 21:00 的夜盘归属下一交易 Session（通常周一），
因此周五日盘收盘的 OI 决定可在周五夜盘执行；夜盘跨午夜仍使用同一合约。
长假前不补夜盘，假后首个 Session 的决定日为日历中上一交易 Session，确认日数不计
周末或休市日。假后第一交易段实际为日盘时，换月也在该日盘开盘生效。
连续读取的日期参数按 Session 选段，UTC 参数按真实 Bar 开始时点截取；两者均不推断
自然日归属，不改变已有股票/日频 calendar_hash。分钟分桶默认策略由 6.4b 单独核对。

日频与分钟级共用 `utils/session_calendar.py` 的 PMC 日历核心，固定依赖 `pandas-market-calendars==5.6.0`，叠加版本化 `config/calendar_overrides.json`。不从行情记录、价格表日期或周一至周五推断交易时段。日历查询不调用聚宽，也不联网。

## 统一日历与交易时段

阶段 6.3 中，股票旧配置窗口由 `utils.time_axis.stock_requirements` 通过
XSHG 2026-01-05 的完整 Session 网格推导，1m/5m/15m 仍为 240/48/16。
此参考日只定义旧 `min_bars/min_days` 的单位；国内期货继续要求显式分桶策略。

```python
import pandas as pd
from utils.session_calendar import SessionCalendar
from utils.trading_calendar import TradingCalendar

calendar = SessionCalendar('NQ', '2026-01-01', '2026-12-31')
session = calendar.session_at(pd.Timestamp('2026-12-01T23:00Z'))
assert session.session_date == '2026-12-02'
grid = calendar.expected_bar_opens(session, '5min', view='RTH')
daily = TradingCalendar.from_pmc()  # XSHG，2016-01-04..2026-09-30
```

NQ/ES 默认使用 PMC 的 `CME Globex Equity`（接口也接受 `CMEGlobex_Equity`），基准现货日历为 `NYSE`；也可以显式指定 `CME_Equity` 或覆盖 `rth_reference_calendar`。两个 CME 日历的历史中断和提前收市定义可能不同，交集始终遵守所选底层日历。

ETH 采用品种完整时段，去除 PMC 声明的午休和交易中断。RTH 严格定义为 **ETH ∩ rth_reference_calendar 常规时段**，不写死开收盘时间。NYSE 夏令时/冬令时、节假日、13:00 ET 半日市都会自动作用于交集。CME 每日维护休市属于两个交易时段之间的空档；前一自然日晚间开盘按 PMC session 标签归属下一交易日。XSHG 的上午 09:30–11:30 和下午 13:00–15:00（上海时间）是两个分段，ETH 与 RTH 相同。

`session_at` 只接受 UTC / Etc/UTC tz-aware datetime 或 Timestamp。按 Bar 开始时刻和左闭右开区间 `[open, close)` 判断，休市返回 None。`expected_bar_opens(session_date 或 Session, frequency, view)` 返回严格递增的 UTC DatetimeIndex；频率由 `Frequency` 解析，分桶及段末不足整桶的规则见下方 6.1，不要求段长整除频率。区间接口 `expected_bar_opens_in_range(start_utc, end_utc, frequency, view)` 筛选 `[start, end)` 内的开始时刻，网格由同一分桶策略决定，不随查询起点漂移。

覆盖区间由交易日标签声明；越界抛出与日频同源的 `CalendarCoverageError`。覆盖首日之前的夜盘，只要属于已覆盖交易日，仍可查询；覆盖末日晚间若已属于下一未覆盖交易日则报错。其他休市时刻按交易所本地日期检查覆盖。非 XSHG 日历必须显式传入覆盖区间，调用方负责确认该区间适用；PMC 规则快照并非独立交易所公告审计。

修正文件包含 schema_version、按 PMC 日历名组织的 add_sessions、remove_sessions 和 special_sessions。每项使用 session_date 与非空 source 核验说明；新增及特殊时段还需提供显式 UTC 的 segments。特殊时段替换当日全部分段，休市修正删除整日。修正同步作用于品种与基准日历。当前 XSHG 与已核验 CSV 的 2611 日逐日一致，三个修正数组为空，核验来源保留在文件中。未来发现差异时逐项登记，禁止无来源补日。

`calendar_hash` 为 SHA256，包含配置、声明覆盖区间、PMC/exchange-calendars/tzdata 版本、实际 UTC 分段和修正文件原始字节哈希，可传给 ResultIdentity。任何修正文件内容变化（包括来源说明）均改变该哈希；已构造对象保留其快照，文件更新后需重新构造。

日频 `TradingCalendar.from_pmc()` 使用同一核心提供交易日；显式 sessions 构造、所有公开日频方法和旧 DuckDB 加载入口继续兼容。`fingerprint` 原公式保持不变，仅使用 calendar_id、覆盖边界与交易日列表，不能替换为分钟级 calendar_hash。默认 `LOCAL_A_SHARE_XSHG_CROSSCHECKED` 的 2611 日快照因此与历史指纹完全一致。`config/trading_calendar.json` 的 table/calendar_id 结构不变。

## 存储与导入

`config/trading_calendar.json` 声明表名和 calendar_id。表结构为 `calendar_id VARCHAR, date DATE, is_open BOOLEAN`，主键为 `(calendar_id, date)`。导入时，核验收据声明的覆盖范围内每个自然日都写入一条记录，休市日也保留。

读取时，覆盖范围取所选日历现存记录的最早日期至最晚日期，范围内部缺行会报错。读取器不对照导入收据的原始边界；若开头或结尾的日期记录被删除，加载可能成功，但覆盖范围会缩小。查询超出缩小后的范围仍会报错，不会把未知日期当作休市。该结构检查不能证明原始边界记录完整。

先将本地重建结果与独立外部日历核对，保存单列 `date` CSV 和核验 JSON，然后运行：

```bash
python -m scripts.import_trading_calendar \
  --database /data/input.duckdb \
  --csv /data/calendar/trading_days.csv \
  --verification /data/calendar/verification.json \
  --receipt /data/calendar/import_receipt.json
```

核验 JSON 至少包括 `status: "verified"`、`calendar_id`、`start_date`、`end_date`、`trading_days`、`differences: {"local_only": [], "online_only": []}`、`files: {"trading_days.csv": "<SHA256>"}`。真实导入应保存重建数据版本、外部来源 URL/提交/哈希和核验方法。工具核验文件完整性与契约；它不会替使用者执行市场日历交叉验证。

导入使用现有 input-store 的 staging、事务、输入版本和 head 检查。重复同一批次可重复执行；旧版本保留。不能把不相邻的覆盖区间接成一个“中间日期全休市”的区间。扩展时应提供连续且核验过的日期范围。

新 Harness 执行从 catalog 冻结日历版本。旧冻结配置缺少日历绑定时会失败，需要为新执行重新生成输入绑定；不要直接修改旧证据及其哈希。

## 读取契约

```python
from utils.trading_calendar import load_trading_calendar

calendar = load_trading_calendar(duckdb_path="/data/input.duckdb")
calendar.sessions("2025-01-01", "2025-01-31")  # 含起止日期
calendar.previous_sessions("2025-02-05", 5)   # 默认不含 anchor
calendar.next_sessions("2025-01-27", 5)
```

未传已有连接时，函数解析显式 `bindings`；未指定时读取环境绑定，并通过 `connect_input()` 创建连接。存在有效版本绑定时严格读取指定版本，不回退到实时 head；查询结束或失败后关闭自己创建的连接。

传入已有 DuckDB 连接时，函数直接借用该连接，不重新解析环境绑定，也不使用额外传入的 `duckdb_path` 或 `bindings`。连接创建方负责事先绑定正确版本，调用方负责关闭连接；给未绑定的已有连接额外传 `bindings` 不会使它自动绑定版本。

查询越界、所需前后交易日数量不足、日历为空，以及现存覆盖区间内部缺少自然日记录均报错。边缘记录缺失的检测边界见上文。

Parquet-only 的价格读取可以通过 `DuckDBAdapter` / `FactorBase` 的 `calendar_duckdb_path` 显式指定独立日历库。这个参数不提供额外价格来源。普通 DuckDB 模式默认从同一连接读取日历。

纯内存的仿真或单元测试可显式构造 `TradingCalendar(sessions, coverage_start, coverage_end)`。这表示调用方声明了日历，不能把真实价格观测日期直接当作已核验交易日。收益计算器接受 `trading_calendar=`，PriceFetcher 返回的 Series 在 attrs 中携带该快照；手动重建 Series 时须保留它或显式传入。

## 已统一的公共执行路径

- 动态股票池：月末和周内调仓日期来自独立日历。基本面“最近可用数据日”仍是数据可用性查询，不等同于市场上一交易日。
- 合成：先按市场交易日补齐，再执行 T−1 移位；日历失败不会退回逐股票简单移位。处理后缓存包含日历指纹。
- 价格与收益：扩展准确的未来交易日；训练/评估均按市场会话行移位。市场整日缺价格保留 NaN，个股上市前/停牌等缺失也不填造价格。收益缓存包含日历及价格内容指纹。
- 本地混频预热、时间分块及分析入口：使用真实前 N 个交易日。分钟 bar 数换算日数向上取整；市场日历本身不保证个股有足够有效 bars。
- Qlib：calendar 文件来自独立日历；二进制列按日历重索引，写入 float32 起始位置和缺失值。格式依据 [Qlib 官方导出实现](https://github.com/microsoft/qlib/blob/main/scripts/dump_bin.py)。旧二进制目录需重新生成。
- 私有 VectorBT 入口和本地对齐：调仓/T−1 使用日历；回测价格整体缺失市场日会报错，避免悄悄缩短持有期。

## 边界

日历定义市场是否开市，不证明每只股票有完整行情、是否可买卖、或分钟级交易时段完整。PIT 复权公式未改变，日历也不负责价格填充。

聚宽云端运行仍遵守平台数据接口，不把本地 DuckDB 带进云端。自然日窗口保留自然日含义。历史私有因子内的 `trading_days_to_calendar_days` 估算辅助函数仍保留用于旧公式；公共混频取数已改用 `history_start`。独立历史研究脚本中的专用时间轴未批量改写，本次不是旧实验重跑或所有研究脚本迁移。

`previous_sessions` / `next_sessions` 的 `include_anchor` 只接受布尔值 `True` 或 `False`；字符串、数字、None等会报错，即使请求数量为0也不例外。默认不包含锚点的行为及合法调用的日期结果不变。

`TradingCalendar.align()` 会将通过校验的日期索引用于实际重索引，支持可解析的字符串日期、Python日期和DatetimeIndex。已有数值保留，缺失市场交易日补NaN；不修改调用者的原对象。重复日期、非交易日、NaT、带时区或非午夜日期仍报错。空输入保留原有直接返回副本的行为。


## 阶段 6.1：时间轴和国内期货

唯一频率解析入口为 `utils.time_axis.Frequency.parse(value)`。`1m/1min`、
`5m/5min`、`15m/15min`、`30m/30min`、`60m/60min/1h` 映射为
`kind='intraday'` 和分钟数；`daily/1d` 映射为 `kind='session'`。
`canonical` 返回 `Nmin/1d`，`joinquant` 返回 `Nm/1d`。
`spelling` 将 `Nm`、`daily`、**`1h`** 归为 legacy，`Nmin/1d` 归为 canonical。
非法输入保留在 `raw`，`is_valid=False`，解析本身不抛异常；消费者决定何时拒绝。

PMC 的 `open_close_map` 决定全部内部收盘/重开成对边界，另含 interruption 列。
`night_close→day_open`、`pause_start→pause_end`、午休都从 Session 中扣除。
XSHG、CME、NYSE 的时段和 calendar_hash 保持阶段 2 身份。
国内期货 ETH 包含夜盘和日盘，RTH 与对应 `*_Day` 参考日历相交；
CFFEX 股指和仅有日盘的 GFEX 均为 ETH=RTH；GFEX 使用自身日历，不存在 `GFEX_Day`。
GFEX 的 LC/PS/PT/SI 品种根绑定对应 `GFEX_<ROOT>`，RTH 参考为 `GFEX`。
周五夜盘属于周一，节前晚间无夜盘，AU/SC 夜盘跨午夜到 02:30；
Session 日期来自 PMC，不能用 Bar 的自然日期替代。
品种根绑定见 `config/futures_calendar_map.json`；缺品种或日历必须报错。

`BarSchedule(calendar, frequency, view='ETH', bucket_strategy=...)`
的 `for_session(date)` 返回有序 DataFrame：`session_id, slot`（从零开始）、
`start_utc, end_utc, available_at_utc, is_partial`。三个时间列均为 UTC，
可知时间等于实际桶结束，半开区间表示桶的墙上时间包络；其中的休息不算交易时间。

- `segment_anchored`：每段从开盘各自切桶，末桶可不足宽度并标 partial；股票默认。
- `session_clock`：累计交易时间切桶，可跨小节休息、午休甚至同一 Session 的周末间隔。
  恰好在休息开始凑满的桶在那里结束，下一桶从重开起算。
- `session`：整个 Session 一个桶，频率必须是 `daily/1d`；日频在前两种策略下也统一为该桶。

**国内期货所有频率都必须显式指定策略**；6.4b 已确认按项目自身约定管理（方案 A），
保留未显式指定时报错，并在 `utils.time_axis` 写入项目级推荐值
`RECOMMENDED_DOMESTIC_FUTURES_BUCKET_STRATEGY = 'segment_anchored'`
（`recommended_bucket_strategy()`），供调用方显式传入。
`expected_bar_opens(..., bucket_strategy=...)` 和 range 入口均委托此网格。
60m 不再要求段长整除；例如 CU 09:00–10:15 分为 60m+15m，后者标 partial。
`BarReturnLabeler` 和 `factor_bridge.session_grid` 接受同名策略。

`Lookback(sessions=N).to_bars(schedule, end_session=...)` 从锚点向前含当日累计实际槽位，
不足日历覆盖时报错；`Lookback(bars=N)` 返回指定根数。N 必须为正整数，二者互斥。
例如 CU 2026-01-05 无夜盘为 225 根、01-06 为 465 根，两 Session 合计 690 根。
6.1 没有取数或自动重采样；6.2 仅增加显式离线构建，见下文。

## 阶段 6.4b：国内期货本地验证与分桶推荐值

国内期货 `15m/60m` 由本地最细频率经 `build-bars` 离线构建，分桶规则属于项目约定：

- **推荐策略 `segment_anchored`**：桶不跨小节休息（10:15）、午休或夜盘→日盘间隔，
  不把休市跳空并入同一根 Bar；与股票默认一致，可用时间 `available_at_utc` 易推理。
- **可选策略 `session_clock`**：保留给需要等长桶（仅尾桶可 short）的研究显式指定。
- **第三种外部口径（天勤自然整点）**：只读外接盘天勤 `3600.parquet` 按自然整点切分
  （如 TA/M 日盘 `09:00/10:00/11:00/13:00/14:00`，其中 `13:00` 桶仅含 `13:30–14:00`），
  不作为生产分桶算法。另外，该天勤缓存的 `3600.parquet` 把周五夜盘记为周六自然日
  `trading_day`，而其 `86400.parquet` 与本地 PMC `session_id` 均为周一交易日。

本地外接盘只读验证样本为 `2025-04-02 .. 2025-05-09`（每品种 24 个 Session，覆盖清明/劳动节前
无夜盘、节后仅日盘、3 次周五夜盘→周一、AU/SC 跨午夜至 `02:30` 及 `M`/`TA` 换月日）：
- 聚宽 `1m`（`*9999` 连续，`CU/AU/M/TA/IF/SC`）：核实 `datetime` 为 `Asia/Shanghai`
  **Bar 结束时间**（减 1 分钟转 UTC 后 `0` off-grid）；144/144 Session 1m 根数与
  `expected_bar_opens` 精确匹配；32/32 周末/节前/跨午夜归属用例通过；`segment_anchored`
  与 `session_clock` 的 `build-bars` 冒烟 48/48 组完整率 `1.0`，`bar_quality` 全为 `complete`。
- 天勤 `TA/M` 交叉核对：45/45 非换月日由 `1m` 构建的 `1d` 与天勤 `86400.parquet`
  在 `high/low/close/volume/open_interest` 上 100% 精确一致（天勤无 `money` 列已跳过；
  `TA` 16 日仅 `open` 存在 2–12 元集合竞价口径差，`M` 对活跃合约 `DCE.m2509` 全字段精确）；
  `CU/AU/IF/SC` 无第二数据源，仅做上述自洽检查。
- **对齐边界**：以后若需把国内期货信号部署到聚宽或与聚宽回测对齐，须按
  `quantation-alignment` 单独核对外部 Bar 口径。

## 阶段 6.2：离线槽位映射

`build-bars` 对每个品种和 Session 以同一日历生成源/目标 `BarSchedule`，
验证目标开始/结束均在源边界上，再将槽位映射交给 DuckDB SQL 聚合。
休息不贡献源槽位；`session_clock` 桶可以跨休息，但不能切开源桶。
段末不足名义时长属于日历定义的有效桶；数据完整性按实际应有源槽位判断。
国内期货导入、累计量转换、可用性判断、构建与审计都接受显式 `bucket_strategy`。
日历只给应有槽位，不从观测数据补造缺失 Session 或价格。
