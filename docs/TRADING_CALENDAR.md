# 本地交易日历

本地研究的交易日由独立输入表提供，不从某只股票的行情、价格表有数据的日期或周一至周五推断。查询、前后 N 个交易日、数据版本和覆盖范围统一由 `utils/trading_calendar.py` 管理。日历查询不调用聚宽，也不联网。

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
