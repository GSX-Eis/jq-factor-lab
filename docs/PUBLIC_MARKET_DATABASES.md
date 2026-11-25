# 公开数据市场库

三个库各自持有 S7 profile，路径由 `QUANT_DATA_BINDINGS.market_dbs` 按 `db_id` 绑定。
这组快照用于统计推断研究。版本标识、原始响应清单和导入证据见本地归档 `reports/NERE-1270/`。

| `db_id` | 来源和样本 | 频率 | 交易日历 |
| --- | --- | --- | --- |
| `kf_us_portfolios_v1` | French 49 个行业的市值加权及等权组合、4 条 FF3 序列，从 1963-07-01 起 | `1d` | PMC XNYS，美东时间 |
| `crypto_binance_spot_v1` | Binance Vision 10 个 USDT 现货交易对，BTC/ETH 为 2024-11 至 2026-10，其余为 2026-10 | `1min`、派生 `1d` | UTC 00:00 开始的连续 24 小时会话 |
| `cn_index_akshare_v1` | 新浪宽基价格指数、申万 2021 版一级行业价格指数，从各自发布日期至 2026-10-31 | `1d` | PMC XSHG，上海时间 |

## 来源及使用条款

- French 的两份日频压缩包来自 Kenneth R. French Data Library。原始文件保留 Fama 与 French 版权声明，公开下载页未列出通用开放许可。按研究用途保存快照，引用时注明数据集、CRSP 版本和下载日期。 [数据目录](https://mba.tuck.dartmouth.edu/pages/faculty/Ken.french/data_library.html)
- Binance Vision 的月包公开下载，发布方提供同名 `.CHECKSUM`。原始和派生数据适用 Binance Vision Dataset Terms。 [格式和校验说明](https://github.com/binance/binance-public-data/blob/master/README.md) [数据条款](https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md)
- AKShare 库的代码使用 MIT 许可。AKShare 声明接口数据用于学术研究，新浪和申万行情按来源条款使用。快照保存 AKShare 返回的原始表格和实际库版本，宽基使用 `stock_zh_index_daily`，行业目录及行情使用 `index_realtime_sw` 和 `index_hist_sw`。 [AKShare 许可](https://github.com/akfamily/akshare/blob/main/LICENSE) [数据使用声明](https://github.com/akfamily/akshare/blob/main/docs/special.md) [指数接口](https://github.com/akfamily/akshare/blob/main/docs/data/index/index.md)

## 下载、冻结与重建

在仓库根目录执行。每个来源使用单独、可写的工作目录，同一工作目录一次运行一个进程。
现有快照通过 `raw/sha256.json` 校验，清单逐文件记录来源、UTC 下载日期和 SHA-256。
未完成的下载可在同一目录重试。原始文件和清单应一起长期保留。用相同版本的依赖重建时，
先把完整的 `raw/` 复制到新工作目录，再执行同一条命令。上游可能修订或下架数据，重新联网
下载会产生新快照。确需更新时使用新的工作目录并记录新版本。

```bash
root=work/NERE-1270
python3 -m data_pipeline.extractors.public_markets.french \
  --work-dir "$root/french" --db "$root/french/market.duckdb"
python3 -m data_pipeline.extractors.public_markets.binance \
  --work-dir "$root/binance" --db "$root/binance/market.duckdb"
python3 -m data_pipeline.extractors.public_markets.akshare_indices \
  --work-dir "$root/akshare" --db "$root/akshare/market.duckdb"
```

`--prepare-only` 可先获取快照、生成 `normalized/` 下的 UTC Bar CSV 和标的 JSON 并查看预检报告。
导入脚本通过 Harness 预演并创建新库，检查 profile 哈希，随后对每批运行 `import-bars --bars --metadata`
及其 `--dry-run`。Binance 日线通过 `build-bars --source 1min --targets 1d` 派生。提交了一批后，
相同快照的重试会根据库中已导入的版本和行数接续。S7 没有建视图的 Harness 命令，脚本在自管库创建
`public_bars_1d` 或 `public_bars_1min` 的只读列投影。随后按 `db_id` 调用 `market-db show`，比对库内与
YAML 的 `profile_hash`，并打开所有对应 Slot。过程证据在 `harness/`，汇总在 `import_report.json`。

新建研究会话时，将这些映射加入现有 `market_dbs` 对象。如果使用相对路径，从仓库根目录运行。

```bash
export QUANT_DATA_BINDINGS='{"market_dbs":{"kf_us_portfolios_v1":"work/NERE-1270/french/market.duckdb","crypto_binance_spot_v1":"work/NERE-1270/binance/market.duckdb","cn_index_akshare_v1":"work/NERE-1270/akshare/market.duckdb"}}'
python3 -m tools.agent_harness.cli market-db show --market-db kf_us_portfolios_v1
```

仓库内 YAML 是路径无关的 Slot 声明，`bar_feed.options.data_version` 默认表示配置代际。
每次导入生成 `slots/*.json`，写入原始输入版本，并指向包含 `database_snapshot_id` 的读取视图。
该标识来自原始输入版本及各频率实际入库的标的、UTC 时间、OHLCV 和完整性标记，记录在 `import_report.json`。
研究命令的 `--slot-config` 应使用这份快照配置，并同时带上 `--market-db`。
同一原始版本的样本库和全量库有各自的内容标识，交叉绑定时读取会因缺少对应视图而失败。
脚本在发起写入型导入或日线派生前撤下已发布的研究视图。只读预演和直接跳过的已提交批次保留原视图。
若写入或发布中断，研究读取会停止。重试时可从已提交的物理 Bar 续跑，完成发布后使用新生成的 Slot。
同库重新生成 Slot 时会移除过期的版本视图，包括早期仅按原始输入版本命名的视图。使用该库最新生成的 Slot。

## 序列定义和缺口

### French

源文件每天给出的百分比除以 100 后，以 1 为初始指数水平，按 `∏(1+r)` 计算。
`-99.99`、`-999` 读取为缺失，省略当日 Bar。下一条观测以此前最后一个已知指数水平
续算，仅应用该条日收益。四个 OHLC 字段都保存同一指数收盘，成交量置零。
Slot 声明为收盘到收盘，并使用 `require_complete_bars`，避免评测将跨缺口的收益视作连续完整区间。
`Mkt-RF` 是市场超额收益，SMB 和 HML 是多空因子收益，RF 是日无风险利率。
四条指数水平都只是原序列的编码。

| 原始列 | 标的 ID |
| --- | --- |
| 行业列，如 `Agric` | `AGRIC_VW.KFDL`、`AGRIC_EW.KFDL`，其他列同样按大写列名加权重后缀 |
| `Mkt-RF` | `MKT_RF.KFDL` |
| `SMB` | `SMB.KFDL` |
| `HML` | `HML.KFDL` |
| `RF` | `RF.KFDL` |

各 Bar 的 `data_version` 还含所属原始 ZIP 的下载日期及完整 SHA-256。

### Binance

最初设想从各交易对首个完整月全部保留。以约 3,600 万分钟乘每行 220 B 估算，仅规范 CSV 就超过
5 GB，任务窗口据此缩短。BTC、ETH 保留连续 24 个月，BNB、XRP、SOL、ADA、DOGE、LINK、LTC、AVAX
保留最近完整月 2026-10。以每行 1,200 B 预留 ZIP、CSV、数据库及导入临时空间，运行前检查预算。

每根 K 线时间戳按位数区分毫秒和微秒。提供方停机或缺失的分钟保持缺失，汇总时区固定为 UTC。
S7 profile 的币种字段限三位大写字母。该库用内部代码 `XUT` 表示 USDT，Slot 标的币种保留 `USDT`。
库内日线使用当日首条开盘、最高、最低、最后收盘和分钟成交量总和。S7 在物理日线表保留
`is_complete`。研究视图只返回完整日，物理表和对账报告保留不完整日。官方 `1d` 包只用来核对
2026-10 的完整天，报告另列缺日及不完整日。

### AKShare

申万目录和历史响应都使用数字代码，如 `801010`。仓库为其分配 `801010.SW` 和提供方交易场所 `SW`。
新浪宽基使用 `XSHG` 或 `XSHE` 后缀。行业范围固定为 2021 版的 31 个一级行业，起点为 2021-12-13。
宽基从发布日开始，发布前提供方回溯记录保留在原始快照，不进入规范 Bar。
[上证指数发布日期](https://www.sse.com.cn/aboutus/publication/factbook/documents/c/10170564/files/300a2dcef0f0452794683bd1bfbf65b0.pdf) [沪深300](https://www.szse.cn/aboutus/trends/news/t20110119_517266.html) [创业板指数](https://www.szse.cn/disclosure/notice/general/t20100531_500454.html) [中证规模指数日期](https://github.com/akfamily/akshare/blob/main/docs/data/index/index.md)

日历使用仓库固定版本的 PMC XSHG。导入报告逐标的记录原始历史跨度和发布后范围的 XSHG 差异。
仅日历有、行情缺少的交易日保留缺口。发布后若出现来源有、XSHG 没有的日期，导入会暂停并列出日期。
