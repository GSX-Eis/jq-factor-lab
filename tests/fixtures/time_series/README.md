# SYNTHETIC 时序研究验收夹具

本目录的全部行情、因子、预期 JSON 都是人为构造的 **SYNTHETIC** 数据，
不是 NQ 真实行情。`source=SYNTHETIC`，固定 `data_version=SYNTHETIC_NERE704_v1`；
空因子表也保留来源和版本列，身份由本说明与预期 JSON 声明。
预期值来自以下手算表，没有调用待测日历、读取器、标签或评估代码。
验收后预期冻结，阶段 2–4 只接线 `tests/time_series_acceptance_support.py` 并移除
相应 xfail 标记；修改预期须经大古同意。

## 时间、列与候选范围

- 品种固定 `NQZ26`，频率 `1min`，全部 CSV 时间为 `2026-12-01` 的 UTC **Bar 开始时间**。
  下文简写 `14:30` 即 `2026-12-01T14:30:00Z`。该日是周二，纽约冬令时 UTC−5。
- RTH 为半开区间 `[14:30, 21:00)`，即 09:30–16:00 ET；维护休市 `[22:00, 23:00)`。
  ETH 在维护期间没有应有 Bar，23:00 是下一交易日（2026-12-02）的首根 Bar。
  ETH 标签可跨 RTH 开盘/收盘，不能跨 ETH 会话边界；RTH 信号须先筛选，
  非 RTH 信号不进入长表，不能计为剔除。
- 每根正常 Bar 都是 `open=101, high=107, low=100, close=106, volume=1`。
  唯一异常列在场景表标出。正常因子值恒为 1，`NaN` 文本是 CSV 缺失值。
  所有信号 `k=1, h=5`；`signal_available_at` 是信号开始后一分钟。
- `expected_labels.json` 每个 case 指定输入文件、信号集合和每个视图的预期完整长表。
  普通 case 只观察指定信号；`view_difference` 观察输入每一根 Bar，供端到端使用。
  输入数据覆盖末端取原 CSV 最后一根 Bar，**在视图筛选前确定**。
  会话内超出覆盖末端为 `insufficient_history`，覆盖内缺口才是 `missing_bar`。
- 被剔除记录保留信号与可知时间，其余时间、价格和收益输出统一 JSON `null`。
  `reasons_by_view` 是用于复核优先级的全部并发原因，实际标签只记一个主原因。
  原因优先级：`factor_missing > insufficient_history > session_boundary > missing_bar > invalid_price`。

## 逐场景构造和手算

| case | Bar 文件（均带 `SYNTHETIC_` 前缀）及全部行 | 因子输入 | 逐步核对与结果 |
|---|---|---|---|
| `valid_five` | `valid_bars.csv`：14:30、31、32、33、34、35 | `factors.csv`：14:30=1 | ① 信号14:30，可知14:31；② k=1 入场14:31 open=101；③ 五根持有Bar为31/32/33/34/35；④ 退出Bar14:35 close=106，实际退出14:36；⑤ 106−101=5点，5÷101×100=4.9504950495049505%，`valid`。 |
| `rth_boundary` | `rth_boundary_bars.csv`：20:57、58、59、21:00、01、02 | `factors.csv`：20:57=1 | ① 信号20:57，可知/入场20:58；② 五根计划Bar为20:58、59、21:00、01、02；③ 原始覆盖到21:02，非末尾不足；④ 21:00起不属RTH，不能从ETH借Bar，`session_boundary`。 |
| `eth_boundary` | `eth_boundary_bars.csv`：21:57、58、59、23:00、01、02、03 | `factors.csv`：21:57=1 | ① 信号21:57，可知/入场21:58；② 连续应有Bar序号下五根为21:58、59、23:00、01、02；③ 维护时间无应有Bar，不是缺Bar；④ 覆盖到23:03但跨过22:00–23:00维护及交易日边界，`session_boundary`。 |
| `missing_bar` | `missing_bars.csv`：14:30、31、32、34、35 | `factors.csv`：14:30=1 | ① 信号14:30，可知14:31；② 计划31/32/33/34/35；③ 末端35存在而会话内33缺失；④ 不补价、不压缩成四根、不顺延到36，`missing_bar`。 |
| `invalid_zero` | `zero_bars.csv`：同valid，但14:33 close=0 | `factors.csv`：14:30=1 | ① 计划31/32/33/34/35齐全；② 中间Bar33 close=0≤0（同时违反low≤close）；③ 即使端点101/106正常也剔除，`invalid_price`。 |
| `invalid_nan` | `nan_bars.csv`：同valid，但14:33 close=NaN | `factors.csv`：14:30=1 | ① 持有期完整；② 中间Bar33收盘非有限；③ 不插值或删除异常行，`invalid_price`。 |
| `insufficient_history` | `tail_bars.csv`：14:30、31、32、33 | `factors.csv`：30/31/32/33均=1 | 逐行信号/可知/所需退出Bar为14:30/31/35、31/32/36、32/33/37、33/34/38；每条所需末Bar均晚于原始末端14:33，且仍在RTH内，四条全为`insufficient_history`。 |
| `factor_nan` | `valid_bars.csv`，六根完整 | `nan_factors.csv`：14:30=NaN | ① 信号14:30未来五根完整；② 因子该行存在但值NaN；③ 最高优先级`factor_missing`。 |
| `factor_absent` | `valid_bars.csv`，六根完整 | `absent_factors.csv`：仅表头 | ① 价格同valid；② 按(14:30,NQZ26)找不到因子行；③ `factor_missing`。 |
| `factor_and_gap` | `missing_bars.csv`：缺14:33 | `nan_factors.csv`：14:30=NaN | ① 14:30因子缺失；② 31/32/33/34/35中的33缺Bar，且35存在；③ 并发`factor_missing + missing_bar`；④ 只记前者，不重复计数。 |
| `tail_and_boundary` | `tail_boundary_bars.csv`：21:57、58、59 | `factors.csv`：21:57=1 | ① 信号21:57，可知21:58；② 五根应有序号需21:58、59、23:00、01、02；③ 后三根超过末端21:59，且跨维护边界；④ 并发`insufficient_history + session_boundary`，只记前者。 |
| `view_difference` | `view_bars.csv`：14:27、28、29、30、31、32、33、34、35 | `factors.csv`：这九个时间均=1 | 下表逐行列出；同一原始文件分别走ETH与RTH，绝不以RTH开盘截断ETH标签。 |

`factors.csv` 的全部时间为 14:27、28、29、30、31、32、33、34、35、20:57、21:57，
值均为1；不在本 case 信号集合的因子行不产生额外信号。

### ETH/RTH 共用数据的逐行长表与计数

| 信号 | 可知 | ETH入场 | ETH退出Bar / 实际退出 | ETH结果 | RTH结果 |
|---|---|---|---|---|---|
| 14:27 | 14:28 | 14:28 | 14:32 / 14:33 | valid，101→106，5点，4.9504950495049505% | 不在候选内，无记录 |
| 14:28 | 14:29 | 14:29 | 14:33 / 14:34 | valid，同上算术 | 不在候选内，无记录 |
| 14:29 | 14:30 | 14:30 | 14:34 / 14:35 | valid，同上算术 | 不在候选内，无记录 |
| 14:30 | 14:31 | 14:31 | 14:35 / 14:36 | valid，同上算术 | valid，同左 |
| 14:31 | 14:32 | — | 需14:36 / 14:37，超末端35 | insufficient_history | insufficient_history |
| 14:32 | 14:33 | — | 需14:37 / 14:38，超末端35 | insufficient_history | insufficient_history |
| 14:33 | 14:34 | — | 需14:38 / 14:39，超末端35 | insufficient_history | insufficient_history |
| 14:34 | 14:35 | — | 需14:39 / 14:40，超末端35 | insufficient_history | insufficient_history |
| 14:35 | 14:36 | — | 需14:40 / 14:41，超末端35 | insufficient_history | insufficient_history |

`expected_evaluation.json` 固定：ETH = 4 valid + 5 excluded = 9候选；
RTH = 1 valid + 5 excluded = 6候选；两视图剔除原因计数均仅 `insufficient_history: 5`。

### 重复与乱序读取

- `duplicate_bars.csv` = valid六行后再附一行14:32；重复键是
  `(NQZ26, 2026-12-01T14:32:00Z, 1min)`。读取抛 `DuplicateKeyError`，不能静默去重。
- `unsorted_bars.csv` 六行顺序为14:33、30、35、31、34、32；按键排序后逐列逐值等于
  `valid_bars.csv`；14:30信号标签也必须与valid相同，不因行位置偏移而改变。
- `sorted_factors.csv` 为14:30、31、32；`unsorted_factors.csv` 为14:32、30、31，
  值均1，排序后完全相同。`duplicate_factors.csv` 为14:30、31、30，
  重复键 `(2026-12-01T14:30:00Z, NQZ26)` 同样必须拒绝。
- 适配层读取结果使用RangeIndex比较CSV行内容；契约层的表校验仍保留调用方原索引，
  自检在比较排序内容前显式reset_index，不修改已有契约。

## 日历小窗口的手算预期

`expected_calendar.json` 使用半开时间区间；无需整日390/1380行也能逐根核对边界。

| 查询 | 手算结果 |
|---|---|
| RTH会话归属 | 14:29无、14:30属2026-12-01、20:59属2026-12-01、21:00无 |
| ETH会话归属 | 14:27及21:59属2026-12-01；22:00及22:59维护无归属；23:00属2026-12-02 |
| `[14:28,14:33)` ETH | 14:28、29、30、31、32，共5根；14:33为右端点不含 |
| 同区间RTH | 去掉开盘前28、29，得14:30、31、32，共3根 |
| `[21:58,23:02)` ETH | 21:58、59，然后跳过维护，23:00、01，共4根；跨日历会话的网格查询允许列出两边，但标签禁止跨会话 |
| 同区间RTH | 收盘21:00之后，无应有Bar，空列表 |

## 测试及后续接线

`test_time_series_fixtures.py` 现在就通过，验证来源版本、JSON标准格式、结构、
LabelRecord不变量、101/106算术、优先级、缺口/异常的原始证据、乱序无损和计数一致。
`test_time_series_acceptance.py` 通过适配层调用六个尚未实现的能力；
每项均strict xfail且只允许`NotImplementedError`，导入/断言失败不会被掩盖。
阶段2接日历/CSV，阶段3接标签，阶段4接完整评估；实现完成后strict XPASS要求移除标记。
本阶段不实现这些能力，不增加依赖，不加载真实数据。
