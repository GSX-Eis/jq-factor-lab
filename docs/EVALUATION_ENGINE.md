# 统一评测引擎

收益标签由交易约定决定，评测轴由研究问题决定。日频标签和 Bar 标签都可用于截面或时序评测。评测按因子、频率、持有期和样本切分分别报告，价格与时间输入遵循仓库既有契约。 [时序研究契约](TIME_SERIES_CONTRACT.md) · [交易日历](TRADING_CALENDAR.md) · [评测政策](EVAL_POLICY.md)

## 目录

- [1. 前瞻收益标签](#1-前瞻收益标签)
- [2. 成对有效样本与秩](#2-成对有效样本与秩)
- [3. 截面评估](#3-截面评估)
- [4. 单品种时序评估](#4-单品种时序评估)
- [5. 方向准确率](#5-方向准确率)
- [6. 重叠收益反例与区间校准](#6-重叠收益反例与区间校准)
- [7. 不可估计状态](#7-不可估计状态)
- [8. 兼容行为与有意差异](#8-兼容行为与有意差异)
- [9. 夹具索引](#9-夹具索引)
- [10. 性能基线](#10-性能基线)
- [11. 内核接口](#11-内核接口)
- [12. 截面统计接口](#12-截面统计接口)
- [13. 时序统计接口](#13-时序统计接口)
- [14. 训练与样本外评测接口](#14-训练与样本外评测接口)
- [15. 统一评测入口](#15-统一评测入口)
- [16. S7 时序诊断与 NQ 1min 验收](#16-s7-时序诊断与-nq-1min-验收)

## 1. 前瞻收益标签

令 $t$ 为信号所在的交易日或 Bar 序号，$k$ 为整数入场偏移，$h\ge 1$ 为整数持有期。下标使用日历应有序列上的位置。

| 标签 | 入场时间及价格 | 退出时间及价格 | 输出 |
| --- | --- | --- | --- |
| 日频，$k\ge 0$ | $t+k$ 交易日收盘，收盘价 $C_{t+k}$ | $t+k+h$ 交易日收盘，收盘价 $C_{t+k+h}$ | `hD`，单位为小数 |
| Bar，$k\ge 1$ | 第 $t+k$ 根 Bar 开始时间，开盘价 $O_{t+k}$ | 第 $t+k+h-1$ 根 Bar 实际结束时间，收盘价 $C_{t+k+h-1}$ | `return_pct`，单位为百分数 |

$$
r^{\mathrm{daily}}_{t,h}=\frac{C_{t+k+h}}{C_{t+k}}-1,\qquad
r^{\mathrm{bar,pct}}_{t,h}=100\left(\frac{C_{t+k+h-1}}{O_{t+k}}-1\right).
$$

日频收益仍归到信号日 $t$ 和资产键。`ForwardReturnsCalculator` 接收收盘价序列，输出日期键和收益列。入场、退出用无时区的交易日键表示，其成交时刻均为交易所当日实际收盘。`execution_lag=0` 表示信号日收盘入场，使用者须确保信号在该成交时刻可用。

在现有 `BarReturnLabeler` 中，信号键 `signal_timestamp` 为信号 Bar 开始的 UTC 时刻，`signal_available_at` 为信号 Bar 实际结束的 UTC 时刻。`entry_timestamp` 为入场 Bar 开始，`exit_bar_timestamp` 为退出 Bar 开始，`exit_timestamp` 为退出 Bar 实际结束。成交价格分别记为 `entry_price` 和 `exit_price`。`price_change_points` 等于退出价减入场价，单位为价格点。

跨标签的统计公式统一使用小数收益 $r$，即 `return_pct / 100` 用于 Bar；Bar 标签原字段仍用百分数。IC、ICIR 与方向命中率不随收益单位的正比例缩放改变。收益、价差及收益标准误对外报告时须标明单位。

### 交易日历、缺口与换月

- 日频沿核验过的开市日推进。休市日不占位，提前收市日仍占一个交易日。应有交易日即使没有任何观测也保留位置。入场或退出收盘价缺失则该标签缺失。日频公式只要求两端收盘价齐全，缺少中间收盘价时仍使用原定两端。
- Bar 按资产、频率、所选 ETH/RTH 视图在交易所应有网格上推进。休市、午休和交易时段中断不占槽位。提前收市以该交易日历规定的实际边界为准，桶的结束时间取调度结果。夏令时按交易所时区生成时段后转换为 UTC，同一当地时刻的 UTC 偏移可随会话改变。
- 只有所选视图网格上已观测的信号 Bar 才进入标签候选。信号 Bar 完全缺行时，该资产、频率和时点在各持有期均产生零条标签记录，候选数和剔除数均不增加。该槽位仍属于应有网格。
- 已观测但标为不完整（`is_complete=False`）的信号 Bar 进入候选，触发 `missing_bar` 判定。对已进入的候选，覆盖范围内 $[t+1,t+k+h-1]$ 中任一应有 Bar 缺行或不完整也触发 `missing_bar`。原定入场和退出槽位始终由 $k,h$ 决定。持有窗口 $[t+k,t+k+h-1]$ 的异常 OHLC 触发 `invalid_price`。覆盖末端不足触发 `insufficient_history`，违反所选会话边界触发 `session_boundary`（默认 `same_session`）。每条候选按 `factor_missing`、`insufficient_history`、`session_boundary`、`missing_bar`、`invalid_price` 的顺序取首个命中的原因，剔除数只计一次。
- 连续期货的信号绑定信号时刻的具体合约。只有连续合约入口可显式启用 `cross_session`。跨换月窗口仍读取信号合约原始 Bar 的入场开盘价与退出收盘价。换月价差不计入盈亏，信号合约缺一根仍按缺口规则处理。连续图上的拼接价或复权价可构造因子，标签价格取自原始合约。日频收盘标签若用于期货，也须先按信号合约取得同一合约的两端原始收盘价。使用旧日频计算器时，按具体合约提供收盘序列，再由调用方按各信号的合约选取对应标签。

同一组手算价格在 $k=h=1$ 时给出两个结果。$t+1$ 开盘 100、收盘 105，$t+2$ 收盘 103.95。日频成交于两个交易日的收盘，$103.95/105-1=-0.01$。Bar 成交于 $t+1$ 同一根 Bar 的开盘和收盘，$100(105/100-1)=+5.0$。

### 统一标签入口

`backtest_and_analysis.evaluation.labels.label_forward_returns` 要求调用方显式传入成交约定、入场偏移 `entry_offset` 和持有期 `horizons`。`ExecutionConvention.CLOSE_TO_CLOSE` 对应日频收盘到收盘，`ExecutionConvention.OPEN_TO_CLOSE` 对应 Bar 开盘到收盘。每种约定定义成交价格字段、允许的最小入场偏移、退出 Bar 或交易日相对入场的网格偏移、收益单位和缺口规则。两种约定的 `horizons` 均接受正整数或由正整数组成的非空可迭代对象，包括一次性迭代器。

- 日频接收以日期和资产为键的收盘价序列，沿用显式日历、价格序列属性、默认交易日历的选择顺序。`DailyReturnLabels.returns` 是以 `(date, asset)` 为索引的 `1D/5D/…` 小数收益表。`DailyReturnLabels.endpoints(horizon)` 按需返回该持有期有效信号的 `entry_date`、`exit_date`、`entry_price` 和 `exit_price`。两个日期字段使用交易日键。
- Bar 入口接收 Bar 表与会话日历，可指定因子表、信号时点、视图、会话边界、分桶及连续合约源，返回 `BarReturnLabeler.label` 的标准长表。Bar 表包含实际成交时刻、价格、百分数收益和剔除原因。视图、网格和换月逻辑由 `BarReturnLabeler` 处理。
- `ForwardReturnsCalculator.calculate_forward_returns` 调用统一标签层的日频实现并返回 `DailyReturnLabels.returns`。计算器的 `periods` 仍要求可迭代对象，旧 `BarReturnLabeler` 的 `horizons` 仍要求整数或序列。缓存继续使用原来的收益表格式，查询成交端点走统一入口。

## 2. 成对有效样本与秩

在信号时点 $t$，$U_t$ 是当时已知的候选资产池，$x_{ti}$ 是可用因子值，$r_{ti,h}$ 是第 $h$ 个标签。每个持有期独立建立掩码。

$$
M_{ti,h}=\mathbf 1\{i\in U_t,\ x_{ti}\text{ 有限且可用},\
                      r_{ti,h}\text{ 有限且标签有效}\}.
$$

信号须属于当前切分。训练样本的真实退出时间须早于或等于下一切分的首个信号时刻。任一持有期的缺失只影响其自身掩码。截面按时点取出有效数组，时序按单品种沿时间取出有效数组，两者均在取出成对样本后分别排名，并列取平均秩。

$$
\operatorname{IC}(x,r)=
\operatorname{Corr}_{\rm Pearson}(\operatorname{rank}_{\rm avg}(x),
                                  \operatorname{rank}_{\rm avg}(r)).
$$

有效因子与收益各自必须有变异。IC 的最低成对样本数沿用各入口的默认口径。

| 入口 | 默认门槛 |
| --- | --- |
| 日频截面（`AlphalensRunner` 兼容口径） | 同一交易日至少 2 个有效资产。旧路径清洗后若有 2 对且因子与收益各自非恒定，保留该截面的 IC。 |
| Bar 截面（`BarCrossSectionalEvaluator`） | 同一信号时点至少 3 个有效资产（`min_cross_section_size=3`）。 |
| 单品种时序（`TimeSeriesEvaluator`） | 至少 3 个有效时点（`min_ic_samples=3`）。 |

输入不足或常数时按第 7 节给出状态。

### 训练与样本外切分

`split: {train_end, oos_start}` 要求 `train_end < oos_start`。训练取信号可用时间
不晚于 `train_end` 的槽位，样本外取不早于 `oos_start` 的槽位，中间的槽位不参与
评测。日频面板也可以用交易日作为两个边界，此时按交易日归属切分。两段各自在
应有网格上计算统计量，保留段内缺失槽位。

每个持有期以训练信号对应的实际成交退出时刻和首个样本外应有信号的
`signal_available_at` 比较。提供 `expected_grid` 时，首个应有信号时刻取
至少一个品种有应有槽位的首行，即使该行因子缺失。退出早于或等于该时刻时
保留有效收益，晚于该时刻时清除。训练和样本外在相等的端点共享这笔已知收盘价。清除计数
`split_purged_count[h]` 统计可用训练信号中本期计划退出越界的键，计划退出
时刻由标签日历给出，因此越界价格后来缺失时计数仍保持一致。每个持有期独立
清除，信号自身仍可参与分桶拟合。日频用退出交易日的实际收盘时间，Bar
用退出 Bar 的实际结束时间，均包括日历规定的提前收市。

### 多周期特征发布时间

低周期信号只可使用同时满足 `high_tf_bar_end <= signal_available_at`
和特征自身发布时间不晚于当前信号的高周期 Bar。恰好收盘时已发布的特征
可以使用，延迟发布的特征在到达发布时间后可用。已发布的 Bar 中选择收盘
时间最新的一根，较旧 Bar 的延迟到达不会覆盖更新的已知值。时间比较统一到
纳秒精度，保留低周期信号的原始时间索引。

## 3. 截面评估

一个截面是同一信号时点的跨资产样本。在每个时点独立计算 $I_t=\operatorname{IC}(x_t,r_{t,h})$。有效 IC 时点集合为 $V$，$n=|V|$。时点等权，截面内样本数单独报告。

$$
\bar I=\frac1n\sum_{t\in V} I_t,\qquad
s_I=\sqrt{\frac{\sum_{t\in V}(I_t-\bar I)^2}{n-1}},\qquad
\mathrm{ICIR}=\frac{\bar I}{s_I}.
$$

中位 IC 取 $\{I_t:t\in V\}$ 的中位数，正 IC 比例取 $\#\{t\in V:I_t>0\}/n$。$n=1$ 可报告均值、中位数与正值比例。标准差和 ICIR 需要 $n\ge2$，ICIR 还要求 $s_I>0$。

### 分桶收益与多空价差

截面分桶默认采用 `per_cross_section`。每个 $t$ 只用 $U_t$ 内当时可用的有限因子值，以 `pd.qcut(..., duplicates='drop')` 的等分位数规则形成右闭区间，含最小端点。相同因子值同桶，重复边界合并，桶按因子值从低到高自 1 编号，并报告实际桶数。标签缺失或持有期变化不触发重切桶。常数因子或无法形成至少两个桶的截面不给出分桶统计。

令 $S_{t,h}=\{i:M_{ti,h}=1\}$，$S_{q,t,h}$ 为其中位于桶 $q$ 的资产，$n_{q,t,h}=|S_{q,t,h}|$。截面基准为当前持有期所有成对有效资产的等权均值。

$$
\bar r_{t,h}=\frac{\sum_{i\in S_{t,h}}r_{ti,h}}{|S_{t,h}|},\qquad
B^{\rm raw}_{q,t,h}=\frac{\sum_{i\in S_{q,t,h}}r_{ti,h}}{n_{q,t,h}},\qquad
B^{\rm excess}_{q,t,h}=B^{\rm raw}_{q,t,h}-\bar r_{t,h}.
$$

同日所有有效资产收益加同一个常数时，原始桶收益平移该常数，截面超额桶收益保持原值。分桶结果须声明 `raw` 或 `cross_section_excess` 口径。旧日频主路径调用 `Alphalens.mean_return_by_quantile(by_date=True)`，沿用默认 `demeaned=True`，在旧清洗样本上先减当日截面均值。手工回退对当日桶内原始收益直接取均值。两条旧路径的差异已登记于第 8 节。 [Alphalens 实现](https://github.com/quantopian/alphalens/blob/master/alphalens/performance.py)

令 $V_q=\{t:n_{q,t,h}>0\}$。对每个桶按时间等权取 $\bar B_q=|V_q|^{-1}\sum_{t\in V_q}B_{q,t,h}$，并报告 $|V_q|$。例如两时点同桶资产数分别为 2 和 10，桶均收益为 0% 和 6%，时间等权结果为 3%。

多空价差按指定的高桶 $q_H$ 与低桶 $q_L$ 计算。默认使用请求的最高桶与最低桶。在二者同一时点均有有效收益时计算 $D_t=B_{q_H,t,h}-B_{q_L,t,h}$，按配对时点等权取 $\bar D$ 并报告配对时点数 $m$。描述性标准误为 $s_D/\sqrt m$，其中 $s_D$ 是配对价差的样本标准差（`ddof=1`）。单桶跨时点标准误同样用桶均值序列的样本标准差除以有效时点数的平方根。至少需要两个时点才能给出相应标准误。

### IC 均值的 HAC 置信区间

统一入口默认使用 `cs_hac_adaptive_v1`。它在完整时间网格上做 AR(1) 预白化，
以间隔幂处理跨缺口转移，依据加权创新选取 Bartlett 滞后。滞后下界为各持有期的
`h - 1`。去均值后的二次型经矩修正，区间与双侧 p 值共用 Student t 分布和有效自由度。
`HACInterval` 报告版本、实际滞后、预白化系数、有效自由度、临界值和缺口模式。
显式整数窗口使用以下固定 Bartlett HAC。

时间轴是预先冻结的应有信号网格（交易日或按日历定义的 Bar 槽位）。跨市场截面取预先绑定的各市场网格的并集。计划休市不占位，网格内无有效 IC 的槽位仍占位。设网格大小为 $N$，$m_t=\mathbf 1\{t\in V\}$，构造中心化得分

$$
z_t=\begin{cases}I_t-\bar I & m_t=1,\\ 0 & m_t=0.\end{cases}
$$

这里的零是缺失观测对协方差乘积的零贡献。IC 序列中的该值仍为缺失。给定分析预先指定的 Bartlett 滞后 $L$，按原网格索引（从 1 开始） 计算

$$
\widehat{\operatorname{Var}}(\bar I)=
\frac{1}{n^2}
\left[
\sum_{t=1}^{N}z_t^2+
2\sum_{\ell=1}^{L}
 \left(1-\frac{\ell}{L+1}\right)\sum_{t=\ell+1}^{N} z_t z_{t-\ell}
\right].
$$

滞后 $\ell$ 的内和只由相隔 $\ell$ 个网格位置的有效配对贡献。逐滞后记录有效配对数，禁止先 `dropna` 再计算邻接关系。$n<2$、$L\ge N$ 或方差非有限、非正时，区间按第 7 节标为缺失。其余情况用正态近似构造 $1-\alpha$ 区间 $\bar I\pm\Phi^{-1}(1-\alpha/2)\sqrt{\widehat{\operatorname{Var}}(\bar I)}$。缺失时该区间针对观测 IC 的均值，依赖性与观测机制仍需满足推断条件。 [Datta 与 Du，缺失观测下的 HAC](https://www.federalreserve.gov/pubs/ifdp/2012/1060/ifdp1060.htm)

## 4. 单品种时序评估

在固定品种、频率、持有期和切分内，样本是沿应有时间网格的因子—收益对。每个有效时点权重相同。点估计为在全部有效时点上一次计算的 Spearman IC。

有训练/OOS 切分时，时序分桶采用 `fitted_on_train`。用训练期可用且有限的因子值计算 $j/Q$（$j=0,\ldots,Q$）经验分位点，按排序后的位置 $(n-1)j/Q$ 线性插值，去掉重复边界。至少须得到两个区间。报告实际桶数，冻结内部分界，并将应用时两外端延伸至 $-\infty$ 与 $+\infty$。内部分界上的值进入左侧桶（右闭约定）。样本外超出训练值域的值进入端点桶，空桶保留原编号及零计数。应用时不重新拟合或编号。每桶在当前切分内用有有效标签的时点计算收益等权均值等统计，标签缺失不改变桶号。

方向 `d` 和可选标准化随分位拟合结果冻结。方向为调用方预先确定的
`+1` 或 `-1`，默认 `+1`。若用数据选择方向，选择过程须局限于训练样本。
标准化按品种使用与训练分位相同的因子样本，可选 `zscore`（均值、总体标准差）
或 `median_mad`（中位数、未缩放的绝对偏差中位数）。常数样本以该常数
为精确中心，零尺度以 1 代替。
标准化是正尺度仿射变换，IC 与分桶继续在原量纲上计算，以保留原有的秩、
并列值与精确分界。方向准确率使用标准化后的符号，直接比较因子值与冻结中心。
启用标准化且品种无可用训练因子时，该品种无法拟合分桶，方向信号弃权。

训练和样本外结果各自附带同一个 `fit_identity` 内容。身份字段包含方案与
版本、请求桶数、方向、标准化方法，以及每个品种的有限训练边界、中心和尺度。
`sha256` 由这些字段按键排序、紧凑分隔的 UTF-8 JSON 计算，哈希字段本身
不参与计算。实际应用边界由训练内部分界和延伸端点确定。截面方案逐时点分桶，
`fit_identity` 为 `None`。

### IC 的块 bootstrap 置信区间

统一入口默认使用 `ts_boot_adaptive_v2`。它用标准化秩乘积的依赖结构选择块长，
以当前持有期的 `h` 为下界。圆形块复制保留缺口并重新排名。原样本在保留缺口的
网格上用经自由度校正的平顶窗估计 Fisher-z 标准误，平顶谱底未过护栏时回退到
经自由度校正的 Bartlett 估计。复制样本用独立完整块的贡献及残块权重估计标准误，Fisher-z 尺度的
学生化枢轴经反演后给出 Spearman 区间。默认请求 999 次复制，至少 900 次有效。
`BootstrapInterval` 报告版本、抽样与区间方法、实际块长、有效块数和复制计数。
校准实验的实现与参数见 [tools/benchmarks/adaptive_inference_calibration.py](../tools/benchmarks/adaptive_inference_calibration.py)。
显式整数块长使用以下移动块百分位区间。

在该品种当前切分的应有网格上做移动块 bootstrap。设网格有 $N$ 个槽位，原始 IC 可估计且 $h\le b<N$ 时，取长度 $b$ 的连续块。从 $N-b+1$ 个可用起点均匀有放回抽取 $\lceil N/b\rceil$ 个块，连接后截取前 $N$ 个槽位。因子、收益和有效掩码共同移动，缺失槽位占据原块位置，块不跨切分边界，也不循环拼接原序列首尾。

每次重采样在拼接后的全样本上重新取成对有效子集、计算两套平均秩，并重算一个完整的 Spearman IC。无有效统计量的复制记为无效并计数。预先记录块长、随机种子、重复次数、最少有效复制数（至少为 2）及置信水平。达到最少有效复制数后，以有效复制 IC 的 $\alpha/2$ 与 $1-\alpha/2$ 经验分位点（线性插值）作为区间端点，否则区间缺失。点估计仍来自原始样本。原始 IC 缺失时，其区间沿用同一缺失原因。

非重叠诊断按应有信号 Bar 网格测试全部 $h$ 个起始偏移。每组先按网格相位选样，再应用该持有期有效掩码，逐组报告样本数和 IC。每组用实际入场与退出时间校验半开持仓区间互不相交。

## 5. 方向准确率

因子先应用预先冻结的方向 $d\in\{-1,+1\}$，默认 $+1$。信号 $s=\operatorname{sgn}(d(x-c))$，未标准化时 $c=0$，启用训练标准化时 $c$ 为冻结中心。$s=+1$ 为多，$s=-1$ 为空，$s=0$ 为弃权。命中率与上涨基准率使用同一分母。成对有效且非弃权的样本集合记为 $A$，零收益纳入分母，既不命中也不计上涨。

$$
N_{\rm dir}=|A|,\qquad
H=\sum_{a\in A}\mathbf 1\{s_a r_a>0\},\qquad
U=\sum_{a\in A}\mathbf 1\{r_a>0\}.
$$

$$
\mathrm{accuracy}=\frac{H}{N_{\rm dir}},\qquad
\mathrm{up\_baseline}=\frac{U}{N_{\rm dir}},\qquad
\mathrm{lift}=\mathrm{accuracy}-\mathrm{up\_baseline}.
$$

提升按命中率减上涨基准率计，以百分点表示时乘 100。时序在单品种有效样本上直接用上述公式。截面先逐时点求命中率与上涨基准率，再对分母为正的时点等权平均，提升用二者的汇总差。汇总同时报告逐时点计数求和得到的命中数、分母及上涨数。按桶报告时先限定同一桶，再用相同规则。例如有效收益有 80% 上涨且每条均看多，命中率为 80%，提升为 0 个百分点。

## 6. 重叠收益反例与区间校准

针对持有期重叠，采用 `HAC 滞后 ≥ h−1` 和 `块长 ≥ h` 作为必要的最低配置。依赖也可能来自因子、价格和缺失机制，达到下限后仍须校准区间覆盖率。

设独立单步收益 $\varepsilon_t$ 方差为 1，$Y_t=\sum_{j=0}^4\varepsilon_{t-j}$。总体自协方差由重叠项给出，$\gamma_\ell=\max(5-|\ell|,0)$。理论长期方差为

$$
\gamma_0 + 2\sum_{\ell=1}^4\gamma_\ell
= 5 + 2(4+3+2+1) = 25.
$$

即使直接使用精确总体自协方差，$L=4$ 的 Bartlett 权重仍会衰减每个非零滞后项，结果为

$$
5 + 2\sum_{\ell=1}^4(1-\ell/5)\gamma_\ell
= 5 + 2(16+9+4+1)/5 = 17.
$$

覆盖率验收应在固定种子和预设容差下检查独立、有自相关、持有期重叠的合成序列，并用已知正负效应检验方向。分析实际报告所用的滞后与块长。 [评测政策](EVAL_POLICY.md)

### 6.1 一次性合成校准

运行 `python3 -m tools.benchmarks.eval_engine_coverage_calibration`。
完整结果写入 `reports/NERE-791/coverage_calibration_20261118.json`。
主种子为 791823。每组使用 512 个无缺失的应有时点，截面有 32 个独立
资产且每配置模拟 2,000 次，单品种时序每配置模拟 500 次且每次区间用
399 个移动块 bootstrap 复制。重叠格使用五步前瞻收益和独立于收益的
AR(1) 因子。自相关格的因子和收益为相互独立的 AR(1)，系数均为 0.8。
正负格采用同期 Pearson 相关为 ±0.35 的高斯对。

各配置的标称覆盖率为 95%。模拟前固定逐配置容差为 3 倍标称蒙特卡洛
标准误，截面标准误约 0.49 个百分点（容差 ±1.46），时序约 0.97 个百分点
（容差 ±2.92）。分母为全部模拟次数，本次所有区间均可估计。
截面覆盖目标为 32 资产样本 Spearman 的总体期望（正负格约 ±0.3261），
时序覆盖目标为总体 Spearman（正负格约 ±0.3360）。

| 合成场景 | 截面最低滞后 → 覆盖率 | 截面建议滞后 → 覆盖率 | 时序最短块长 → 覆盖率 | 时序建议块长 → 覆盖率 |
| --- | ---: | ---: | ---: | ---: |
| 独立零效应 | 0 → 94.15% | 0 → 94.15% | 1 → 94.00% | 1 → 94.00% |
| AR(1) 零效应 | 0 → **65.25%** | 16 → **92.45%** | 1 → **62.00%** | 17 → **91.40%** |
| 五步重叠零效应 | 4 → **91.95%** | 16 → 93.75% | 5 → **91.60%** | 17 → 93.20% |
| 正效应 | 0 → 94.85% | 0 → 94.85% | 1 → 93.80% | 1 → 93.80% |
| 负效应 | 0 → 95.20% | 0 → 95.20% | 1 → 96.80% | 1 → 96.80% |

粗体表示低于预设容差。同一配置只计算一次。正负效应在该网格上
均达到 100% 点估计符号正确率和 100% 区间方向检出率，双侧区间均排除零。

对本次独立场景，使用下限 $L=h-1$、$b=h$。对本次五步重叠场景，
建议采用 $L=\max(h-1,16)$、$b=\max(h,17)$。相同窗口在 AR(1) 场景
仍覆盖不足。AR(1) 两个建议窗口均未达标。用于推断前，按实际样本长度、依赖和缺失机制
重新登记配置并在独立合成随机流校准。

校准登记由 §6.1 的 14 个唯一配置组成，可通过
[REGISTERED_CALIBRATIONS](../backtest_and_analysis/evaluation/calibration.py) 审计。
每条冻结的 `CalibrationEvidence` 记录场景、方法、窗口、应有时点数、标称与观测
覆盖率、容差、是否达标及报告引用。`HACInterval` 和 `BootstrapInterval`
通过 `calibration_status` 和 `calibration_evidence` 表示登记结果。
两字段默认是 `uncalibrated` 和 `None`。`reason` 独立记录当前区间能否估计。

调用方须按实际依赖显式声明 `calibration_scenario`。登记场景包括
`independent_null`、`ar1_null`、`overlap_null`、`positive_effect` 和
`negative_effect`。`resolve_interval_calibration()` 按场景、实际区间方法
（截面 Spearman IC 均值的 Bartlett HAC 或时序 Spearman 的移动块重排名）、
实际滞后或块长、应有网格长度及标称覆盖率精确匹配。截面用 `grid_size`，
时序用各品种的应有网格长度。切分后逐段匹配，网格内缺失槽位仍计入长度。
匹配时按登记记录返回 `calibrated_pass` 或 `calibrated_fail`。场景未声明、
参数未匹配或没有登记依据时返回 `uncalibrated`。状态描述已登记的合成配置。

`evaluate_interval_usability()` 用区间的实际方法、窗口、应有网格长度和标称覆盖率
重新核对依据。缺失、与登记内容不符或与登记状态矛盾的依据按 `uncalibrated` 报告。
匹配的登记状态仍保留，当前样本是否符合冻结覆盖设计在显著性判定时检查。

- `descriptive_stats_usable` 在点估计有限且无点估计失败原因时为 `True`。
  例如只有一个有效截面时仍可描述 IC 均值，即使 ICIR 和 HAC 区间不可估计。
- `significance_interpretation_usable` 要求区间可估计、登记状态为
  `calibrated_pass` 且覆盖依据适用。§6.1 的 HAC 要求每个应有槽位都有有效 IC。
  移动块 Bootstrap 要求所有应有槽位都有有效因子收益对，且请求次数与有效复制次数
  均等于登记的 399 次。未满足适用条件时仍可报告可计算的 IC 与区间。
  调用方还须核对场景及其余估计设置的适用性。择优须满足第 6.2 节的多重检验要求。

#### 时序 Bootstrap 双侧 P 值

用 `bootstrap: {p_value: true}`（Bar 配置）或 `p_value=True`（时序引擎）
启用 `ts_boot_adaptive_v3_p`。它沿用 v2 的原样本标准误、抽样、区间算法和随机流。
原假设为 $\rho = 0$，令 $t_{obs}=z/se$，其中 $se$ 为原样本
`fisher_z_standard_error`。把已用于区间的每个有效枢轴
$t^*=(z^*-z)/se^*$ 代入双侧尾部计数，得到
$p=(1+\#\{|t^*|\ge |t_{obs}|\})/(1+B_{valid})$。区间使用其自身的标称
覆盖率，P 值直接使用枢轴。

启用后默认请求 9,999 次复制且要求至少 9,000 次有效。诊断范围采用预登记的
五个场景，要求核验持仓区间、完整的单品种网格 $T\in\{512,1024\}$、
$h\in\{1,5,20\}$、95% 区间、至少 4 个有效块，以及每复制最少配对数等于 4。
每个单品种区间发布 `p_value` 和 `p_value_reason`。缺失时的原因码如下。

- `insufficient_replicates_for_pvalue` 表示请求数小于 9,999、有效数未达到区间要求，
  或有效比例低于 90%。
- `bootstrap_interval_unestimable` 表示区间因其他原因无法估计。
- `outside_pvalue_calibration_domain` 表示场景或设计超出预登记诊断范围。
- `multi_asset_pvalue_undefined` 用于多品种汇总。每个品种自己的区间可以报告诊断值。

本地归档：预登记 `reports/NERE-1269/ts_bootstrap_pvalue_preregistration.json` ·
模拟校准报告 `reports/NERE-1269/ts_bootstrap_pvalue_calibration.md`

#### NERE-1279 确认性重校准

独立的路线诊断在 T=512 的九个零效应格合并得到 5.1117% 的拒绝率，
低于预登记的 5.25% 路线 A 门槛。随后换用独立种子，主矩阵使用
T ∈ {512, 1024, 2048}、h ∈ {1, 5, 20} 和 α ∈ {0.05, 0.01, 0.0025}，
并加入方差标准化 t(4)、GARCH(1,1) 和泊松跳空。T=4096 仅做耗时试算。

对每个零效应格，用 delta 法合并 warp-speed 临界值与独立评估样本的
蒙特卡洛方差。单侧 99% 拒绝率上界必须 ≤ 1.5α。全部 162 个零效应阈值
组成一个 Holm 族，家族误判失败率目标为 1%。每个 (T, α) 的合并拒绝率
必须 ≤ 1.1α，每个正负对照格的同向功效必须 ≥ 80%。另以每样本 9,999 次
复制抽查六个合成配置的生产 P 值在 α=0.05 的拒绝率。

独立高斯零效应的 T=512、h=20、α=0.0025 格拒绝率为 0.314%，
单侧 99% 上界 0.3840% 超过 0.375% 硬上限。该格在排除失败新增特征后
仍属于存留范围，超出预登记允许移除的 T=512、α=0.05 配置。
本轮重校准未通过，v3 P 值继续在其原有单品种诊断范围内发布。
本地归档：确认性登记 `reports/NERE-1279/ts_recalibration_preregistration.json` ·
逐格报告与嵌套抽查 `reports/NERE-1279/ts_recalibration.md`

### 6.2 择优的尝试账本与多重检验

在查看结果前，将参与同一次选择的全部 `(factor_id, horizon)` 传给
`calibration.TrialLedger`。每个候选调用一次 `record(key, Trial(...))`，
保留统计量、原始检验 p 值、观测数、区间以及 `evaluate_interval_usability()`
返回的校准适用性。无原始 p 值或区间的尝试也要登记，可用 `ledger.records`
取回。候选缺记录或缺原始 p 值时，账本暂停多重校正和择优
（`select_best` 抛错，Bar 产物列出缺失候选）。时序 v3 区间附带的 P 值来自
重采样枢轴。NERE-1279 确认性重校准未达到预登记的硬上限要求。
v3 继续用于诊断，`evaluate_interval_usability()` 返回 `uncalibrated`，
Bar 候选账本的时序 P 值仍为空。

- `adjusted_p_value` 按完整登记族校正原始 p，仅供审计。
- `inference_adjusted_p_value` 用于判定。旧固定窗口域的族阈值最多为区间
  标称错误率，旧 S5 自适应 HAC 域要求 95% 区间和 0.05 的族阈值。
  NERE-1280 域的区间可取 95% 或 99%，族阈值独立取 0.05 或 0.01。
  该域按 §6.3 的预登记口径生成择优 p 值。其余候选以 1 参加校正，
  保留完整分母，并返回空的推断 p 值。
- `significant` 要求候选通过上述门禁，且推断校正 p 值不大于
  登记阈值 `alpha`。`best` 是显著候选中原始 p 最小者，原始 p 并列按
  登记顺序；无显著候选时为 null。
- Bar 产物的 `selection_complete` 表示原始 p 已齐且完成校正。
  `n_observations` 在时序中为有效配对数，在截面中为有效截面数。
  `n_effective` 继续作为旧字段保留相同整数，`n_dependence_adjusted`
  目前为 null（没有估计依赖调整后的样本量）。
- 公开账本按候选键核对持有期。手工登记调用
  `evaluate_interval_usability(..., horizon=h, cross_section_sizes=counts)`，
  截面的 `counts` 为各槽实际有效配对数。旧固定窗口域和旧 S5 核心域要求
  32 个资产，S5 泛化域允许第 6.3 节登记的恒定宽度 32–64。NERE-1280
  只登记恒定宽度 256 或 1024，且要求完整网格。时序使用区间内的有效配对及重采样数。`verified_horizon`
  只在提供并通过设计检查后生成，必须与候选键和登记场景的持有期一致。
  设计信息缺失或不匹配的候选仍进入审计校正。调用方负责确认实际依赖与声明场景匹配。

`adjust_pvalues(p_values, method=...)` 按输入顺序返回调整后 p 值。
Holm (`holm`) 在原始 p 值有效时控制族错误率（FWER），允许检验间
任意依赖。Benjamini–Hochberg (`bh`) 控制错误发现率（FDR），要求检验
独立或满足适用的正依赖条件。候选依赖结构未知时用 Holm。
账本核对登记清单，完整清单由发起选择的调用方提供。

可运行公开基准脚本
[tools/benchmarks/trial_ledger_calibration.py](../tools/benchmarks/trial_ledger_calibration.py)复现独立合成压力实验。脚本输出各场景的覆盖率和多重检验结果。

### 6.3 自适应适用域与独立校准

自适应校准按方法版本和声明的依赖场景匹配登记域。核对项包括有效样本数、
网格长度、缺失模式及比例、该域登记的标称覆盖率和持有期结构下界。早期
自适应域登记 95% 区间，NERE-1280 登记 95% 和 99% 区间。截面还须核对
Spearman IC 的有效槽宽与有效自由度。时序还须核对持仓区间、有效块数、
申请复制数及有效复制数和比例。未获认证或超出适用域的观测保持 `uncalibrated`，
在 `TrialLedger.select_best()` 中以 p=1 参与推断校正。

`evaluation.v2` 保存 `interval_usability.adaptive_observation`。登记证据绑定版本、
适用域 ID 和报告 SHA-256。截面候选的账本门控核对观测数、持有期、区间端点和
按产物区间标称覆盖率重算的 Student t 双侧审计 p 值，包括登记阈值处的边界。
Bar 入库按可信的登记前源码版本复验 NERE-1280 涉及的冻结适用性，新产物按当前
登记复验。时序 v3 的诊断 P 值见 §6.1。原 14 项固定窗口登记按 §6.1 解释。

S4 用根种子 `93620261120` 对 160 个覆盖格和 4 个相关候选族执行历史联合门禁。
该次验证有 19 个泛化覆盖格欠覆盖，`iid` 混合族 BH FWER 为 8.60%，当时两法均未登记。

S5 用独立根种子 `93620261125` 对 88 个覆盖格和 4 个相关候选族每格重复 500 次，
按方法独立判定。HAC 的 8 个核心格、54 个泛化格、2 个方向判据和 4 个相关族全部通过。
混合族 BH 用 FDR 判定，FWER 随报告提供。`iid` 混合族的 BH FDR 为 3.22%，FWER 为 8.80%。
Bootstrap 的 8 个核心格和 2 个方向判据通过，18 个泛化格中有 3 个覆盖 460/500，
低于逐格下限 461/500。S5 整体判定为失败，HAC 依独立门禁获准登记。

- S5 为 `cs_hac_adaptive_v1` 登记 8 个核心域和 1 个泛化域。核心场景为 `iid`、`strong_ar`、
  `overlap`、`gaps`、`heteroskedastic`、`combined`、`positive_effect` 和
  `negative_effect`，使用固定 `T=512, N=32` 及对应持有期与精确缺口模式。
  泛化场景 `generalized_ar_heteroskedastic` 要求 `256 ≤ T ≤ 1024`、有效槽的恒定
  宽度 `32 ≤ N ≤ 64`、至少 180 个有效槽、缺失率至多 30% 和有效自由度至少 4。
  支持 `h ∈ {1, 5, 20}`，缺口模式为 `none`、`periodic_17_3` 或 `periodic_17_5`。
  匹配且估计有效时返回 `calibrated_pass`，账本用已绑定的双侧 p 值进行显著性判定。
  `T=128`、`N=16` 或其他超域观测保持 `uncalibrated`。
NERE-1280 对完整截面网格的 `cs_hac_adaptive_v1` 双侧解析 P 值做了独立预登记。
矩阵覆盖离散的 `N ∈ {32, 64, 256, 1024}`、`T ∈ {256, 512, 1024}`、
`h ∈ {1, 5, 20}` 和 `α ∈ {0.05, 0.01, 0.0025}`，加入单位方差 t4 和
GARCH(1,1) 的零效应及正负对照。单格以单侧 99% Clopper–Pearson 上界
检验 `size ≤ 1.5α`，用双侧精确二项检验对全部 540 个零效应格做 1% Holm
校正。每个 `(N, T, α)` 合并零效应拒绝率须 `≤ 1.1α`，每格正负对照的同向
功效须至少 80%。退路沿用全族 Holm 结论，只可保留通过的预设离散水平组合。
完整矩阵未通过。按预登记退路排除 t4 和 GARCH，追加的证据只适用于显式声明
`independent_null`、`ar1_null` 或 `overlap_null` 的完整恒宽 Spearman 网格。
适用档位为 `N ∈ {256, 1024}`、`T=1024`、`h ∈ {1, 20}` 和
`α ∈ {0.01, 0.05}`。原有 S5 和固定窗口证据沿用各自的域。
对本次追加域，账本审计校正使用产物原始解析 P 值。择优用同一个统计量、HAC
方差和自由度，按预登记的 95% 区间口径重算 P 值，再取大于等于该值的最小
登记档，超过 0.05 时使用 1。95% 或 99% 区间的端点均按原口径保留。
家族阈值独立取 0.05 或 0.01。
可运行公开基准脚本 [tools/benchmarks/eval_engine_coverage_calibration.py](../tools/benchmarks/eval_engine_coverage_calibration.py) ·
[包内保留摘要](../backtest_and_analysis/evaluation/calibration_data/cs_hac_adaptive_v1_size.json)

- S5 自适应 HAC 包内覆盖摘要见
  [cs_hac_adaptive_v1_coverage.json](../backtest_and_analysis/evaluation/calibration_data/cs_hac_adaptive_v1_coverage.json)。

- `ts_boot_adaptive_v1` 登记 0 域，生产一律返回 `uncalibrated`。
  `T=128` 和 `T=256` 同时低于 S5 泛化候选域的 `T=512` 下界。
- `ts_boot_adaptive_v2` 在 NERE-962 S3 以独立根种子完成每格 500 次复验，8 个核心格、
  18 个泛化格及正负方向判据全部通过，登记 8 个核心域和 1 个泛化域。核心沿用上述
  8 个场景名，每域精确绑定 `T=512`、单品种及对应的持有期与缺口模式。
  泛化域 `generalized_ar_heteroskedastic` 仅支持单品种的 `T ∈ {512, 1024}`、
  `h ∈ {1, 5, 20}` 与缺口模式 `none`、`periodic_17_3`、`periodic_17_5`，
  至少 360 个有效配对、缺失率至多 30%。泛化域证据取偏离 95% 最大的
  `3:008` 格，覆盖 462/500。
  所有域要求验证持仓区间和精确缺口位置，至少 4 个有效块，请求复制数至少 999，
  配置的最少有效复制数至少 900，有效复制同时达到配置下限与 90%，每复制最少配对数
  精确等于 4。域内且估计有效的观测回读为 `calibrated_pass` 并绑定 `verified_horizon`。
  域外、缺少核验字段或声明与观测不符的区间保持 `uncalibrated`。v2 的登记覆盖区间，
  v3 的 P 值诊断及其单独的发布门禁见 §6.1–§6.2。

可运行公开基准脚本
[tools/benchmarks/adaptive_inference_calibration.py](../tools/benchmarks/adaptive_inference_calibration.py)复现自适应区间的合成覆盖率和多重检验实验。
[Bootstrap v2 包内覆盖摘要](../backtest_and_analysis/evaluation/calibration_data/ts_bootstrap_v2_coverage.json)

## 7. 不可估计状态

不可估计的指标返回缺失值（表格 `NaN`，JSON `null`），并带有原因码及可用计数。计数为零是有效的计数结果。只让该指标缺失，它依赖的独立统计量仍可报告。

| 情形 | 原因码 |
| --- | --- |
| 截面成对有效资产数低于阈值 | `insufficient_cross_section` |
| 时序成对有效时点数低于阈值 | `insufficient_samples` |
| 样本数达标但因子恒定 / 收益恒定 | `constant_factor` / `constant_return`，同时恒定时取前者 |
| 截面汇总没有有效 IC | `no_valid_cross_sections` |
| 仅一个有效截面的 ICIR 和样本标准差 | `insufficient_valid_cross_sections`（有效截面不足） |
| 有效 IC 序列的样本标准差为零，ICIR 缺失 | `zero_ic_std` |
| 未形成两个分位区间 | `unable_to_form_quantiles` |
| 桶无有效观测 / 多空没有同一时点配对 | `empty_bucket` / `no_paired_bucket_returns` |
| 描述性收益标准误只有一个时点 | `insufficient_time_points` |
| HAC 无法估计 | `insufficient_valid_cross_sections`（少于两个有效 IC）、`insufficient_hac_grid`（$L\ge N$）或 `degenerate_hac_variance`（估计方差非有限或非正），按此顺序 |
| 网格无法提供至少两个块起点（$b\ge N$） / 有效复制不足 | `insufficient_bootstrap_grid` / `insufficient_bootstrap_replicates` |
| 方向准确率分母为零 | `no_directional_samples` |

## 8. 兼容行为与有意差异

| 项目 | 旧路径行为 | 统一定义或处理 |
| --- | --- | --- |
| 仅一个有效 IC 时的日频 IR（有意差异） | `AlphalensRunner` 将唯一 IC 填入 IR | 统一结果的 ICIR 缺失，原因 `insufficient_valid_cross_sections`。旧格式适配器的兼容行为单独验证。 |
| 日频 NumPy 无符号入场偏移（旧缺陷修复） | 旧版位移计算可使 `np.uint8(1)` 取负回绕为 255，使用错误的入场价或错误的有效信号键。 | 日频先验证偏移，再转为 Python 整数，使位移和成交端点使用同一个偏移。旧缓存键无法区分相同数值的偏移类型，新计算使用 `forward_returns.v4` 缓存键。 |
| 截面分桶收益口径 | 旧日频主路径默认减当日截面均值，手工回退用原始均值 | 第 3 节明确两种口径，结果声明所用口径。 |

## 9. 夹具索引

- [`tests/test_eval_engine_s0_labels.py`](../tests/test_eval_engine_s0_labels.py) 覆盖两种标签的相同价格反例、缺 Bar、提前收市、夏令时切换和期货换月。
- [`tests/test_eval_engine_s4_unified_labels.py`](../tests/test_eval_engine_s4_unified_labels.py) 通过统一入口验证相同价格反例的成交时间、价格及单位，并将日频结果逐值比对到重构前的冻结输出。
- [`tests/test_alphalens_runner_correctness.py`](../tests/test_alphalens_runner_correctness.py) 验证日频两资产的 30 个有效截面均保留 IC。

S0-B 的数值反例见 [`tests/test_eval_engine_s0_counterexamples.py`](../tests/test_eval_engine_s0_counterexamples.py)。

| 反例 | 用例 | 固定预期 |
| --- | --- | --- |
| 缺失与排名 | `test_spearman_ranks_only_pairwise_valid_observations` | 成对有效后排名得到 0.5，先给全列因子排名得到 0.3273268354。 |
| 持有期独立性 | `test_missing_five_day_labels_do_not_change_one_day_statistics` | 删除 5 日标签后，1 日 IC、样本数、分桶收益和不可估计状态逐时点一致。旧路径会在清洗时去掉常数因子截面，夹具同时保留该截面的成对样本数和原因码供统一结果验收。 |
| 原始及截面超额分桶收益 | `test_legacy_quantile_return_response_to_common_shift` | 加同一常数后，旧 Alphalens 主路径的超额桶收益不变，强制回退的原始桶收益平移同一常数。 |
| 时间权重及价差配对 | `test_quantile_time_weights_and_paired_spread_standard_error` | 时间等权得到 3%，混合 12 条观测得到 5%。同一时点配对后的价差均值为 4%，描述性标准误为 2%。 |
| 单一有效截面的 ICIR | `test_single_valid_daily_ic_legacy_ir_and_frozen_icir` | 旧 IR 为唯一 IC（+1），统一结果期望缺失并带 `insufficient_valid_cross_sections`，对应第 8 节的有意差异。 |
| 方向准确率 | `test_directional_accuracy_uses_same_denominator_for_hits_and_up_baseline` | 全看多的命中率和上涨基准率均为 80%，提升为 0。零信号弃权，零收益进入分母且不命中。 |
| HAC | `test_bartlett_lag_four_understates_five_step_long_run_variance`、`test_hac_preserves_missing_slots_in_lag_one_products` | 五项重叠收益的理论长期方差为 25，四阶 Bartlett 值为 17。缺失反例的一阶 Bartlett 均值方差在保留网格时为 0.03125，压紧后为 0.015625。 |

| 阶段 | 回归入口与夹具 |
| --- | --- |
| S1 | [Alphalens 参照测试](../tests/test_eval_engine_s1_alphalens_ref.py)、[日频主路径与回退快照](../tests/test_eval_engine_s1_runner_golden.py)、[合成日频快照](../tests/fixtures/eval_engine/baseline/synthetic.json)、[性能基准脚本](../tools/benchmarks/eval_engine_baseline.py)。S1-B 私有 Harness 合成回放的结果见第 10.2 节。 |
| S2 | [面板与内核逐值对照](../tests/test_eval_engine_s2_kernels.py)、[内核性能门禁脚本](../tools/benchmarks/eval_engine_kernels.py)。 |
| S3 | [截面统计](../tests/test_eval_engine_s3_cross_section.py)、[时序统计](../tests/test_eval_engine_s3_time_series.py)、[合成校准及尝试账本](../tests/test_eval_engine_s3_calibration.py)。 |
| S4 | [日频标签端点冻结值](../tests/fixtures/eval_engine/baseline/s4_forward_returns.json)。 |
| S5 | [无前视篡改测试](../tests/test_eval_engine_s5_no_lookahead.py)。 |
| S6 | [统一入口与旧适配器](../tests/test_eval_engine_s6a_adapters.py)、[Harness 和聚宽构建回归](../tests/test_eval_engine_s6_zero_regression.py)。 |
| S7 | [时序诊断与合成 NQ 验收](../tests/test_eval_engine_s7_ts_diagnostics_nq_acceptance.py)、[冻结夹具](../tests/fixtures/eval_engine/ts_diagnostics_contract/README.md)。 |

## 10. 性能基线

NERE-803 在 S1-B 验收版本上测量旧评测入口。三组输入均使用种子 791 生成，每隔 113 个因子值设为缺失。日频使用合成交易日和 `ForwardReturnsCalculator` 生成 1 日收益，分钟评测使用 `BarReturnLabeler` 生成同 Session 的 Bar 收益。单品种 NQ 使用 2020–2025 年 `CME Globex Equity` 的 ETH 1 分钟网格，扣除日历中的休市和中断，并评测 1、5、15 和 60 根 Bar 的持有期。

下面是 S1-C 在 S1-B 验收版本上生成旧入口基线时使用的命令。

```sh
python3 -m tools.benchmarks.eval_engine_baseline
```

在当前版本运行这条命令会调用当前适配器。所得耗时仅代表当前版本，历史门禁仍需使用
S1-C 冻结产物。

每个方法在新的进程中串行运行三次。墙钟从评测调用开始，包含旧入口的清洗、IC、分桶及其默认诊断，输入生成与标签计算在计时之前完成。SciPy 参照对每个持有期先取成对有效样本，再在截面路径用 `groupby(date).apply(spearmanr)`、在时序路径沿整个 NQ 序列计算 IC。脚本核对每期有效配对数、能估计 IC 的截面或品种数和 IC 均值，旧入口与参照的 IC 均值绝对误差上限为 `1e-12`。日频先核对每个有限因子都有唯一的有限标签，再用 `AlphalensRunner` 的 `max_loss=0` 校验清洗后完整保留这些配对。

峰值内存使用子进程 `resource.getrusage(RUSAGE_SELF).ru_maxrss`，统一换算为字节，覆盖导入、造数和评测。JSON 另记评测开始前的峰值。遇到超时，JSON 保留已完成的尝试，并用 `timeout_stage` 区分准备和评测。准备超时的尝试将评测耗时和两项 RSS 留空。如果另一方法已完成准备，该场景仍有经测得的输入规模和有效配对数。评测超时保留已测得的评测耗时与 RSS，IC 和该方法的耗时中位数留空。若评测进程需要强制终止，评测前已实测的 RSS 可作为下界。

2026-11-18 在 Apple M2 上完成全部 18 次运行。环境为 Python 3.12.7、NumPy 1.26.4、pandas 2.3.3。表中时间取三次中位数，RSS 取每种方法的三次最大值。

| 场景 | 时点 × 资产 × 持有期数 | 成对有效样本数 | 旧入口 / SciPy（秒） | 旧入口 / SciPy 峰值 RSS（MiB） |
| --- | ---: | --- | ---: | ---: |
| 日频截面 | 1,000 × 2,000 × 1 | 1,982,300 | 21.376 / 0.992 | 807.2 / 683.2 |
| 分钟截面 | 60,000 × 50 × 1 | 2,971,272 | 105.673 / 32.916 | 2,068.8 / 1,843.7 |
| NQ 1min 时序 | 2,127,150 × 1 × 4 | 2,106,787 / 2,100,635 / 2,085,269 / 2,016,090（1 / 5 / 15 / 60 根） | 36.332 / 2.897 | 3,740.3 / 3,785.6 |

两组截面的 IC 均值逐次一致，NQ 的最大绝对差为 `5.55e-17`。NQ 旧入口的内存峰值在造数阶段已达到。逐次耗时、RSS、IC 和日历哈希均随 NERE-803 交付。冻结基线
`perf_baseline_20261118.json` 存于该私有工单附件。公开检出可用随仓库分发的工具和合成夹具运行第 10.1 节的冒烟。

### 10.1 S2-B NumPy 内核性能门禁

在公开仓库根目录可运行合成冒烟。它将三组小样本的内核结果与实时 SciPy 计算
对照，无需冻结基线，并跳过全尺寸性能门槛。

```sh
python3 -m tools.benchmarks.eval_engine_kernels --smoke
```

历史 S2-B 全尺寸验收命令如下。运行前须将 NERE-803 附件中 S1-C 冻结的 JSON
放到 `--baseline` 指定的位置。

```sh
python3 -m tools.benchmarks.eval_engine_kernels --baseline reports/NERE-791/perf_baseline_20261118.json
```

正式运行使用同一批合成输入生成器，逐场景核对规模、标签元数据和预计算配对数。
每次运行都用内核返回的计数逐持有期比对 S1-C SciPy 参照，要求配对数和可估计
IC 组数相等、IC 均值绝对误差不超过 `1e-12`。墙钟与 S1-C 一样从已有标签开始。
S2-B 计入将标签整理为数组以及全部持有期的 Spearman。两个截面场景还计入五分位
分配和桶内求和及计数。合成 Bar 输入是稠密网格，可依照造数顺序重排为数组。
S1-C 的旧入口另包含清洗、完整分桶统计和默认诊断。NQ 门禁计量单品种时序
Spearman。训练期分位拟合见第 14 节。

墙钟和进程峰值 RSS 采用 S1-C 的三次独立进程串行口径。另起进程用
`tracemalloc` 记录五分位分配期间新增的峰值分配量，已有面板不计入该列。
NQ 的内存剖析在时间轴上做全样本描述性分位分配。三组合成因子值均为一般
数量级的有限值或缺失值，分位计算走正常有序边界路径，极值导致的无序边界
分支未在此基准触发。

2026-11-18 的最终复测与 S1-C 机器和版本信息一致。时间取三次中位数，进程
RSS 取三次最大值，分位分配采用单独的内存采样。

| 场景 | 成对有效样本数（按持有期） | S1-C 旧入口 / SciPy（秒） | S2-B（秒） | 门槛（秒） | 旧入口 / S2-B 峰值 RSS（MiB） | S2-B 分位分配峰值（MiB） |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 日频 1,000 × 2,000 | 1,982,300 | 21.376 / 0.992 | 0.524 | 21.376 | 807.2 / 617.8 | 113.2 |
| 分钟截面 60,000 × 50 | 2,971,272 | 105.673 / 32.916 | 0.557 | 1.000 | 2,068.8 / 2,498.5 | 201.8 |
| NQ 1min 2,127,150 × 1 × 4 | 2,106,787 / 2,100,635 / 2,085,269 / 2,016,090 | 36.332 / 2.897 | 2.137 | 3.000 | 3,740.3 / 3,748.0 | 119.7 |

最终复测三项均通过，成对样本数和可估计 IC 组数逐次逐持有期一致，最大 IC
均值绝对误差为 `1.055e-15`。最终三组的进程峰值 RSS 均在准备阶段达到。
首轮 NQ 中位数为 3.353 秒，后两轮为 2.410 和 2.137 秒。
额外一次 NQ CPU 剖析中排名耗时 1.950 秒，占评测阶段的约 94%，其中
`argsort` 占 1.446 秒。逐次数据和剖析随 NERE-815 交付。
门禁产物名为 `perf_kernels_20261118.json`。

### 10.2 S6-B 端到端门禁

历史 S6-B 全尺寸验收命令如下。运行前须将第 10 节所述 S1-C 冻结基线
放到 `--baseline` 指定的位置。公开检出可运行第 10.1 节的合成内核冒烟。

```sh
python3 -m tools.benchmarks.eval_engine_e2e --baseline reports/NERE-791/perf_baseline_20261118.json
```

从 S1-C 的预计算标签开始计时，运行切换后的日频、Bar 截面与 Bar 时序三个适配器。
计入输入校验、面板整理、统计和旧接口产物格式化。每个场景在独立进程中测三次，
记录单次耗时、峰值 RSS、输入规模、有效配对数、可估计 IC 组数和 IC 均值。
配对数和可估计组数须与 S1-C 逐持有期相等，IC 均值绝对误差须在 `1e-12` 内。
日频同时由 `max_loss=0` 校验评测入口保留了输入配对。

2026-11-19 在 Apple M2 上完成最终九次运行，墙钟耗时 138.36 秒，退出码 0。
表中为评测调用耗时（秒）。

| 场景 | 三次耗时 | 中位数 | 上限 |
| --- | ---: | ---: | ---: |
| 日频截面 | 3.791172、3.634596、3.511918 | 3.634596 | 21.375780 |
| 分钟截面 | 0.593254、0.555982、0.590985 | 0.590985 | 1.000000 |
| NQ 1min 时序 | 1.756502、1.778434、1.837921 | 1.778434 | 3.000000 |

三项速度门禁均通过。九次运行的逐持有期配对数和可估计 IC 组数与 S1-C
逐项一致，IC 均值最大绝对差为 `8.604e-16`。Bar 的完整标签表、排除统计和
其他 v1 字段均在计时内生成。最终产物 `perf_s6_e2e_20261119_r5.json`
随 NERE-827 交付。

S1-A 合成日频主路径与回退路径均通过 `1e-12` 比对。普通 float32 保留按日
单精度去均值的舍入，pandas 可空 `Float32` 沿用原始收益回退和空标准误。
S1-B 合成 Harness 复跑的三个窗口入选因子与顺序完全一致，九个权重的最大绝对
误差为 0。固定输入下，聚宽构建生成的 126,151 字节单文件通过 S6-A SHA-256
比对。S0–S6 和受影响调用方的 30 文件回归为 802 项通过、2 项跳过，已开启 S1-B
合成复跑。跳过的两项分别为真实日频主路径和强制回退，均需外置库挂载后验证。

## 11. 内核接口

`backtest_and_analysis.evaluation.panel.EvaluationPanel.from_long` 接收按
`(datetime, asset)` 排列的唯一二级索引。因子列默认叫 `factor`，可用
`factor_column` 指定，其余每列对应一个持有期。输出为日期 × 资产的
`float64` 因子和收益数组。数值列须能无损转换为 `float64`，无法无损
转换时按列报错。`returns[h]` 与 `valid[h]` 按持有期访问。
`valid[h]` 仅在因子和该期收益均有限的位置为真。缺行在面板中填
`NaN`。省略网格时，日期升序、资产按首次出现顺序排列。调用方可传入
完整的 `datetimes` 与 `assets` 并指定轴顺序，保留整行缺失的应有槽位。
`to_long()` 按原行列顺序还原已观测键和值，数值列为 `float64`。

`backtest_and_analysis.evaluation.kernels` 的 `axis=1` 沿资产做截面计算，
`axis=0` 沿时间做时序计算。可选的 `valid` 是同形布尔数组。

| 接口 | 输入及返回 |
| --- | --- |
| `average_rank(factor, returns, *, axis, valid)` | 使用两数组的成对有限样本，返回两个与输入同形的平均秩数组。未参与的单元为 `NaN`。 |
| `pearson(factor, returns, *, axis, valid, min_samples)`、`spearman(...)` | 返回沿计算轴规约的一维 `values`、`counts`、`reasons`。默认 `min_samples=2`，由评测入口按第 2 节传入适用门槛。可估计时原因是 `None`，其余值为 `NaN`。样本不足优先于恒定，因子和收益同时恒定时返回 `constant_factor`。 |
| `quantile_buckets(factor, n_quantiles, *, axis, valid)` | 用有限因子和可选的信号可用掩码切桶。返回同形的 `labels`、将计算轴替换为 `n_quantiles + 1` 的 `edges`，以及一维实际桶数 `n_bins`。桶号从 1 开始，0 表示未分桶。边界按 `pd.qcut(labels=False, retbins=True, duplicates='drop')` 的右闭、含最小端点规则编号，未使用的边界槽位填 `NaN`。直接传入内核时，插值先按因子原始类型计算相邻差，溢出时沿用参照的边界及桶号。第 3 节的至少两桶门槛由调用方用 `n_bins` 判断。 |
| `bucket_sums(values, labels, n_buckets, *, axis, valid)` | 用 `np.bincount` 汇总有限、有效且有桶号的值。返回 `sums` 和 `counts`，计算轴替换为长度 `n_buckets` 的桶轴，顺序为 1 至 `n_buckets`。空桶的和与计数均为 0。 |

截面分桶时把信号可用掩码传给 `quantile_buckets`，再用 `valid[h]`
与同一组桶号调用 `bucket_sums`。

## 12. 截面统计接口

`backtest_and_analysis.evaluation.cross_section.evaluate_cross_section` 接收
`EvaluationPanel`。调用方按时间升序传入完整应有信号网格，缺行保留为
空槽位。可选的 `signal_valid` 是与因子同形的布尔资产池及信号可用掩码。
分桶、秩自相关与换手率使用信号时刻的有效因子。每个持有期的 IC、
分桶收益与方向准确率进一步使用该持有期的 `panel.valid[h]`。

```python
from backtest_and_analysis.evaluation.cross_section import evaluate_cross_section

result = evaluate_cross_section(
    panel,
    frequency="daily",
    hac_lag=4,
    n_quantiles=5,
    return_mode="cross_section_excess",
)
```

默认 `ic_method="spearman"`，也可使用 `"pearson"`。日频默认
`min_samples=2`，指定 `frequency="bar"` 时默认 3。两者均可覆盖。
`n_quantiles` 默认为 5，要求至少为 2。`return_mode` 默认为 `"raw"`，
也可选择 `"cross_section_excess"`。`low_quantile` 与
`high_quantile` 默认为第 1 桶和请求的最高桶，可显式指定。
`direction` 默认为 `+1`，也可取 `-1`，用于方向准确率。
直接调用 `evaluate_cross_section` 时，`hac_lag` 默认为 `"auto"`，也接受非负整数
或逐持有期窗口映射。自动滞后按当前持有期应用 `h - 1` 的下界。直接传入整数为
最终固定窗口，调用方负责满足各期下界。各期共用 `alpha`，默认 0.05。
[统一入口](#15-统一评测入口)收到整数时为每期取 `max(hac_lag, h - 1)`。

| 结果字段 | 形状与含义 |
| --- | --- |
| `quantiles`、`quantile_reasons` | 分桶边界、时间 × 资产的桶号、逐时点实际桶数和状态。实际桶数不足 2 时，该行桶号为 0，状态为 `unable_to_form_quantiles`。 |
| `rank_autocorrelation` | 逐时点结果。先在每个信号截面内取平均秩，再对相隔 `diagnostic_period` 个网格槽位的共同资产算 Pearson 相关。 |
| `turnover` | 时间 × 请求桶数，列按桶号升序。分子是当前桶中新进入的资产数，分母是当前桶资产数。`diagnostic_period` 默认为 1。任一端没有该桶的成员时为 `NaN`。 |
| `horizons[h].ic`、`ic_summary` | 随网格对齐的 IC、有效配对数和原因码，及时间等权汇总。 |
| `horizons[h].hac` | IC 均值方差、置信区间端点、有效 IC 和网格槽位数。 `pair_counts` 长度为 `L`，位置 `i` 对应滞后 `i + 1`，超出网格的滞后计为 0。 |
| `horizons[h].bucket_returns`、`bucket_counts`、`bucket_summary` | 时间 × 请求桶数的当期桶均值和样本数，以及以桶号为键的时间等权均值、描述性标准误、有效时点数及原因。 |
| `horizons[h].spread`、`spread_summary` | 同时点配对的高桶减低桶价差序列，以及对应时间等权汇总。 |
| `horizons[h].direction`、`direction_by_bucket` | 总体及以桶号为键的命中率、上涨基准率、提升、汇总计数和原因。 |

已按时间网格排好的 IC 向量也可直接传给 `bartlett_hac(ic, lag=L, alpha=0.05)`，
其中非有限项占据原网格槽位。结果用 `NaN` 表示缺失数值，原因优先级见第 7 节。

## 13. 时序统计接口

`backtest_and_analysis.evaluation.time_series.evaluate_time_series` 每次接收
一个频率、一个切分的 `EvaluationPanel`，持有期键为正整数网格步数。
`panel.datetimes` 按时间升序覆盖应有信号网格，缺数据的应有槽位仍占位置。
品种的预期网格不同时，可在面板中传入各品种网格的并集，并用同形布尔掩码
`expected_grid` 标出各品种自身的应有槽位。可选的同形布尔掩码
`signal_valid` 标出信号可用性。IC、分布与方向统计还应用各持有期的
`panel.valid[h]`。切分与标签清除由调用方在构建面板和掩码时完成。

`holding_intervals[h]` 为同形的 `(entry_timestamp, exit_timestamp)`
两个 NumPy `datetime64` 数组，时区感知的输入应先统一转换为 UTC。
有效配对必须有真实、正长度的持仓区间。相位组按实际入场时间检查
`[entry_timestamp, exit_timestamp)`，首尾相接合法，重叠抛出
`AssertionError` 并给出品种、持有期和偏移。

```python
from backtest_and_analysis.evaluation.time_series import evaluate_time_series

result = evaluate_time_series(
    panel,
    holding_intervals,
    block_length=12,
    seed=791,
    n_boot=999,
    min_valid_replicates=900,
)
```

`block_length` 默认为 `"auto"`，按各持有期的 `h` 限制块长下界。显式整数块长
必须满足所有持有期的下界。每个品种用记录的种子在自身网格上重采样。
默认 `alpha=0.05`、`min_samples=3`、`ic_method="spearman"`、`direction=+1`。
`ic_method="pearson"` 时，点估计和相位 IC 使用 Pearson，bootstrap 仍对应
另行报告的 `spearman_ic`。

单切分统计接口可直接在传入的面板上对各品种的可用因子单独切
`n_quantiles=5` 个分位桶，
结果记录实际桶数和边界。未形成两个分位区间时，桶号为 0，返回
`unable_to_form_quantiles`。也可传入同形整数 `bucket_labels`，
其中 0 表示未分桶，1 至 `n_quantiles` 是已声明的桶号，保留空桶。
外部分桶的边界槽位为 `NaN`。缺少收益标签的信号保留桶号。

| 结果字段 | 含义 |
| --- | --- |
| `horizons[h].instruments[asset].ic`、`spearman_ic` | 当前品种全部成对有效时点的 IC、配对数和原因码。共用内核的三个数组各有一个元素。 |
| `bootstrap` | Spearman 区间端点、网格大小、块长、种子、复制次数、有效复制数、最少有效复制数、`alpha`、置信水平 `1 - alpha` 和原因码。 |
| `returns`、`returns_by_bucket` | 总体及各桶的样本数、等权均值、样本标准误和线性插值的 `p5`、`p25`、`p50`、`p75`、`p95`。尾均值包括等于阈值的观测，对应 `mean_le_p5`、`mean_le_p25`、`mean_ge_p75`、`mean_ge_p95`。总体无观测时为 `insufficient_samples`，空桶为 `empty_bucket`，只有一条观测时仅标准误缺失。 |
| `direction`、`direction_by_bucket` | 按第 5 节规则计算的总体及各桶方向准确率、上涨基准率、提升、命中数、分母与上涨数。 |
| `non_overlapping_ic` | 以 `0` 至 `h - 1` 为键的相位 IC、配对数和原因码，包含空相位。 |
| `horizons[h].ic_summary`、`non_overlapping_ic_summary` | 总体或逐相位的品种总数、有效 IC 数、品种等权均值及中位数。没有有效 IC 时为 `no_valid_instruments`。 |

## 14. 训练与样本外评测接口

`backtest_and_analysis.evaluation.split.evaluate_splits` 接收面板、
`SplitConfig` 或同字段字典，以及沿面板时间轴排列的完整、升序、有时区的
`signal_available_at`。按频率选用真实信号可用时间。Bar 面板键若为开盘
时间，须另外传入可用时刻。多市场可用时间不同时，调用方先按可用时间对齐
面板。资产轴与应有时间网格由调用方预先确定。

- `exit_timestamps[h]` 使用与面板同形的 UTC `datetime64` 数组。日频由
  `DailyReturnLabels.scheduled_exit_dates(h)` 取得每个信号交易日的
  计划退出日期，再映射到已核验的交易日收盘时间。调用方按面板轴对齐。
  Bar 通过 `label_forward_returns(..., include_scheduled_exit=True)`
  取得 `scheduled_exit_timestamp`，它在安排好的退出价缺失时仍有值。
  Bar 的时间与覆盖边界按标签入口的会话网格及覆盖范围处理。
- `mode="cross_section"` 使用 `per_cross_section`。`mode="time_series"`
  使用 `fitted_on_train`，还须传入同形、同持有期键的
  `entry_timestamps`。时序持有期键用正整数。统计沿用面板的收益单位，
  已有参数沿用第 12、13 节的名称。
- 输出以 `train` 和 `oos` 为键，各自提供 `statistics`、逐持有期对齐的
  `valid` 与 `split_purged_count`，以及 `quantile_scheme`、`fit_identity`。
  时序拟合身份支持字符串或整数品种标识。
- `align_released_features` 在单一品种与高周期特征流内执行第 2 节门控，
  以低周期信号可用时间为结果索引。输入含 `high_tf_bar_end`、特征自身的
  `signal_available_at` 及特征列。无任何已发布 Bar 时输出缺失。

## 15. 统一评测入口

`EvaluationEngine` 接收已对齐、已标注的 `EvaluationPanel`。每个面板对应
日频或一种 Bar 频率。统一接口按传入收益量纲计算，持有期键用网格步数。
直接构造跨频率研究面板时使用小数收益。`evaluate_bars()` 传入百分数收益，
使主契约中的标准表、切分和诊断沿用同一单位。
标签计算仍按第 1 节选择日频或 Bar 的执行约定。

```python
from backtest_and_analysis.evaluation import EvaluationEngine

run = EvaluationEngine("cross_sectional", frequency="daily").evaluate(
    panel, quantile_scheme="per_cross_section",
    return_mode="cross_section_excess", hac_lag=4,
)
daily = run.splits["full"].statistics
```

`EngineResult.splits` 在全样本时包含 `full`，切分时包含 `train` 和 `oos`。
每个切分带有统计结果、逐持有期有效掩码、清除计数、分位方案和拟合身份。
日频时序可将 `DailyReturnLabels.returns` 的 `"1D"`、`"5D"` 等列直接并入
因子面板。收益、有效掩码与成交时间映射都按正整数交易日步数匹配，接受整数与
`D/d` 后缀的等价键。同一映射内的重复别名会报错。统计结果、掩码和清除计数
统一输出整数键，日频截面沿用输入键。分割仍使用实际收盘可用时间，日频收益
沿用收盘到收盘的小数口径。连续期货的开盘到收盘标签按 Bar 网格计算百分数收益，
跨 session 时需显式提供连续合约源，并按信号所在的原合约取成交价格。
[`QuantileScheme`](../backtest_and_analysis/evaluation/quantiles.py) 支持
`per_cross_section` 和 `fitted_on_train`。
`cross_sectional` 采用 `per_cross_section`，支持 `raw` 和
`cross_section_excess`。截面结果含 HAC 区间、分桶收益、价差、秩自相关、
换手和方向准确率。时间序列 `time_series` 采用 `raw`，有训练切分时用
`fitted_on_train`，也可给全样本计算传入已冻结的 `quantile_fit`。
全样本时序未传入冻结拟合时切描述性分位，`quantile_scheme` 和
`fit_identity` 为 `None`。

训练与样本外切分的时间戳和掩码沿用第 14 节的约定。全样本时序推断使用
`entry_timestamps` 和 `exit_timestamps`，输出块 bootstrap 区间、条件收益
分布和已验证的非重叠诊断。省略两个持仓时间戳时，`intervals_verified` 为
`False`，相位 IC 按网格位置抽取，bootstrap 区间的原因码为
`holding_intervals_unavailable`。截面持有期使用整数网格步数，日频也支持
`"5D"` 这类键。统一入口的 `hac_lag` 默认 `"auto"`，逐持有期使用 `h - 1`
结构下界。显式整数固定为 `max(hac_lag, h - 1)`。需要直接指定逐期滞后时，使用
[截面统计接口](#12-截面统计接口)。时序 `block_length` 默认 `"auto"`，各持有期以
`h` 为下界。种子为 0，复制 999 次，至少 900 次有效。

日频 runner 字典沿用旧 Spearman 舍入和 pandas 样本标准差，
并保留唯一 IC 作为 IR。分位主路径由引擎计算当日截面超额收益的逐桶均值和逐桶
标准误，标准误为桶内样本标准差除以样本数的平方根，少于两条观测时为缺失。
runner 的显式回退开关返回原始逐桶均值和旧版宽表。

日频 runner 引擎路径在 `result["ic"].attrs["hac_inference"]` 按持有期输出
HAC 校准与可用性元数据。旧 `newey_west_mean_test(values, max_lag)`
二元组和由其生成的月度 `ic_t_stat`、`ic_p_value` 保留为未校准的描述性
诊断量。可计算时可查阅这些数值，显著性解释不可用。

### Bar 单趟评估和功能开关

`evaluate_bars()` 先生成标签，再由统一引擎一次处理每个评估分组。
轻量运行按频率和持有期处理观察样本，标准时序 IC 表取候选序列的第 0 相位。
时序描述性分位按有效标签分桶，索引连续，收益以百分数发布。
请求推断或诊断时，时序按频率和标的分组，截面按频率分组。每组调用一次
`EvaluationEngine.evaluate()`，从同一 `EngineResult` 组装 `descriptive` 标准表及
`splits` 网格统计。时序推断检查应有网格的所有相位及实际持仓区间。诊断渲染
读取该结果中的推断统计。

| `EvaluationEngine.evaluate()` 开关 | 作用 | Bar 用法 |
| --- | --- | --- |
| `all_non_overlapping_phases` | 时序默认检查全部 `h` 个相位，设为 `False` 时仅计算第 0 相位 | 标准时序表使用第 0 相位，网格推断使用全部相位 |
| `signal_diagnostics` | 截面默认计算秩自相关和分桶换手，设为 `False` 时跳过 | Bar 截面关闭 |
| `return_diagnostics` | 默认计算方向准确率及时序收益分布或截面分桶时间均值、价差等明细 | 轻量表关闭额外明细，网格推断开启 |

Bar Python API 和 Harness 通过 `split`、`bootstrap`、`hac_lag`、
`calibration_scenario`、`trial_selection` 等显式选项请求推断，详见
[参数表](TIME_SERIES_CONTRACT.md#统一引擎参数)。
`EvaluationResult.ic_inference` 按 `(frequency, horizon_bars, instrument_id)`
提供标准表 IC 的可用性，截面键的 `instrument_id` 为 `None`。
轻量时序缺少经验证的持仓区间，Bootstrap 区间不可用，IC 仍可作为描述性统计。
`summary.json` 将标准表可用性、网格推断样本的区间及门控、候选账本一并发布为
[`bar_evaluation.v2`](TIME_SERIES_CONTRACT.md#bar_evaluationv2-产物契约)。

## 16. S7 时序诊断与 NQ 1min 验收

时序诊断夹具以仓库内已确认的输入行、标准 Bar 输出字段和数学预期为依据。
`time-series-analysis --config <path> --diagnostics-report` 通过 Harness
发布 `time_series_diagnostics.v1`、`diagnostics_report.csv` 和 `diagnostics_report.md`，
同时发布四个标准 Bar 文件。切分报告使用 `time_series_diagnostics.v2`。
诊断 JSON 内嵌评估 `summary.json`。测试核对文件哈希、退出时间、剔除守恒、
同一时序组的条件收益、方向计数、全部网格相位和移动块重排名。
报告字段与发布时序见 [时序研究契约](TIME_SERIES_CONTRACT.md#时序深度诊断报告)。
诊断报告在每个持有期的 `interval_usability` 中附加校准状态、依据和两项可用性，
并在 Markdown 区间表后逐品种和持有期展示判定。全样本诊断 CSV 列与
`bootstrap` 字段沿用 S7 的诊断契约。

NQ 1min 的 S7 结果来自固定种子的**合成工程验收**，报告标记
`real_market_data_used=false`。训练为 2020–2025 年，OOS 为 2026 年。
以 CME Globex Equity 应有分钟网格生成 ETH 输入，RTH 与 NYSE 会话取交集。
每种视图分别用清除后的 1 Bar 训练 IC 确定方向，以训练信号拟合五分位
和 z-score 参数。各视图的 OOS 复用对应训练拟合身份。持有期为 1、5、15、
60 根，区间用 240 Bar 块做 99 次移动块重排名，用于工程验收。

| 视图 | 训练网格槽位 | OOS 网格槽位 |
| --- | ---: | ---: |
| ETH | 2,127,150 | 354,825 |
| RTH | 585,960 | 97,530 |

合成年界处于休市间隔，`same_session` 标签的实际跨界清除数为 0。
独立边界夹具覆盖退出等于首个 OOS 可用信号时刻时保留、晚一根时清除。
[合成报告生成器](../tests/fixtures/eval_engine/ts_diagnostics_contract/nq_1min.py)
输出 JSON 和 Markdown，并记录数据及拟合身份。
真实 NQ 1min 在外置盘挂载、数据覆盖核验后，按已冻结的边界和评测政策补跑。
