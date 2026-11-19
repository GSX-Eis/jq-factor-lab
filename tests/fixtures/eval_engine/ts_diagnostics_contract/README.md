# 时序诊断验收夹具

本夹具依据仓库中已确认的 `docs/TIME_SERIES_CONTRACT.md`、`tests/fixtures/time_series/`
和 `docs/EVALUATION_ENGINE.md`。Bar v1 的 CLI 契约包括摘要 JSON 和三张 CSV。
启用 `time-series-analysis --diagnostics-report` 后，同一次执行发布增强 JSON、CSV 与 Markdown。
`expected.json` 的 `diagnostics_cli` 固定该发布契约。训练/OOS 专项报告由本目录的合成验收脚本生成。

## 手算样例

- `bars.csv` 用 UTC Bar 开始时间。统一开盘价为 100，因此收益的百分数与退出价点差相等。
  信号在下一分钟可知，下一根开盘入场，所持最后一根 Bar 收盘退出。
  14:34 和 14:38 缺因子，每个持有期各记两次 `factor_missing`。
  14:38 同时超出可用尾部，按优先级仍只记缺因子。
  其余尾部剔除在 1、2、5 Bar 持有期分别是 0、1、3 条。27 个候选中 17 个有效、10 个剔除。
- 1 Bar 有效收益依信号顺序是 −4、−2、0、2、−2、6、8。两桶按因子分别收到
  前四和后三条。2 Bar 的有效配对完全单调，IC 为 1。5 Bar 的有效收益为
  4、−2、6、8，收益秩为 2、1、3、4，IC 为 0.8，仅一个 phase-zero 样本使该 IC 为 null。
- 同一 CSV 中 2 Bar 偏移 1 的有效信号为 14:31、14:33、14:35，收益为 0、4、6，
  因子为 −1、1、3，因此有效样本数为 3，Spearman IC 为 1。应有网格不压缩缺口。
- 时序诊断 CLI 区间用独立 SciPy 计算。9 个槽位每次有放回选两个长度为 5 的块并截到 9 个槽，
  全部重抽使用种子 0。1 Bar 在 999 次抽样均可估计，95% 区间下界为 −0.11655430348287174。
  5 Bar 仅 748 次可估计，低于 900 次门槛，区间端点为空。
- `direction_and_returns` 的零信号用于收益分布，但不计入方向分母。
  有效收益排序为 −4、−2、−2、0、2、4、6、8，使用线性百分位。
  尾部均值包含阈值上的并列值。空桶使用 null 和状态。
- `offsets` 有两个会话。剔除无标签行后保留其应有网格位置。
  偏移 0 的收益单调递减，偏移 1 单调递增，同一偏移内区间最多首尾相接。
- `bootstrap` 的区间由独立的 SciPy `spearmanr` 枚举固定种子的 64 次抽样得到。
  每次从 9 个应有网格槽的 6 个合法起点有放回抽取 3 个长度为 4 的块，
  截至 9 个槽，剔除无效对后重新排名。区间取有效重复的线性 10% 与 90% 分位。
- 年界用例单独覆盖退出时间等于首个 OOS 信号时保留、晚一根 Bar 时清除。
  完整 CME 年界的 2026 首个会话在节后开始，两视图的同会话标签在该边界实际清除数为零。

## 合成 NQ 报告

先进行本地只读数据探测，将路径与状态写到 `work/<issue>/data_probe.json`。
只有探测状态为 `no_real_nq_1min_source_found` 时才使用本合成脚本。
从仓库根目录执行，`--output` 指向未使用的报告路径。

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python tests/fixtures/eval_engine/ts_diagnostics_contract/nq_1min.py \
    --probe work/NERE-828/data_probe.json \
    --output reports/NERE-791/nq_1min_eval_20261119.json
```

在同一份完整 ETH 合成行情上计算 ETH 与 RTH。方向取各视图清除跨界标签后的训练期
1 Bar Spearman IC 符号，均值、标准差和分位边界由训练信号计算。
统一引擎在每种视图中把同一冻结拟合应用于 OOS 一次。
每个区间使用 99 次固定种子的 240 Bar 移动块重采样，供合成链路验收使用。
JSON 保存统计量、版本身份与本地探测证据，Markdown 提供人工复核摘要。
