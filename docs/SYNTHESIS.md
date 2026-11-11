# 合成算法

本模块导出可复用实现，不附带个人实验配置、研究数据或预训练模型。算法数值默认值与合成测试夹具保留；它们不代表推荐的交易参数。

完整串联本地导入、因子计算、合成、行情收益标签和聚宽打包，请先运行[本地研究流程](LOCAL_WORKFLOW.md)。下面的随机特征示例用于单独熟悉算法接口。

## 可运行示例

```bash
python -m pip install -e '.[dev,synthesis]'
python -m tools.demo_cli factor-synthesis-demo --method weighted_sum --output-dir outputs/synthesis-weighted
```

`weighted_sum` 是默认方法；省略 `--method` 也能运行。本随机特征示例只运行 `weighted_sum`。它构造12个虚构标的、100个工作日、4列随机特征；固定权重方法不训练，保留一期信号延迟，写出 scores.csv、summary.json 和本次权重描述。必须使用新的输出目录。

`synthetic_panel` 的人工响应是同一行特征公式，不能声明为下一交易日收益，也没有经核验的退出日。`pca`、`lasso`、`ridge`、`elasticnet`、`quantile_elasticnet`、`lightgbm` 参数仍可解析；此示例不含这些方法的无效调参选项。在进入 pipeline 和创建输出目录之前明确抛出 `LabelIsolationError`，整次命令非成功退出。PCA 虽不拟合响应，其 pipeline 同样要求严格窗口的退出上下文；本例不绕过该检查。

需要教学行情标签时，使用 `examples/local_factor_workflow.py`：它通过显式 `TUTORIAL_SYNTHETIC` 日历和真实标签计算路径产生前向收益。这个流程使用固定权重，不能当成七种模型都已完成严格 CV 验收。自行接入训练方法时须传递真实生产方的退出映射或带来源的完整日历与有效周期。历史输出保留，不回写，也不为本人工响应伪造退出日。

## 模块

| 目录或文件 | 能力 |
| --- | --- |
| pipeline/ | LASSO、ElasticNet、Ridge、分位数回归、PCA、LightGBM、固定权重、市值条件合成、动态混合、样本加权 |
| preselection/ | IC/IR、距离相关、时序 CV、稳定性选择、层次聚类、Barra 分组、波动率一致性、Knockoff、组合边际筛选和 MI 实现 |
| qlib_adapter/ | 线性/树模型包装、数据转换、面板分组 purge/embargo CV、预处理器、自定义目标与评价指标 |
| core/ | 因子与价格读取、复权、标签计算、截面预处理、股票池和日历过滤 |
| diagnostics/ | 共线性、VIF、条件数和权重/入选集合稳定性 |
| weighting/、portfolio_plugins.py | 通用权重及组合约束扩展接口 |
| factor_composer.py、factor_reader.py、composite_factor_base.py | 原有组合器、数据读取与因子基类衔接接口 |

Barra 分组需调用方自行提供分组配置。MI 已由包初始化注册；可直接通过 `get_preselection_strategy("mi")` 获取，与其他预筛选入口一致。

## 股票键与权重约束

Qlib 转换保留原始股票键映射，模型输出恢复调用方的代码和预测顺序。重复 `(date, stock_code)` 键，以及 `600000.SH` 与 `600000.XSHG` 这类映射到同一内部证券的别名会报错；跨日期混用别名也应在输入前统一。直接使用转换器时，需要传入 `original_index` 才能恢复原始格式。

`score`、`rank`、`softmax` 在 `top_pct` 选出的股票内分配权重：先保留每只股票的 `min_weight`，再按策略偏好分配剩余资金，达到 `max_weight` 的股票不再接收分配；剩余偏好全为零时均分。最终权重和为 1，且遵守上下界。若入选数量 N 不满足 `N * min_weight <= 1 <= N * max_weight`，返回明确错误；需要调整入选数量或约束，不会自动突破上限。全相同分数也使用该规则，不退化为单只股票满仓。

## 依赖与验证边界

基础算法依赖 scikit-learn、dcor、NumPy、Pandas、SciPy。`synthesis` 可选依赖安装 LightGBM；`qlib` 可选依赖安装原生 Qlib。普通线性模型及原生 LightGBM 回退路径无需 Qlib。MPS 求解器另需支持 MPS 的 PyTorch，未纳入本次验证。

当前同步保留原算法的行为与参数。不同实现的训练/验证划分和时间间隔口径并不完全相同：例如 PCA pipeline 使用日历天扣减 target_period，线性 pipeline 使用交易日期边界；FactorComposer 现在按完整交易日历调度，并按标签实际可观测时间筛选训练与验证样本；其他模型分支须分别检查。使用真实未来收益标签前，应按具体接口检查标签退出日、训练截止日和预处理拟合范围。公开示例只证明接口可运行，不构成所有算法无前视偏差或可直接实盘的证明。

运行 `python -m pytest -q` 会覆盖公开 tests/ 及各合成模块保留的测试。全套验证需安装 dev,synthesis；没有 LightGBM 时该方法的端到端测试跳过。原生 Qlib、GPU/MPS、真实行情、完整交易回测、部署和 Windows 未验证。

本次 macOS 验证中，12 项 BLAS 线程限制测试因底层运行库不暴露可控制线程池而跳过；1 项真实数据库测试按原声明跳过。其余测试使用合成输入，未连接私人数据库。

## 动态股票池失败与空结果

`core.pool_manager.apply_dynamic_stock_pool_filter()` 在启用动态股票池时传播数据源、日期和选股异常，不能把失败当作未过滤成功。合法空成员池或空期间映射返回空面板；仅在明确关闭动态过滤时原样返回输入。调用方应停止或明确记录失败，不能据此发布未受股票池约束的结果。

## DuckDB结果保存

`ResultWriter` 与 `FactorReader` 使用 DuckDB。`ResultWriter.write()` 默认替换该因子的分析结果，保持全量重算语义；分批增量分析必须显式传 `date_range=(start, end)`，该区间内旧分组会一起替换，其他日期保留。SQLite结果文件不再作为回退来源。

## Qlib 接口退役与重新训练兼容性

已移除未被主线调用的 `FactorBinConverter` 及 `bin_converter` 模块；内存转换继续使用 `DataConverter`。同时移除了模块级便利函数 `pearson_ic_loss`、`cross_sectional_ic_loss`、`ic_eval_metric`、`sharpe_surrogate_loss`、`sharpe_eval_metric` 和 `compute_cv_metric`。外部调用方需要迁移到对应的目标函数类和指标类接口，不能继续导入这些名称。

Sharpe 训练目标修正了梯度的 epsilon 与截断处理。使用旧配置重新训练 `sharpe_surrogate` 或 `topk_sharpe` 时，模型、预测及 Sharpe 结果可能变化；已保存的模型和历史结果不会被重写。原生 LightGBM Dataset 的日期排序会同步重排特征、标签和权重；已经排序的输入保持结果一致。Sharpe 评估回调要求原生 Dataset，不能作为 sklearn 风格回调使用。

## 数据集接口与评分兼容性更新

已移除未被主线调用的 `CVMetricEvaluator.evaluate_from_index`、`FactorPilotDataset.feature_names`、`FactorPilotDataset.get_segment_index`、`FactorPilotDataset.get_original_index_for_segment`，不保留兼容壳。`FactorSynthesisHandler`、`FactorSynthesisHandlerStandalone` 及 `data_handler` 模块也已退役；共享 processors、DataConverter、FactorPilotDataset 和模型包装器保留。仓库外直接调用或导入这些移除名称会失败。

反向证券转换遇到 None、NaN、pd.NA、空串或纯空白键时抛 ValueError。直接构造 FactorPilotDataset 后请求标签，会按选中特征的完整键取标签并保持行序；重复键或选中键缺标签时抛 ValueError，不填值或聚合。已有 NaN 标签值保留，只取特征的路径不增加标签限制，已对齐的工厂路径保持不变。

单日或没有有效日期的长短收益不足以计算样本标准差，Sharpe 评分返回不可用分数 `-inf`，不再返回 NaN；多日公式、年化以及低标准差返回0的规则不变。全候选无效时选择首个候选的既有回退保持不变。

LGB 搜索将计划候选总数传入 CV 评价器，失败候选也计入，多候选 DSR 分数与参数选择可能变化。此变更不回写历史结果。LGB训练/验证及Linear非pooled的 `net_topk_dsr` 评分现已传入证券身份，非零成本净DSR搜索的分数及参数选择可能变化；零成本和其他指标保持不变。成本率、换手公式、日期抽样及evaluate缺省契约不变；pooled配置在baseline非有限时仍可能退回非pooled。

### 近似 DSR 的解释边界

`deflated_sharpe` 及共用内核的 `net_topk_dsr` 是现有近似排序分数，有限结果在[0,1]，**不是经校准的统计置信概率**。计算值和配置保持兼容：

- 设 N 为计划候选数、T 为 n_obs、s 为未年化收益均值/样本标准差。N≤1 时参考阈值为0，否则为 `norm.ppf(1-1/N)/sqrt(T)`；N=2 的阈值仍为0，因而相对一次没有附加惩罚。
- 方差近似为 `(1 + 0.5*s² - skew*s + (kurtosis-3)*s²/4) / T`，分母为 n_obs。非有限或小于1e-8的方差置为1e-8，再以 `norm.cdf((s-阈值)/sqrt(方差))` 评分。
- 该截断可能让短样本饱和为1，不能解释为“100%可信”。收益少于2项或最终评分非有限返回 `-inf`；标准差小于1e-8返回0。T≤2时偏度取0，T≤3时非超额峰度取3。

现有实现没有估计跨候选 Sharpe 离散度或有效独立试验数，不应宣称完整实现论文中的概率校准。新的统计设计需单独评估，不能静默替换已有配置行为或重跑历史实验。

## 预筛选行为与版本标识更新

DCor 工作器、相关缓存、CV 验证及分布统计的非预期计算错误现在向调用方传播，不能继续当成缺少候选或未过滤成功；样本不足的原有 None 路径保留。Knockoff FDR 只接受有限的[0,1]值，层次聚类簇数必须是有限正整数（兼容整数值 float），未知 CV metric 抛 ValueError。Barra 阈值别名冲突仍按原优先级处理，并发出 warning。

以下获批变化可能改变重新运行的选择结果，报告中提供版本标识；历史结果和冻结哈希不会被回写。

| 变化 | 报告标识 |
| --- | --- |
| Barra补足、ICIR/DCor保底及DCorCV内外保底优先遵守相关上限，允许数量不足 | `fallback_correlation_version=cap_first_v2` |
| DCorCV每折只用训练数据预选和去相关，包括内层top3；最终全窗口选择保留 | `fold_train_only_v2` |
| DCor/DCorCV阈值后保留输入候选优先级，集合、指标和阈值不变 | `threshold_order_version=input_priority_v2` |
| 单折验证终点限在有效窗口内，不越过holdout起点；空折走无折路径，purge/gap/标签不变 | `cv_split_version=holdout_bounded_v2` |
| CV诊断每次select清空并返回本次折、得分及最优阈值 | `cv_diagnostics_v2` |
| 抽样前先排序日期再使用局部RNG；非抽样输入不变 | `sampling_version=sorted_dates_v2` |
| Gaussian/TimeSeries knockoff每次select清空旧报告 | `report_lifecycle_version=per_call_v2` |
| 拒绝非半正定的2C-S，抛ValueError，不降维、不换协方差公式 | `covariance_validation_version=psd_guard_v2` |
| 聚类距离模式独立于n_clusters门槛，其他strict小输入边界保留 | `distance_gate_version=distance_independent_v2` |
| count模式所有出口遵守max_factors，按指标优先级取；距离模式不截断、保底不扩大候选池 | `count_limit_version=all_exits_v2` |

未被主线调用的 `FactorDistributionAnalyzer.from_config_dict` 已移除，不保留兼容壳，外部直接调用将失败。严格模式逐簇错误传播、时序knockoff的e值口径、层次聚类有效样本口径及quantile_elasticnet事件预处理不在本批变更内。配置声明的因子数可能在运行时因缺失或预筛缩小；没有据此证明历史实验已受影响，也没有重跑历史实验。

## Qlib 搜索、对齐及分位数回归更新

`QlibLinearWrapper.get_coefficients_df()` 已退役，不保留兼容壳；仓库外调用该方法会失败。
现有训练返回的 `model_info` 仍包含系数及选择结果。

以下变化可能改变使用旧配置重新训练得到的参数、模型或预测；已保存的实验结果、模型和哈希不会被改写：

- RidgeCV 使用显式构造的 purged 折，保留原 purge 长度；没有可用折时抛 ValueError。
- 非 pooled 搜索过滤单折及聚合后的非有限评分，无有效候选时拒绝继续选参。
- ElasticNet rerank 按实际拟合的 l1/alpha 网格恢复轴，避免 squeeze 后的轴解释错误；默认及单轴网格的候选与选参可能变化。
- Lasso/ElasticNet rerank 和 custom 最终 refit 沿用 `lasso_tol`；默认容差保持不变，显式非默认容差可能改变模型。
- stable-support 按选中特征的完整键对齐标签，倒序或交错标签会得到正确行序；标签路径的重复键或缺键抛 ValueError，正常工厂路径不变。
- Quantile IRLS 同时检查系数及截距变化，不在截距尚未稳定时提前终止；当前目标和 solver 保留。
- Quantile CV 没有折或没有有限胜出分数时，在最终 refit 前抛 ValueError；保留部分成功折的均值和有限同分规则。

`pinball_loss` 要求一维、等长的标签和预测，否则抛 ValueError，避免广播产生错误损失。
一维等长输入和空数组的原有处理不变。Quantile 与 ElasticNet 的现有报告元数据字段未在这次更新中调整。

## 预筛统计合同及错误处理更新

层次聚类 strict 模式现在传播距离指标、逐簇代表和预筛后端的计算错误；缺少所需指标列时明确报错。
`PreSelectionCache.get_dcor()` 新增可选 `raise_on_error=False`，strict 调用显式开启，默认调用保持原行为。
Stability 的标量指标、bootstrap 与相关预筛，以及 MI 计算/相关筛选错误不再伪装成成功的空结果或全部保留。
样本不足的既有返回路径及普通模式中未授权改变的降级路径保留。

以下合同可能改变新运行的统计或选择结果，历史结果、冻结配置与哈希不回写：

| 变化 | 版本标识与边界 |
| --- | --- |
| Time-series Knockoff 的 e 值分母使用当前阈值下 `W <= -threshold` 的数量；原负值总数诊断保留 | `e_value_version=threshold_negative_v2` |
| 层次聚类使用因子/标签共同非缺失样本，至少10对；IC按日剔除不足10对及常量截面，不再伪造零IC日，无有效指标时走既有拒绝路径 | `metric_sample_version=pairwise10_valid_days_v2` |
| Stability IC/IR统一共同样本、常量截面处理，IR使用样本标准差ddof=1；不足有效日期保持原拒绝规则 | `metric_sample_version=pairwise10_ddof1_v2` |
| quantile_elasticnet 的event_driven因子默认改为winsorize_zscore，显式覆盖仍优先 | 日志 `preprocess_default_version=quantile_linear_v2` |
| MI保底候选也遵守相关上限，允许少于fallback_top_n | `fallback_correlation_version=cap_first_v2` |
| MI按传入指标优先级贪婪去相关；超过20000行时使用局部seed42抽样 | `input_priority_v2`、`local_rng42_v2` |

层次聚类距离计算不再重置全局随机状态，距离矩阵不变；预筛显式 `sampling_mode` 会透传，默认值不变，报告为 `sampling_mode_v2`。
Stability、MI、Portfolio每次select入口清空报告，包含disabled、空输入和strict失败路径；报告标识 `per_call_v2`。
Stability诊断 `selection_count` 先round再转int（`rounded_count_v2`），不改变概率和选择结果。
bootstrap次数必须为正，Portfolio换手成本必须有限且非负。

退役的内部方法不保留兼容壳：层次聚类 `_select_representative_factors`，Stability `_compute_dcor_for_all_factors`、`_filter_by_correlation`。
外部直接调用这些方法会失败。Stability真实主线一直走 `_pre_filter_by_correlation`，按指标优先级遍历；
此前发布的 `correlation_filter_version=input_priority_v2` 保留，但不代表此前Stability主线曾发生输出变化。

## 输入绑定、存储接口与合成诊断更新

因子值和池结果写入按显式列名对应，允许物理表列顺序不同。旧结果库保护检查完整的因子名集合，并按 DuckDB 的 ASCII 大小写规则识别表和列。Reader 保留调用方的输入绑定；失败的 writer 候选不会替换已安装实例。

以下无主线调用的接口已移除，不保留兼容壳；外部直接导入或调用会失败：

- `MonthlyFactorResultWriter` 及其模块、`write_monthly_factor_results`。
- `get_duckdb_results_writer`、`ResultWriter.close`、`DuckDBResultsWriter.close`。
- `FactorReader.get_factor_statistics`、`FactorReader.close_connections`。
- `CompositeFactorBase.resynthesize`。

池结果链、共享统计、`monthly_factor_output_table` 配置和旧表保留；实际数据库连接仍按原生命周期关闭。历史数据库和已落盘实验结果不会自动迁移。

因子缓存使用 `factors_v3` 和 `processed_calendar_v3`：冻结绑定身份包含实际关系，raw 缓存按因子集合共享，处理后缓存区分有序因子列表。旧缓存文件保留。价格查询转义股票名称中的单引号，日期边界保持原合同。

旧 `FactorComposer` 拒绝与结构字段冲突的因子名 `date`、`stock`、`target_return`。自动 PCA 每个训练窗口重新选择维数；显式维数保持原行为。重要性与解释方差来自每个成功合成结果的独立诊断快照，按实际训练特征顺序映射，不再被后续拟合覆盖。

`CompositeFactorBase` 的因子相关性按共同 `(date, stock)` 键对齐，然后逐对删除缺失，继续使用 Pearson；异位缺失时结果可能与旧版不同。完整数据和相同缺失掩码的结果不变。

`FactorComposer.evaluate_performance` 的 `ic` 现在是每日横截面 Spearman 的等权均值：每天至少两股有效配对且两侧非恒定。`ir` 为有效日 IC 均值除以样本标准差（`ddof=1`），至少两个有效日且标准差大于零才有值，否则缺失，不年化、不填零。新增 `ic_version=daily_cross_section_v1`；MSE/R2 保持原口径。这会改变未来的 IC/IR，历史指标和冻结哈希不回写。

## Qlib 元数据与无调用组件移除

Quantile 搜索元数据记录实际最佳 pinball 分数、最终 IRLS 次数、实际 split 数、候选网格总数和有贡献评分的候选数。部分失败折的聚合规则不变；预测不变，新运行的模型证据哈希会变化，已有证据不回写。

`GaussRankProcessor`、`LimitedFillnaProcessor`、`PITFactorCalculator`、`create_pit_calculator_from_config` 及 PIT 模块已移除。外部直接导入这些名称会失败。实际使用的 `PerFactorPreprocessor`（含 gauss_rank 模式）、`PostCSStandardizeProcessor`、`BlendWithOriginalProcessor` 及 pipeline helper 保留。
MI 由包初始化注册，可通过策略工厂获取；这些接口行为不代表已验证历史实验效果。


## 预筛选与 pipeline 合同更新

Portfolio 的普通回退先保留输入列中存在的 mandatory 因子并去重，再按原列顺序补足至原目标数量与 mandatory 数量的较大值；正常评分路径也保留无有效评分的 mandatory，并记录 `no_valid_score`。不存在的列仍跳过，`empty_on_missing_baseline` 的显式空返回优先。版本为 `mandatory_selection_version=present_mandatory_v2`。

Volatility 的样本门槛及 `n_obs` 只累计该因子在有效 IC 日期上的有限因子/标签配对数，原每日至少3对的 IC 规则不变，版本 `bucket_sample_version=valid_ic_finite_pairs_v2`。保底候选也在原候选池中执行相关上限，允许缺额并记录，不扩池补足；禁用策略仍返回全列，版本 `fallback_correlation_version=cap_first_v2`。每次调用清空报告和 diagnostics，完整索引重复时报 ValueError，不聚合或去重。

DynamicBlend 的日 RankIC 及信号相关性先取共同有限配对，至少10对且双方非常量后才计算 Spearman；保留 signed IC、有效日等权、跨日 ddof=1 和不足日回退。`sample_contract_version=finite_pairs_10_v2`。非有限 IC 均值或标准差产生明确失败诊断并沿用 base_ratio 回退，原始统计值保留；`diagnostic_version=finite_statistics_v2`。空 Top-K 输入保留空索引结构；非空规则不变。

PCA 训练和预测共享训练时已解析的预处理配置；冲突时以 `preprocessing` 为准，原有 `preprocess` 配置仍兼容，版本 `preprocessing_version=shared_config_v2`。Lasso 和 LightGBM 的分布分析都透传 `max_sample_dates` 与 `use_vectorized`，默认仍为100/True，版本 `distribution_config_version=sampling_controls_v2`。LightGBM 后处理报告的缺省值修正为实际行为 False；零 gamma 分支补充实际选择诊断，不改变信号。

PCA、Lasso、LightGBM 及相关过滤/混合路径的计算异常会传播，不再伪装为空成功、等权或静态成功；滚动调用方没有捕获时，任一窗口异常会终止整次运行。明确的样本不足、无模型、无系数等既有回退保留。非有限配置或实际使用的权重、父基线值会被拒绝。

无主线调用的 `factor_synthesis.pipeline.custom_objectives` 整个模块已删除，六个旧函数不保留兼容壳，外部直接导入会失败。实际使用的 Qlib `QlibCustomObjectives` 保留。

上述版本及已授权样本/保底变化只影响未来运行，历史实验结果和冻结哈希不回写、不自动重算。退出日接口、短历史 cutoff 方案及 LightGBM 的 `topk_ic_pct` 透传不属于这次修改。


## Composer 的交易日历与可观测标签

`FactorComposer.synthesize_factors` 现在必须显式传入关键字参数 `target_available_at` 和 `trading_calendar`。前者须与收益表具有相同的唯一行列轴，后者须非空、无缺失、唯一且有序；直接调用旧签名会报错。`CompositeFactorBase` 已按标签实际使用的未来观测构造这两个参数。

`training_window` 和 `update_frequency` 按完整交易日历计数，训练窗口不包含当前日期；监督训练只使用截至当前收盘已经可观测的标签（等于截止时刻可用）。`min_observations` 是清洗后的样本行数，`min_training_dates` 是不同训练日数。Lasso/Ridge 的验证按完整日期分组并排除尚未可观测的训练标签，缩放和异常值处理仅拟合训练折；LightGBM 使用时间尾部验证。PCA 不要求未来标签。拟合失败时沿用最近成功的模型与缩放器，没有成功模型时跳过预测；`training_history` 记录尝试、实际训练日和沿用情况。这会改变未来训练及预测，不回写历史结果。

`get_pricing_data` 对分钟数据的结束日采用次日午夜的半开上界，包含结束日当天全部 bar；其他价格入口和训练标签入口的结束日合同未在本次统一修改。已观测的非正或非有限价格/复权因子在本次获批的标签及 ratio 入口被拒绝，缺失值仍按原合同处理。预处理参数错误与真实数据访问异常不再静默回退；合法小样本、常量或分箱退化仍保留原回退规则。单日期和空输入的残差结果现在保持 Series 形状。

## LightGBM、聚宽数据读取与构建检查更新

LightGBM 搜索只接受有限的训练、验证和最终评分，摘要来自同一胜出记录；启用搜索却没有有限胜者时明确报错，不再改用默认参数训练。native 排名训练按日期稳定排序，特征、标签和权重同步排列。预测按训练特征名单重排；缺列、多列、重复列或训练名单缺失均报错。

L1 目标明确使用 native/mae；Qlib 后端显式接收 bagging_freq、min_child_samples、random_state。旧配置重新训练可能得到不同模型，已有模型、实验结果和冻结哈希不回写。

估值查询的全部批次为空时使用已有连续基本面回退，保留 strict 规则；不补部分空批，也不增加重试。滚动资金流先取窗口所需的前史，再裁回请求日期和股票；历史不足仍为 NaN。滚动窗口使用有限值独立求和，避免累计相减的数值抵消。启用扩展窗口且有合法起始日期的纯 custom 查询保留完整区间，不再截到120行。大规模分钟取数按原 API 边界分段；注入的字段合同可映射并去重供应商因子名称。

构建器拒绝不同依赖模块的顶层非导入同名绑定，即使定义文本相同也报错。无冲突、菱形依赖和同源重复导入保持原行为。

QlibLGBWrapper.load_model、JQFactorAdapter.set_get_factor_values、JQFactorAdapter.set_pit_mode 已退役，不保留兼容壳；外部直接调用会失败。本次不修改现有策略部署产物；后续新构建策略会采用上述数据读取行为，生产策略替换仍需人工审核。

## 联合输入隔离、日历缓存与接口退役

训练折使用真实标签退出日，并要求退出严格早于验证开始；PCA 的训练窗口也遵守同一隔离上下文。调用方必须提供真实生产方的退出映射，或带来源的完整交易日历和有效周期，缺失或不一致时抛 LabelIsolationError。内层隔离由实际退出决定，旧 embargo 数量参数不再替代退出日校验；target_period 对评分及外层报告的作用保留。这个异常会穿透预筛、诊断和 runner，停止本次运行而不是继续后续方法；普通诊断错误仍按原合同记录。人工同日目标的教学示例只允许固定权重方法，其余六种方法明确拒绝。

市值中性化按完整研究窗口的已核验交易日历移一日，不前填，也不取窗口外预热。prepared snapshot 使用 v4，身份包括实际标签合同和窗口内有序 sessions 摘要；缓存命中恢复六面板的索引、Series属性和真实退出上下文，源修订或窗口内日历变化时重建。旧命名空间保留。

诊断新增输出字段 stability_switch_version="window_count_v2"。切换稳定性定义：以实际权重窗口数N比较切换次数严格大于N×0.5。缺共同样本的诊断报告 insufficient；这不是通过全部诊断。历史诊断、实验结果和冻结哈希不回写、不重跑。

本批移除无调用接口及旧实现，不保留兼容壳：FactorBase.prepare_for_alphalens、独立聚合及mul/sub/add/divide/max/min和九个数学格式包装；core缺失率/市值中性化/标签模式及next-open包装、rank/rank_zscore；main对应无消费者代理；PostCSStandardizeProcessor、BlendWithOriginalProcessor及pipeline对应后处理/混合导出；DynamicStockPoolManager的get_all_trade_days_from_db包装、create_common_filter_sets、_validate_index_constituents_table、get_available_fields；已退役的lasso_mps_solver模块，以及temporary_output、ensure_dirs、get_path_info、ensure_qlib_initialized包装。外部直接导入、调用或覆盖这些接口会失效。存续日历、选池、PCA标准化、Lasso独立混合和真实Alphalens分析保留。现有云端策略不重新构建或部署；需要更新时另行人工审批。
