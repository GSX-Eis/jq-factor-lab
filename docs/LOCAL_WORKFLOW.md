# 从本地行情到聚宽文件

运行完整教学流程：

```bash
python -m examples.local_factor_workflow --output-dir outputs/local-workflow
```

每次使用新目录。流程只生成虚构行情，不访问账号、下载数据或提交交易。需要项目基础依赖；输出 JSON 汇总写到 stdout，过程日志写到 stderr。

| 步骤 | 产物 | 下一步如何使用 |
| --- | --- | --- |
| 生成符合行情导入契约的 CSV | `csv/daily/DEMO*.csv` | 交给 `DuckDBPriceImporter` |
| 调用正式导入器 | `market.duckdb` 的 `price_data_daily` | 因子通过显式 `duckdb_path` 只读取数 |
| 加载因子文件、计算并保存 | `user_factors.py`、`factors.duckdb`、`features.csv` | `FactorReader` 读取并按 `(date, stock_code)` 对齐 |
| 调用固定权重合成 pipeline | `weights/`、`scores.csv` | 两个教学因子各占 0.5，不做模型训练 |
| 从价格生成真实定义的未来收益 | `labels.csv`、`evaluation_rows.csv` | 标签按信号日期索引，可与特征按相同键连接 |
| 打包合成因子 | `requirements.json`、`joinquant_strategy.py` | 审阅后在聚宽环境另行验证 |

`examples/local_factor_workflow.py` 给出每一步的实际调用，包含落库再读取，不是只在内存里拼接演示结果。部署类使用上一交易日截面，与本地 pipeline 已延迟一期的输出对齐比较；测试还通过本地模拟行情运行生成文件，检查这条时间边界。生成文件也做 Python 编译检查。尚未执行聚宽云端回测。

## 换成自己的数据和因子

在脚本副本中替换“生成 CSV”步骤，将自己的 `daily/` 目录交给同一个导入器，或直接指定已有 `market.duckdb`，跳过生成与导入。同步替换起止日期、股票池；不要保留教学用日期与虚构代码。

将 `FactorLoader.load_factor_class_from_file()` 的输入替换为自己的 `.py` 文件，并在 `names` 中列出类名。类遵循 [因子接口](FACTOR_INTERFACE.md)。显式配置 `requirements.json` 的字段和历史窗口，按 [聚宽打包说明](JOINQUANT.md) 传入文件；不依赖自动扫描私人目录。

本例采用固定权重，部署类 `TutorialComposite` 实现未平移的同一公式，生成策略通过 `context.previous_date` 取上一交易日截面，形成与本地 pipeline 相同的一期延迟。替换因子、预处理或权重时，需要同步修改部署类并保留逐值校验；任意训练模型不会自动变成聚宽可执行策略。

## 标签与时间边界

本例标签为 `adjusted_close[t + period] / adjusted_close[t] - 1`，复权价格为 `close * factor`，period 按统一日期面板排序后的行计算，股票缺失价格仍保留为缺失值。数据完整时，1D 和 5D 分别对应 1、5 个交易观测；使用自有数据前应明确日历对齐和缺失值规则。

本地固定权重 pipeline 已把输入信号延迟一期，因此此处 `execution_lag=0`，避免再次延迟。聚宽部署类内部不再 `shift(1)`；生成策略在执行日 t 查询 previous_date=t−1，使用同一个历史截面。标签仍索引在信号日期，不要再把标签索引移动一次。`evaluation_rows.csv` 只保留有完整因子和未来价格的日期，不把尾部缺失收益补零。

要替换算法演示中的人工响应，可用 `features.csv` 与 `labels.csv` 按相同键连接后的 `1D` 或 `5D` 列。训练样本的收益退出日必须不晚于训练截止日；训练/验证/预测日期需要隔离，预处理只能在允许的数据范围拟合。本例不训练模型，合成行情上的输出不能解释为投资业绩。
