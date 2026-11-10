# JQ Factor Lab agent 入口

本仓库提供因子接口、本地数据导入、算法合成与聚宽代码生成。先按任务阅读对应说明：

| 任务 | 参考 |
|---|---|
| 串联本地 CSV、DuckDB、合成与聚宽打包 | [完整教学流程](docs/LOCAL_WORKFLOW.md) |
| 配置单因子分析、选择输出目录 | [配置参考](docs/CONFIG.md) |
| 实现因子、声明数据需求 | [因子接口](docs/FACTOR_INTERFACE.md) |
| 从空白骨架开始、检查合成输入 | [最小因子模板](docs/FACTOR_TEMPLATE.md) |
| 日期、股票键、复权、缺失值与窗口 | [数据语义](docs/SEMANTICS.md) |
| 导入已有本地文件 | [数据导入](docs/DATA_IMPORT.md) |
| 调用合成算法、理解输入输出 | [算法合成](docs/SYNTHESIS.md) |
| 显式指定因子并生成聚宽文件 | [聚宽打包](docs/JOINQUANT.md) |

安装、演示与测试命令见 [README](README.md)。修改时以当前源码和相应接口说明为准；用合成数据或临时目录验证。报告实际执行的检查及尚未验证的运行环境。接口测试通过不能替代因子有效性或交易可行性评价。

示例输出放入调用者指定的新目录，避免覆盖已有数据。只加载可信 Python 模块与缓存，不提交真实行情、凭证或运行结果。聚宽构建命令只生成本地文件；云端运行与交易需要调用者另行安排。

贡献和测试要求见 [CONTRIBUTING](CONTRIBUTING.md)，漏洞报告见 [SECURITY](SECURITY.md)。项目许可证为 [Apache-2.0](LICENSE)。
