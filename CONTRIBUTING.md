# 贡献指南

欢迎提交可复现的问题和范围明确的改动。问题说明请包含 Python/系统版本、命令、预期行为、实际结果，以及使用合成输入的最小复现。报告安全问题请遵循 [SECURITY](SECURITY.md)。

修改前先阅读 [AGENTS](AGENTS.md) 及相关接口文档。因子代码会被执行，只使用可信模块；不要提交行情、凭证、账户、预训练模型、研究结果或个人机器路径。使用临时目录和合成数据验证，不连接真实账户。

```bash
python -m pip install -e '.[dev,synthesis]'
python -m pytest -q
python -m examples.generate_data --output data/demo.csv
python -m tools.agent_harness.cli single-factor-analysis --config config/demo.json --output-dir outputs/contribution-analysis
python -m tools.agent_harness.cli factor-synthesis-demo --output-dir outputs/contribution-synthesis
```

输出目录必须尚不存在。影响标签、日历、预处理或选模的修改应覆盖错误路径与边界，并说明行为变化；不能关闭退出日隔离校验来让测试或示例通过。提交时列出实际通过的检查和未验证环境，更新相关文档及 [CHANGELOG](CHANGELOG.md)。云端运行和完整交易回测仍需独立验证。

通用源码、专用文档和教学示例通过维护者的发布流程同步；维护者会把接受的贡献纳入对应来源，防止后续同步覆盖。

## 许可证

本项目采用 [Apache-2.0](LICENSE)。主动提交以合入本项目的贡献，适用 Apache-2.0 第 5 条；第三方代码须保留其原始许可和署名。
