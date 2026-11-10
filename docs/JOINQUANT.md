# 聚宽运行工具链

公开版提供 JQFactorAdapter、共用因子运行模块和单文件构建器。它允许用户将自己的因子打包到聚宽运行。本地构建不联网；生成文件运行时会请求聚宽数据。

```bash
python -m scripts.build_jq_strategy \
  --factor-files examples/momentum.py \
  --requirements config/factor_requirements.json \
  --cloud-api-mapping config/cloud_api_mapping.json \
  --single-file --index 000300.XSHG \
  --out outputs/demo_jq_strategy.py
```

默认已经启用公开模式，也可显式传 `--public`。必须提供 `--factor-files`、`--requirements`。所有路径相对当前工作目录解析。`--factors` 可指定类名；省略时使用所列模块发现的所有因子类。整个模块的辅助函数与常量也会被内联，请先审阅模块内容。

不会扫描默认因子目录。`from factors...` 的递归依赖也必须显式列入 `--factor-files`，并按安装的构建器所在仓库根目录解析；外部单文件因子可直接指定绝对路径。重复类名、未知指定类、缺失文件都会拒绝构建。不支持自动打包任意第三方库；第三方依赖需要聚宽环境自行提供。

不传 `--cloud-api-mapping` 时映射为空。附带基础映射支持市值、估值和原始资金流；`get_factor_values` 默认为空列表，按需加入用户所需的聚宽字段名。不会附带个人字段选集、滚动资金流参数或原实验配置。行业支持 `sw_l1_code`、`sw_l1_name`。

历史窗口由 requirements 中各因子的 `min_days`（日频）或 `min_bars`（分钟频率）加上 `slack_ratio` 推导，并取当前频率组的最大值。需要调整窗口时修改对应 requirements。已移除原先不生效的 `--window` 参数；继续传入会以未知参数报错并退出（退出码 2），不会静默忽略。三种输出路径沿用相同的窗口计算规则。

`--freq daily/1m/5m` 指定单频率输出；`--single-file` 合并频率；两者均省略时按频率输出带后缀的文件。原构建器仍将 15/30/60 分钟声明归入 5 分钟组，此类频率不作为公开构建示例支持范围。分钟模式按完整交易日取数、裁剪，主要用于已完成交易日的计算。

生成入口默认在前一交易日的指数成分上计算第一个因子并记录日志，`--index` 可修改指数。它没有下单、调仓或自动策略发布；多因子调用可使用生成模块的 `compute_factor_batch`。调用方需要提供实际交易策略入口。

adapter 统一行情与非行情字段，执行空数据、缺字段和重复键检查，支持价格 PIT 复权。回测模式调用聚宽前复权；研究模式组合原始价格与复权系数。PIT 价格复权不保证所有财务事件时点或真实交易账本一致。

保留 API 注入接口用于离线测试。真实运行需聚宽已有环境与相应数据权限；不附带本地 JQData SDK、登录、批量数据下载、远程文件传输或自动回测提交。本地 CSV→DuckDB 导入仍是独立的离线功能。旧在线复权类不恢复。

已使用合成输入测试生成、加载、计算，以及缺输入和递归依赖边界。未连接真实聚宽执行验收。打包器适用的本地 Python 版本与项目一致；兼容改写不构成对任意云端 Python 或第三方依赖版本的保证。

## 构建前检查

打包器隔离各因子模块的顶层辅助函数与赋值（包括 lambda），并保留函数参数、局部变量和默认参数的作用域。模块级 if/try/for 等控制流中的绑定，以及类体中与模块全局重名的绑定（例如 `offset = offset`），目前会明确拒绝构建。遇到 `factor_namespace_isolation_failed` 时，把相关辅助定义移到明确的顶层，或改成不与模块全局名称重名的显式类参数；不要绕过检查后上传。

cloud API mapping 必须是 JSON 对象。`get_price`、`get_factor_values` 使用字符串列表，`get_valuation` 使用字符串到字符串的映射，`get_money_flow` 的每个条目须含字符串 `field`；配置 rolling_days 时同时给出正整数 min_periods，且不超过窗口。可以省略不需要的键，不能用 null 代替列表或对象。配置类型错误会在生成文件前报出字段路径。

## 输入选择错误

显式文件包含重复因子类名时返回 `duplicate_factor_class_in_explicit_files`；`--factors` 请求了未发现的类时返回 `requested_factor_class_missing`。两种失败都向 stderr 写入 JSON（`ok: false`、`error.code/message/details`），退出码为 1，并在写入构建结果之前停止。`details.classes` 列出冲突或缺失的类名；更名或修正选择后重新构建。
