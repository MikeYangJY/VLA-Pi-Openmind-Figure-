# `scripts/serve_policy.py` 中文阅读说明

**定位：** 部署入口。

读取命令行指定的模型配置和权重，创建Policy并启动WebSocket服务。

**建议读法：** Args→create_policy→create_trained_policy→WebsocketPolicyServer→serve_forever。

**易错点：** 启动服务只提供模型推理；机器人环境循环由客户端负责。

[注释源码](../../code/scripts/serve_policy.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/scripts/serve_policy.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [EnvMode](../../code/scripts/serve_policy.py#L21) | 读取命令行指定的模型配置和权重，创建Policy并启动WebSocket服务。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [Checkpoint](../../code/scripts/serve_policy.py#L32) | 读取命令行指定的模型配置和权重，创建Policy并启动WebSocket服务。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [Default](../../code/scripts/serve_policy.py#L43) | 读取命令行指定的模型配置和权重，创建Policy并启动WebSocket服务。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [Args](../../code/scripts/serve_policy.py#L49) | 读取命令行指定的模型配置和权重，创建Policy并启动WebSocket服务。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [create_default_policy](../../code/scripts/serve_policy.py#L93) | 本函数位于“部署入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 DEFAULT_CHECKPOINT.get → _policy_config.create_trained_policy → _config.get_config 追踪具体实现。 |
| [create_policy](../../code/scripts/serve_policy.py#L106) | 本函数位于“部署入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 _policy_config.create_trained_policy → _config.get_config → create_default_policy 追踪具体实现。 |
| [main](../../code/scripts/serve_policy.py#L121) | 本脚本入口：读取命令行指定的模型配置和权重，创建Policy并启动WebSocket服务。 Args→create_policy→create_trained_policy→WebsocketPolicyServer→serve_forever。 |
