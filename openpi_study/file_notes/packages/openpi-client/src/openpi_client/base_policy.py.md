# `packages/openpi-client/src/openpi_client/base_policy.py` 中文阅读说明

**定位：** 客户端接口。

定义infer和reset，统一远程策略与本地包装器的使用方式。

**建议读法：** 先看接口，再看WebsocketClientPolicy与ActionChunkBroker如何实现。

**易错点：** 接口返回动作，不承担环境执行。

[注释源码](../../../../../code/packages/openpi-client/src/openpi_client/base_policy.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/packages/openpi-client/src/openpi_client/base_policy.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [BasePolicy](../../../../../code/packages/openpi-client/src/openpi_client/base_policy.py#L12) | 定义infer和reset，统一远程策略与本地包装器的使用方式。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [BasePolicy.infer](../../../../../code/packages/openpi-client/src/openpi_client/base_policy.py#L17) | 本函数位于“客户端接口”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。 |
| [BasePolicy.reset](../../../../../code/packages/openpi-client/src/openpi_client/base_policy.py#L22) | 清理上一episode/调用周期留下的状态，或调用底层环境进行重置。 |
