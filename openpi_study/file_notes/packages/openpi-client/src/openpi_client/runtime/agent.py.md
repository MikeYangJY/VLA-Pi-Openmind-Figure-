# `packages/openpi-client/src/openpi_client/runtime/agent.py` 中文阅读说明

**定位：** 运行接口。

定义根据观测取得动作及重置的Agent抽象。

**建议读法：** Runtime向Agent要动作；具体PolicyAgent把调用转给policy。

**易错点：** Agent不是语言agent推理模型，而是机器人runtime里的接口名称。

[注释源码](../../../../../../code/packages/openpi-client/src/openpi_client/runtime/agent.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/packages/openpi-client/src/openpi_client/runtime/agent.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [Agent](../../../../../../code/packages/openpi-client/src/openpi_client/runtime/agent.py#L11) | 定义根据观测取得动作及重置的Agent抽象。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [Agent.get_action](../../../../../../code/packages/openpi-client/src/openpi_client/runtime/agent.py#L22) | 本函数位于“运行接口”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。 |
| [Agent.reset](../../../../../../code/packages/openpi-client/src/openpi_client/runtime/agent.py#L28) | 清理上一episode/调用周期留下的状态，或调用底层环境进行重置。 |
