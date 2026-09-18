# `packages/openpi-client/src/openpi_client/runtime/environment.py` 中文阅读说明

**定位：** 环境接口。

规定环境必须支持重置、观测、执行与episode结束检查。

**建议读法：** 把接口与aloha_sim/env.py的实现逐一对应。

**易错点：** 抽象接口本身不含机器人驱动。

[注释源码](../../../../../../code/packages/openpi-client/src/openpi_client/runtime/environment.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/packages/openpi-client/src/openpi_client/runtime/environment.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [Environment](../../../../../../code/packages/openpi-client/src/openpi_client/runtime/environment.py#L11) | 规定环境必须支持重置、观测、执行与episode结束检查。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [Environment.reset](../../../../../../code/packages/openpi-client/src/openpi_client/runtime/environment.py#L21) | 清理上一episode/调用周期留下的状态，或调用底层环境进行重置。 |
| [Environment.is_episode_complete](../../../../../../code/packages/openpi-client/src/openpi_client/runtime/environment.py#L30) | 检查本环境定义的episode结束条件；结束不一定与成功同义，要看具体条件。 |
| [Environment.get_observation](../../../../../../code/packages/openpi-client/src/openpi_client/runtime/environment.py#L40) | 读取并组织当前环境的传感器/状态，形成策略下一次决策的输入。 |
| [Environment.apply_action](../../../../../../code/packages/openpi-client/src/openpi_client/runtime/environment.py#L47) | 把策略选择的动作传给底层环境；仿真推进状态，真机可能产生实际运动。 |
