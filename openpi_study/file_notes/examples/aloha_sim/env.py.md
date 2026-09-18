# `examples/aloha_sim/env.py` 中文阅读说明

**定位：** 仿真环境适配。

将ALOHA仿真观测、动作和任务结束条件接到统一runtime。

**建议读法：** reset→_convert_observation→get_observation→apply_action。

**易错点：** 模拟环境的成功标志与真实机器人验收指标不同。

[注释源码](../../../code/examples/aloha_sim/env.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/aloha_sim/env.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [AlohaSimEnvironment](../../../code/examples/aloha_sim/env.py#L16) | 将ALOHA仿真观测、动作和任务结束条件接到统一runtime。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [AlohaSimEnvironment.__init__](../../../code/examples/aloha_sim/env.py#L23) | 初始化AlohaSimEnvironment的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._rng、self._gym、self._last_obs、self._done、self._episode_reward。 |
| [AlohaSimEnvironment.reset](../../../code/examples/aloha_sim/env.py#L37) | 清理上一episode/调用周期留下的状态，或调用底层环境进行重置。 |
| [AlohaSimEnvironment.is_episode_complete](../../../code/examples/aloha_sim/env.py#L46) | 检查本环境定义的episode结束条件；结束不一定与成功同义，要看具体条件。 |
| [AlohaSimEnvironment.get_observation](../../../code/examples/aloha_sim/env.py#L52) | 读取并组织当前环境的传感器/状态，形成策略下一次决策的输入。 |
| [AlohaSimEnvironment.apply_action](../../../code/examples/aloha_sim/env.py#L63) | 把策略选择的动作传给底层环境；仿真推进状态，真机可能产生实际运动。 |
| [AlohaSimEnvironment._convert_observation](../../../code/examples/aloha_sim/env.py#L73) | 本函数位于“仿真环境适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 image_tools.convert_to_uint8 → image_tools.resize_with_pad → np.transpose 追踪具体实现。 |
