# `packages/openpi-client/src/openpi_client/runtime/runtime.py` 中文阅读说明

**定位：** 控制循环。

按频率协调环境、策略代理和订阅者，运行多个episode。

**建议读法：** run→_run_episode→_step：读观测、要动作、执行动作、广播事件并计时。

**易错点：** runtime的控制循环频率不等于每一步都完整推理大模型。

[注释源码](../../../../../../code/packages/openpi-client/src/openpi_client/runtime/runtime.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/packages/openpi-client/src/openpi_client/runtime/runtime.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [Runtime](../../../../../../code/packages/openpi-client/src/openpi_client/runtime/runtime.py#L17) | 按频率协调环境、策略代理和订阅者，运行多个episode。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [Runtime.__init__](../../../../../../code/packages/openpi-client/src/openpi_client/runtime/runtime.py#L23) | 初始化Runtime的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._environment、self._agent、self._subscribers、self._max_hz、self._num_episodes。 |
| [Runtime.run](../../../../../../code/packages/openpi-client/src/openpi_client/runtime/runtime.py#L45) | 本函数位于“控制循环”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self._run_episode → self._environment.reset 追踪具体实现。 |
| [Runtime.run_in_new_thread](../../../../../../code/packages/openpi-client/src/openpi_client/runtime/runtime.py#L56) | 本函数位于“控制循环”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 threading.Thread → thread.start 追踪具体实现。 |
| [Runtime.mark_episode_complete](../../../../../../code/packages/openpi-client/src/openpi_client/runtime/runtime.py#L64) | 本函数位于“控制循环”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。 |
| [Runtime._run_episode](../../../../../../code/packages/openpi-client/src/openpi_client/runtime/runtime.py#L71) | 组织一次任务的reset、逐步控制、频率调度与结束通知。 |
| [Runtime._step](../../../../../../code/packages/openpi-client/src/openpi_client/runtime/runtime.py#L104) | 从环境读取观测，向agent要动作，交给environment执行，再通知订阅者。 |
