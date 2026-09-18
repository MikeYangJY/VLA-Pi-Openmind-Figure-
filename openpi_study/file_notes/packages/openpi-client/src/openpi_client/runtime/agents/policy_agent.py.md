# `packages/openpi-client/src/openpi_client/runtime/agents/policy_agent.py` 中文阅读说明

**定位：** 策略适配器。

把BasePolicy包装成runtime所需Agent。

**建议读法：** get_action调用policy.infer；reset转发到策略。

**易错点：** 这层不新增训练，也不自动改变动作空间。

[注释源码](../../../../../../../code/packages/openpi-client/src/openpi_client/runtime/agents/policy_agent.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/packages/openpi-client/src/openpi_client/runtime/agents/policy_agent.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [PolicyAgent](../../../../../../../code/packages/openpi-client/src/openpi_client/runtime/agents/policy_agent.py#L14) | 把BasePolicy包装成runtime所需Agent。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [PolicyAgent.__init__](../../../../../../../code/packages/openpi-client/src/openpi_client/runtime/agents/policy_agent.py#L20) | 初始化PolicyAgent的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._policy。 |
| [PolicyAgent.get_action](../../../../../../../code/packages/openpi-client/src/openpi_client/runtime/agents/policy_agent.py#L27) | 本函数位于“策略适配器”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self._policy.infer 追踪具体实现。 |
| [PolicyAgent.reset](../../../../../../../code/packages/openpi-client/src/openpi_client/runtime/agents/policy_agent.py#L32) | 清理上一episode/调用周期留下的状态，或调用底层环境进行重置。 |
