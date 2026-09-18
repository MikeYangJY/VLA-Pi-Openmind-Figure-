# `packages/openpi-client/src/openpi_client/action_chunk_broker.py` 中文阅读说明

**定位：** 动作执行缓存。

缓存策略一次返回的动作块，每次infer只取当前时间步。

**建议读法：** 首次请求模型→按_cur_step切片→计数→达到执行长度后清空缓存→重新请求。

**易错点：** 这不是RTC异步前缀约束；执行缓存期间不会自动用每帧新观测重新预测。

[注释源码](../../../../../code/packages/openpi-client/src/openpi_client/action_chunk_broker.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/packages/openpi-client/src/openpi_client/action_chunk_broker.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [ActionChunkBroker](../../../../../code/packages/openpi-client/src/openpi_client/action_chunk_broker.py#L17) | 在多次调用间缓存动作块，按执行步数输出单步动作。 |
| [ActionChunkBroker.__init__](../../../../../code/packages/openpi-client/src/openpi_client/action_chunk_broker.py#L28) | 初始化ActionChunkBroker的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._policy、self._action_horizon、self._cur_step、self._last_results。 |
| [ActionChunkBroker.infer](../../../../../code/packages/openpi-client/src/openpi_client/action_chunk_broker.py#L40) | 只有缓存为空时才调用内部策略；随后每次取当前动作索引。达到执行长度便清空，下一次才利用新观测预测。 |
| [ActionChunkBroker.infer.slicer](../../../../../code/packages/openpi-client/src/openpi_client/action_chunk_broker.py#L48) | 本函数位于“动作执行缓存”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 isinstance 追踪具体实现。 |
| [ActionChunkBroker.reset](../../../../../code/packages/openpi-client/src/openpi_client/action_chunk_broker.py#L67) | 重置内部策略并清除旧episode动作缓存，防止下一任务执行剩余动作。 |
