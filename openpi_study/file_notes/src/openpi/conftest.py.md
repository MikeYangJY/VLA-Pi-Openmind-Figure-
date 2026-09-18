# `src/openpi/conftest.py` 中文阅读说明

**定位：** 测试环境。

为pytest选择可用的JAX设备并设置测试相关行为。

**建议读法：** 读自动fixture和pytest_configure，理解为什么无GPU也可跑部分轻量测试。

**易错点：** 测试能启动不代表大模型可在该设备高效推理。

[注释源码](../../../code/src/openpi/conftest.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/conftest.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [set_jax_cpu_backend_if_no_gpu](../../../code/src/openpi/conftest.py#L16) | 本函数位于“测试环境”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 pynvml.nvmlInit → pynvml.nvmlShutdown 追踪具体实现。 |
| [pytest_configure](../../../code/src/openpi/conftest.py#L28) | 本函数位于“测试环境”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 set_jax_cpu_backend_if_no_gpu 追踪具体实现。 |
