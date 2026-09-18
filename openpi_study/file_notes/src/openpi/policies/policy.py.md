# `src/openpi/policies/policy.py` 中文阅读说明

**定位：** 推理主入口。

把原始观测、输入变换、模型采样和输出变换封装成统一infer接口。

**建议读法：** infer：输入复制→变换→加batch轴→构造Observation→sample_actions→去batch轴→反变换；Recorder额外保存输入输出。

**易错点：** 返回的是一段动作；Policy本身不会驱动真实电机。记录的infer_ms也不自动包含完整网络/机器人闭环延迟。

[注释源码](../../../../code/src/openpi/policies/policy.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/policy.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [Policy](../../../../code/src/openpi/policies/policy.py#L31) | 可调用的推理管线：变换、模型、逆变换。 |
| [Policy.__init__](../../../../code/src/openpi/policies/policy.py#L35) | 初始化Policy的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._model、self._input_transform、self._output_transform、self._sample_kwargs、self._metadata。 |
| [Policy.infer](../../../../code/src/openpi/policies/policy.py#L82) | 输入单条观测，做输入变换并加batch轴，调用sample_actions，再去batch轴和做输出逆变换；同时附模型调用计时。 |
| [Policy.metadata](../../../../code/src/openpi/policies/policy.py#L128) | 本函数位于“推理主入口”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。 |
| [PolicyRecorder](../../../../code/src/openpi/policies/policy.py#L133) | 不改变策略选择，额外把输入输出保存供调试。 |
| [PolicyRecorder.__init__](../../../../code/src/openpi/policies/policy.py#L139) | 初始化PolicyRecorder的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._policy、self._record_dir、self._record_step。 |
| [PolicyRecorder.infer](../../../../code/src/openpi/policies/policy.py#L152) | 照常调用内部策略，再保存这一时刻的输入输出；不改变动作选择。 |
