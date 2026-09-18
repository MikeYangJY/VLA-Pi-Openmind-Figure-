# `src/openpi/models/lora.py` 中文阅读说明

**定位：** 可选微调。

为线性/Einsum和前馈层添加低秩可训练分支，以降低微调参数量。

**建议读法：** LoRAConfig决定rank与缩放；setup创建低秩矩阵；__call__将低秩更新叠加到原运算。

**易错点：** LoRA是参数更新方式，不是KI；还要配合freeze_filter控制哪些参数实际训练。

[注释源码](../../../../code/src/openpi/models/lora.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/lora.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [LoRAConfig](../../../../code/src/openpi/models/lora.py#L19) | 为线性/Einsum和前馈层添加低秩可训练分支，以降低微调参数量。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [LoRAConfig.scaling_value](../../../../code/src/openpi/models/lora.py#L38) | 本函数位于“可选微调”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 math.sqrt 追踪具体实现。 |
| [Einsum](../../../../code/src/openpi/models/lora.py#L43) | 为线性/Einsum和前馈层添加低秩可训练分支，以降低微调参数量。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [Einsum.setup](../../../../code/src/openpi/models/lora.py#L54) | 建立该模块用到的参数和子层，之后前向调用重复使用它们。 |
| [Einsum.__call__](../../../../code/src/openpi/models/lora.py#L70) | 执行为线性/Einsum和前馈层添加低秩可训练分支，以降低微调参数量。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [Einsum._make_lora_eqns](../../../../code/src/openpi/models/lora.py#L86) | 本函数位于“可选微调”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 ValueError → re.match → m.groups 追踪具体实现。 |
| [FeedForward](../../../../code/src/openpi/models/lora.py#L108) | 逐token的非线性特征变换，不在token间直接交换信息。 |
| [FeedForward.setup](../../../../code/src/openpi/models/lora.py#L118) | 建立该模块用到的参数和子层，之后前向调用重复使用它们。 |
| [FeedForward.__call__](../../../../code/src/openpi/models/lora.py#L150) | 执行逐token的非线性特征变换，不在token间直接交换信息。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [FeedForward._dot](../../../../code/src/openpi/models/lora.py#L174) | 本函数位于“可选微调”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 jnp.dot → w.astype → lora_weights[0].astype 追踪具体实现。 |
