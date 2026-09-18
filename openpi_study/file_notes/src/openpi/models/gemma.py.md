# `src/openpi/models/gemma.py` 中文阅读说明

**定位：** 共享模型底层。

用不同参数分支处理VLM前缀与动作后缀，并通过注意力交换信息。

**建议读法：** Config决定宽度；Attention计算Q/K/V；Block加入归一化、残差和MLP；Module逐层运行并管理缓存。

**易错点：** 两个分支可以有不同hidden width，但注意力头维必须兼容；分支交互不等于先生成语言再执行。

[注释源码](../../../../code/src/openpi/models/gemma.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/gemma.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [Config](../../../../code/src/openpi/models/gemma.py#L52) | 用不同参数分支处理VLM前缀与动作后缀，并通过注意力交换信息。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [get_config](../../../../code/src/openpi/models/gemma.py#L69) | 按名称返回配置；名称不匹配时进入错误处理，配置对象随后控制模型或实验构建。 |
| [RMSNorm](../../../../code/src/openpi/models/gemma.py#L125) | 按均方根归一化；可通过条件产生scale/shift/gate。 |
| [RMSNorm.__call__](../../../../code/src/openpi/models/gemma.py#L131) | 按最后一维均方根归一化。无cond时用常规可学习scale；有cond时生成scale/shift/gate，实现时间条件调制。 |
| [Embedder](../../../../code/src/openpi/models/gemma.py#L154) | 用不同参数分支处理VLM前缀与动作后缀，并通过注意力交换信息。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [Embedder.setup](../../../../code/src/openpi/models/gemma.py#L162) | 建立该模块用到的参数和子层，之后前向调用重复使用它们。 |
| [Embedder.encode](../../../../code/src/openpi/models/gemma.py#L173) | 用token ID索引embedding矩阵，并按隐藏维度缩放嵌入。 |
| [Embedder.decode](../../../../code/src/openpi/models/gemma.py#L181) | 用embedding矩阵的转置把隐藏表示投影回词表分数；动作flow头不是在这里输出关节值。 |
| [Attention](../../../../code/src/openpi/models/gemma.py#L187) | 将每个token的信息按相关性加权汇集，Q/K/V分别用于查询、匹配和内容聚合。 |
| [Attention.__call__](../../../../code/src/openpi/models/gemma.py#L197) | 对各参数分支分别生成Q/K/V，统一注意力头尺寸，在token维上交互，再拆回分支并投影；可读取已有前缀缓存。 |
| [FeedForward](../../../../code/src/openpi/models/gemma.py#L287) | 逐token的非线性特征变换，不在token间直接交换信息。 |
| [FeedForward.__call__](../../../../code/src/openpi/models/gemma.py#L298) | 执行逐token的非线性特征变换，不在token间直接交换信息。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [Block](../../../../code/src/openpi/models/gemma.py#L323) | 归一化、注意力、残差与前馈组成的一层Transformer。 |
| [Block.__call__](../../../../code/src/openpi/models/gemma.py#L336) | 执行归一化、注意力、残差与前馈组成的一层Transformer。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [Module](../../../../code/src/openpi/models/gemma.py#L384) | 用不同参数分支处理VLM前缀与动作后缀，并通过注意力交换信息。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [Module.setup](../../../../code/src/openpi/models/gemma.py#L396) | 建立该模块用到的参数和子层，之后前向调用重复使用它们。 |
| [Module.embed](../../../../code/src/openpi/models/gemma.py#L435) | 本函数位于“共享模型底层”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.embedder.encode(tokens).astype → self.embedder.encode 追踪具体实现。 |
| [Module.__call__](../../../../code/src/openpi/models/gemma.py#L443) | 执行用不同参数分支处理VLM前缀与动作后缀，并通过注意力交换信息。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [Module.init](../../../../code/src/openpi/models/gemma.py#L470) | 本函数位于“共享模型底层”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.embed → jnp.zeros → self 追踪具体实现。 |
| [_apply_rope](../../../../code/src/openpi/models/gemma.py#L485) | 对Q/K向量分量施加随token位置变化的旋转，让注意力感知相对位置；不改变序列中token的物理排列。 |
| [_name](../../../../code/src/openpi/models/gemma.py#L507) | 本函数位于“共享模型底层”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。 |
| [_gated_residual](../../../../code/src/openpi/models/gemma.py#L520) | 把子层输出加回输入；存在gate时先调节子层贡献，这是AdaRMS相关残差控制。 |
