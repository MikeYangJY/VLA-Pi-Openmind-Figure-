# `src/openpi/models/siglip.py` 中文阅读说明

**定位：** 视觉编码器。

把每个相机图像切成patch并编码为视觉token，为VLM提供前缀条件。

**建议读法：** 从Module与decode_variant找到尺寸，再读_Module.__call__的patch/位置编码，最后看Encoder和池化选项。

**易错点：** π0使用pool_type=none保留空间token；不要把整张图只想成一个向量。

[注释源码](../../../../code/src/openpi/models/siglip.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/siglip.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [posemb_sincos_2d](../../../../code/src/openpi/models/siglip.py#L37) | 本函数位于“视觉编码器”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 jnp.arange → jnp.einsum → y.flatten 追踪具体实现。 |
| [get_posemb](../../../../code/src/openpi/models/siglip.py#L54) | 本函数位于“视觉编码器”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.param → nn.initializers.normal → np.sqrt 追踪具体实现。 |
| [MlpBlock](../../../../code/src/openpi/models/siglip.py#L68) | 视觉Transformer中的逐token前馈网络。 |
| [MlpBlock.__call__](../../../../code/src/openpi/models/siglip.py#L80) | 执行视觉Transformer中的逐token前馈网络。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [Encoder1DBlock](../../../../code/src/openpi/models/siglip.py#L95) | 一次视觉token注意力与前馈更新。 |
| [Encoder1DBlock.__call__](../../../../code/src/openpi/models/siglip.py#L108) | 执行一次视觉token注意力与前馈更新。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [Encoder](../../../../code/src/openpi/models/siglip.py#L136) | 按顺序堆叠视觉编码层。 |
| [Encoder.__call__](../../../../code/src/openpi/models/siglip.py#L152) | 执行按顺序堆叠视觉编码层。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [MAPHead](../../../../code/src/openpi/models/siglip.py#L194) | 把每个相机图像切成patch并编码为视觉token，为VLM提供前缀条件。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [MAPHead.__call__](../../../../code/src/openpi/models/siglip.py#L206) | 执行把每个相机图像切成patch并编码为视觉token，为VLM提供前缀条件。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [_Module](../../../../code/src/openpi/models/siglip.py#L223) | 把每个相机图像切成patch并编码为视觉token，为VLM提供前缀条件。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [_Module.__call__](../../../../code/src/openpi/models/siglip.py#L247) | 执行把每个相机图像切成patch并编码为视觉token，为VLM提供前缀条件。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [Module](../../../../code/src/openpi/models/siglip.py#L336) | 本函数位于“视觉编码器”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 _Module → decode_variant 追踪具体实现。 |
| [decode_variant](../../../../code/src/openpi/models/siglip.py#L344) | 本函数位于“视觉编码器”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 variant.split 追踪具体实现。 |
