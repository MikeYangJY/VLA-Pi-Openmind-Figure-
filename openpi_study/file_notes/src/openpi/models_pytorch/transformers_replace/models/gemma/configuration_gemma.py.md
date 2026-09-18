# `src/openpi/models_pytorch/transformers_replace/models/gemma/configuration_gemma.py` 中文阅读说明

**定位：** 第三方适配。

Gemma配置补丁，增加π0.5使用的AdaRMS等配置字段，保留上游版权。

**建议读法：** 看GemmaConfig的构造参数怎样保存到配置；将use_adarms与pi05开关联系起来。

**易错点：** 这是与特定transformers版本配套的文件，不是任意版本都可替换的通用补丁。

[注释源码](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/configuration_gemma.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models_pytorch/transformers_replace/models/gemma/configuration_gemma.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [GemmaConfig](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/configuration_gemma.py#L33) | Gemma配置补丁，增加π0.5使用的AdaRMS等配置字段，保留上游版权。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [GemmaConfig.__init__](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/configuration_gemma.py#L124) | 初始化GemmaConfig的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.vocab_size、self.max_position_embeddings、self.hidden_size、self.intermediate_size、self.num_hidden_layers。 |
