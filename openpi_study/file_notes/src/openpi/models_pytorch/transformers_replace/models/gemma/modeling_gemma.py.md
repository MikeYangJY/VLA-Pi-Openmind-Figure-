# `src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py` 中文阅读说明

**定位：** 第三方适配。

Hugging Face Gemma的适配实现，提供条件归一化、注意力、缓存及标准任务头。

**建议读法：** 主线读GemmaRMSNorm、GemmaAttention、GemmaDecoderLayer、GemmaModel；分类头暂时跳过。

**易错点：** 标准语言模型分类/生成接口只是兼容层；openpi动作是由自己的action_out_proj产生。

[注释源码](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [GemmaRMSNorm](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L56) | PyTorch Gemma归一化适配层，支持π0.5的条件调制。 |
| [GemmaRMSNorm.__init__](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L60) | 初始化GemmaRMSNorm的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.eps、self.dim、self.cond_dim、self.dense、self.weight。 |
| [GemmaRMSNorm._norm](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L80) | 计算特征的归一化部分，缩放/条件调制等后续步骤由调用者补充。 |
| [GemmaRMSNorm.forward](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L91) | 执行PyTorch Gemma归一化适配层，支持π0.5的条件调制。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [GemmaRMSNorm.extra_repr](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L126) | 提供打印模块时显示的配置说明，不参与数值前向。 |
| [GemmaMLP](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L134) | Hugging Face Gemma的适配实现，提供条件归一化、注意力、缓存及标准任务头。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [GemmaMLP.__init__](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L138) | 初始化GemmaMLP的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.config、self.hidden_size、self.intermediate_size、self.gate_proj、self.up_proj。 |
| [GemmaMLP.forward](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L152) | 执行Hugging Face Gemma的适配实现，提供条件归一化、注意力、缓存及标准任务头。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [GemmaRotaryEmbedding](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L158) | Hugging Face Gemma的适配实现，提供条件归一化、注意力、缓存及标准任务头。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [GemmaRotaryEmbedding.__init__](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L162) | 初始化GemmaRotaryEmbedding的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.rope_type、self.max_seq_len_cached、self.original_max_seq_len、self.config、self.rope_init_fn。 |
| [GemmaRotaryEmbedding.forward](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L185) | 执行Hugging Face Gemma的适配实现，提供条件归一化、注意力、缓存及标准任务头。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [rotate_half](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L202) | 本函数位于“第三方适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 torch.cat 追踪具体实现。 |
| [apply_rotary_pos_emb](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L213) | 将位置相关cos/sin作用到Q/K，实现RoPE位置编码。 |
| [repeat_kv](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L244) | 本函数位于“第三方适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 hidden_states[:, :, None, :, :].expand → hidden_states.reshape 追踪具体实现。 |
| [_gated_residual](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L259) | 把子层输出加回输入；存在gate时先调节子层贡献，这是AdaRMS相关残差控制。 |
| [eager_attention_forward](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L284) | 本函数位于“第三方适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 repeat_kv → torch.matmul → key_states.transpose 追踪具体实现。 |
| [GemmaAttention](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L311) | Gemma注意力层，处理投影、位置编码、KV缓存和注意力计算。 |
| [GemmaAttention.__init__](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L317) | 初始化GemmaAttention的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.config、self.layer_idx、self.head_dim、self.num_key_value_groups、self.scaling。 |
| [GemmaAttention.forward](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L344) | 执行Gemma注意力层，处理投影、位置编码、KV缓存和注意力计算。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [GemmaDecoderLayer](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L395) | Hugging Face Gemma的适配实现，提供条件归一化、注意力、缓存及标准任务头。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [GemmaDecoderLayer.__init__](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L399) | 初始化GemmaDecoderLayer的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.hidden_size、self.self_attn、self.mlp、self.input_layernorm、self.post_attention_layernorm。 |
| [GemmaDecoderLayer.forward](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L414) | 执行Hugging Face Gemma的适配实现，提供条件归一化、注意力、缓存及标准任务头。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [GemmaPreTrainedModel](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L459) | Hugging Face Gemma的适配实现，提供条件归一化、注意力、缓存及标准任务头。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [GemmaPreTrainedModel._init_weights](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L477) | 按层类型初始化参数，需与随后加载预训练权重的步骤区分。 |
| [GemmaModel](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L494) | Hugging Face Gemma的适配实现，提供条件归一化、注意力、缓存及标准任务头。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [GemmaModel.__init__](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L498) | 初始化GemmaModel的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.padding_idx、self.vocab_size、self.embed_tokens、self.layers、self.norm。 |
| [GemmaModel.get_input_embeddings](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L518) | 返回底层组件引用，供模型组合或框架兼容接口使用，不会在这里执行一次完整推理。 |
| [GemmaModel.set_input_embeddings](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L523) | 替换底层组件引用；这属于模型组装接口，不是梯度更新。 |
| [GemmaModel.forward](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L532) | 执行Hugging Face Gemma的适配实现，提供条件归一化、注意力、缓存及标准任务头。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [KwargsForCausalLM](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L645) | Hugging Face Gemma的适配实现，提供条件归一化、注意力、缓存及标准任务头。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [GemmaForCausalLM](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L650) | Hugging Face Gemma的适配实现，提供条件归一化、注意力、缓存及标准任务头。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [GemmaForCausalLM.__init__](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L658) | 初始化GemmaForCausalLM的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.model、self.vocab_size、self.lm_head。 |
| [GemmaForCausalLM.get_input_embeddings](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L669) | 返回底层组件引用，供模型组合或框架兼容接口使用，不会在这里执行一次完整推理。 |
| [GemmaForCausalLM.set_input_embeddings](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L674) | 替换底层组件引用；这属于模型组装接口，不是梯度更新。 |
| [GemmaForCausalLM.get_output_embeddings](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L679) | 返回底层组件引用，供模型组合或框架兼容接口使用，不会在这里执行一次完整推理。 |
| [GemmaForCausalLM.set_output_embeddings](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L684) | 替换底层组件引用；这属于模型组装接口，不是梯度更新。 |
| [GemmaForCausalLM.set_decoder](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L689) | 替换底层组件引用；这属于模型组装接口，不是梯度更新。 |
| [GemmaForCausalLM.get_decoder](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L694) | 返回底层组件引用，供模型组合或框架兼容接口使用，不会在这里执行一次完整推理。 |
| [GemmaForCausalLM.forward](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L703) | 执行Hugging Face Gemma的适配实现，提供条件归一化、注意力、缓存及标准任务头。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [GemmaForSequenceClassification](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L797) | Hugging Face Gemma的适配实现，提供条件归一化、注意力、缓存及标准任务头。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [GemmaForSequenceClassification.__init__](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L801) | 初始化GemmaForSequenceClassification的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.num_labels、self.model、self.score。 |
| [GemmaForSequenceClassification.get_input_embeddings](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L812) | 返回底层组件引用，供模型组合或框架兼容接口使用，不会在这里执行一次完整推理。 |
| [GemmaForSequenceClassification.set_input_embeddings](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L817) | 替换底层组件引用；这属于模型组装接口，不是梯度更新。 |
| [GemmaForSequenceClassification.forward](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L826) | 执行Hugging Face Gemma的适配实现，提供条件归一化、注意力、缓存及标准任务头。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [GemmaForTokenClassification](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L901) | Hugging Face Gemma的适配实现，提供条件归一化、注意力、缓存及标准任务头。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [GemmaForTokenClassification.__init__](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L905) | 初始化GemmaForTokenClassification的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.num_labels、self.model、self.dropout、self.score。 |
| [GemmaForTokenClassification.get_input_embeddings](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L923) | 返回底层组件引用，供模型组合或框架兼容接口使用，不会在这里执行一次完整推理。 |
| [GemmaForTokenClassification.set_input_embeddings](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L928) | 替换底层组件引用；这属于模型组装接口，不是梯度更新。 |
| [GemmaForTokenClassification.forward](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py#L937) | 执行Hugging Face Gemma的适配实现，提供条件归一化、注意力、缓存及标准任务头。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
