# `src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py` 中文阅读说明

**定位：** 第三方适配。

组合视觉塔、图文投影和语言模型，处理多模态输入及KV缓存。

**建议读法：** 跟踪get_image_features、PaliGemmaMultiModalProjector与PaliGemmaModel.forward；再看缓存mask。

**易错点：** 理解图像特征怎样进入语言hidden space即可，不必先学完标准文本生成API。

[注释源码](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [PaligemmaModelOutputWithPast](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py#L51) | 组合视觉塔、图文投影和语言模型，处理多模态输入及KV缓存。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [PaliGemmaCausalLMOutputWithPast](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py#L74) | 组合视觉塔、图文投影和语言模型，处理多模态输入及KV缓存。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [PaliGemmaMultiModalProjector](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py#L100) | 组合视觉塔、图文投影和语言模型，处理多模态输入及KV缓存。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [PaliGemmaMultiModalProjector.__init__](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py#L104) | 初始化PaliGemmaMultiModalProjector的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.linear。 |
| [PaliGemmaMultiModalProjector.forward](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py#L111) | 执行组合视觉塔、图文投影和语言模型，处理多模态输入及KV缓存。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [PaliGemmaPreTrainedModel](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py#L119) | 组合视觉塔、图文投影和语言模型，处理多模态输入及KV缓存。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [PaliGemmaPreTrainedModel._init_weights](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py#L136) | 按层类型初始化参数，需与随后加载预训练权重的步骤区分。 |
| [PaliGemmaModel](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py#L153) | 组合视觉塔、图文投影和语言模型，处理多模态输入及KV缓存。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [PaliGemmaModel.__init__](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py#L161) | 初始化PaliGemmaModel的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.vision_tower、self.multi_modal_projector、self.vocab_size、self.language_model、self.pad_token_id。 |
| [PaliGemmaModel.get_input_embeddings](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py#L176) | 返回底层组件引用，供模型组合或框架兼容接口使用，不会在这里执行一次完整推理。 |
| [PaliGemmaModel.set_input_embeddings](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py#L182) | 替换底层组件引用；这属于模型组装接口，不是梯度更新。 |
| [PaliGemmaModel.set_decoder](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py#L187) | 替换底层组件引用；这属于模型组装接口，不是梯度更新。 |
| [PaliGemmaModel.get_decoder](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py#L192) | 返回底层组件引用，供模型组合或框架兼容接口使用，不会在这里执行一次完整推理。 |
| [PaliGemmaModel._update_causal_mask](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py#L199) | 本函数位于“第三方适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 isinstance → torch.finfo → past_key_values.get_max_cache_shape 追踪具体实现。 |
| [PaliGemmaModel.get_image_features](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py#L271) | 本函数位于“第三方适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.vision_tower → self.multi_modal_projector 追踪具体实现。 |
| [PaliGemmaModel.forward](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py#L292) | 执行组合视觉塔、图文投影和语言模型，处理多模态输入及KV缓存。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [KwargsForCausalLM](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py#L416) | 组合视觉塔、图文投影和语言模型，处理多模态输入及KV缓存。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [PaliGemmaForConditionalGeneration](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py#L425) | 组合视觉塔、图文投影和语言模型，处理多模态输入及KV缓存。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [PaliGemmaForConditionalGeneration.__init__](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py#L437) | 初始化PaliGemmaForConditionalGeneration的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.model、self.lm_head。 |
| [PaliGemmaForConditionalGeneration.get_input_embeddings](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py#L445) | 返回底层组件引用，供模型组合或框架兼容接口使用，不会在这里执行一次完整推理。 |
| [PaliGemmaForConditionalGeneration.set_input_embeddings](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py#L450) | 替换底层组件引用；这属于模型组装接口，不是梯度更新。 |
| [PaliGemmaForConditionalGeneration.get_output_embeddings](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py#L455) | 返回底层组件引用，供模型组合或框架兼容接口使用，不会在这里执行一次完整推理。 |
| [PaliGemmaForConditionalGeneration.set_output_embeddings](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py#L460) | 替换底层组件引用；这属于模型组装接口，不是梯度更新。 |
| [PaliGemmaForConditionalGeneration.set_decoder](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py#L465) | 替换底层组件引用；这属于模型组装接口，不是梯度更新。 |
| [PaliGemmaForConditionalGeneration.get_decoder](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py#L470) | 返回底层组件引用，供模型组合或框架兼容接口使用，不会在这里执行一次完整推理。 |
| [PaliGemmaForConditionalGeneration.get_image_features](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py#L476) | 本函数位于“第三方适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.model.get_image_features 追踪具体实现。 |
| [PaliGemmaForConditionalGeneration.language_model](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py#L483) | 本函数位于“第三方适配”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。 |
| [PaliGemmaForConditionalGeneration.vision_tower](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py#L489) | 本函数位于“第三方适配”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。 |
| [PaliGemmaForConditionalGeneration.multi_modal_projector](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py#L495) | 本函数位于“第三方适配”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。 |
| [PaliGemmaForConditionalGeneration.forward](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py#L504) | 执行组合视觉塔、图文投影和语言模型，处理多模态输入及KV缓存。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [PaliGemmaForConditionalGeneration.prepare_inputs_for_generation](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py#L596) | 本函数位于“第三方适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 super().prepare_inputs_for_generation → model_inputs.get → isinstance 追踪具体实现。 |
| [PaliGemmaForConditionalGeneration._prepare_4d_causal_attention_mask_with_cache_position](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py#L648) | 本函数位于“第三方适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 attention_mask.dim → torch.finfo → torch.full 追踪具体实现。 |
