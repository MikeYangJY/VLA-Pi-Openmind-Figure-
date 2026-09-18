# `src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py` 中文阅读说明

**定位：** 第三方适配。

SigLIP图像/文本编码网络与兼容任务头；π0主要取其中视觉塔。

**建议读法：** 从SiglipVisionEmbeddings到SiglipEncoder和SiglipVisionTransformer，注意patch与位置编码。

**易错点：** 文件中完整图文对比学习/分类头并不都被π0动作路径调用。

[注释源码](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [_trunc_normal_](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L50) | 本函数位于“第三方适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 warnings.warn → norm_cdf → tensor.uniform_ 追踪具体实现。 |
| [_trunc_normal_.norm_cdf](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L57) | 本函数位于“第三方适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 math.erf → math.sqrt 追踪具体实现。 |
| [trunc_normal_tf_](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L94) | 本函数位于“第三方适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 torch.no_grad → _trunc_normal_ → tensor.mul_(std).add_ 追踪具体实现。 |
| [variance_scaling_](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L123) | 本函数位于“第三方适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 _calculate_fan_in_and_fan_out → trunc_normal_tf_ → math.sqrt 追踪具体实现。 |
| [lecun_normal_](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L150) | 本函数位于“第三方适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 variance_scaling_ 追踪具体实现。 |
| [default_flax_embed_init](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L156) | 本函数位于“第三方适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 variance_scaling_ 追踪具体实现。 |
| [SiglipVisionModelOutput](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L168) | SigLIP图像/文本编码网络与兼容任务头；π0主要取其中视觉塔。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [SiglipTextModelOutput](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L188) | SigLIP图像/文本编码网络与兼容任务头；π0主要取其中视觉塔。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [SiglipOutput](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L204) | SigLIP图像/文本编码网络与兼容任务头；π0主要取其中视觉塔。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [SiglipOutput.to_tuple](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L235) | 本函数位于“第三方适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 getattr(self, k).to_tuple → getattr → self.keys 追踪具体实现。 |
| [SiglipVisionEmbeddings](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L243) | SigLIP图像/文本编码网络与兼容任务头；π0主要取其中视觉塔。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [SiglipVisionEmbeddings.__init__](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L247) | 初始化SiglipVisionEmbeddings的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.config、self.embed_dim、self.image_size、self.patch_size、self.patch_embedding。 |
| [SiglipVisionEmbeddings.interpolate_pos_encoding](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L271) | 本函数位于“第三方适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 torch.jit.is_tracing → self.position_embedding → self.position_embedding.weight.unsqueeze 追踪具体实现。 |
| [SiglipVisionEmbeddings.forward](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L313) | 执行SigLIP图像/文本编码网络与兼容任务头；π0主要取其中视觉塔。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [SiglipTextEmbeddings](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L328) | SigLIP图像/文本编码网络与兼容任务头；π0主要取其中视觉塔。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [SiglipTextEmbeddings.__init__](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L332) | 初始化SiglipTextEmbeddings的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.token_embedding、self.position_embedding。 |
| [SiglipTextEmbeddings.forward](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L348) | 执行SigLIP图像/文本编码网络与兼容任务头；π0主要取其中视觉塔。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [eager_attention_forward](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L379) | 本函数位于“第三方适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 torch.matmul → key.transpose → nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to 追踪具体实现。 |
| [SiglipAttention](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L403) | SigLIP图像/文本编码网络与兼容任务头；π0主要取其中视觉塔。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [SiglipAttention.__init__](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L409) | 初始化SiglipAttention的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.config、self.embed_dim、self.num_heads、self.head_dim、self.scale。 |
| [SiglipAttention.forward](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L433) | 执行SigLIP图像/文本编码网络与兼容任务头；π0主要取其中视觉塔。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [SiglipMLP](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L483) | SigLIP图像/文本编码网络与兼容任务头；π0主要取其中视觉塔。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [SiglipMLP.__init__](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L487) | 初始化SiglipMLP的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.config、self.activation_fn、self.fc1、self.fc2。 |
| [SiglipMLP.forward](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L498) | 执行SigLIP图像/文本编码网络与兼容任务头；π0主要取其中视觉塔。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [SiglipEncoderLayer](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L506) | SigLIP图像/文本编码网络与兼容任务头；π0主要取其中视觉塔。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [SiglipEncoderLayer.__init__](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L510) | 初始化SiglipEncoderLayer的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.embed_dim、self.layer_norm1、self.self_attn、self.layer_norm2、self.mlp。 |
| [SiglipEncoderLayer.forward](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L522) | 执行SigLIP图像/文本编码网络与兼容任务头；π0主要取其中视觉塔。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [SiglipPreTrainedModel](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L563) | SigLIP图像/文本编码网络与兼容任务头；π0主要取其中视觉塔。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [SiglipPreTrainedModel._init_weights](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L583) | 按层类型初始化参数，需与随后加载预训练权重的步骤区分。 |
| [SiglipEncoder](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L632) | SigLIP图像/文本编码网络与兼容任务头；π0主要取其中视觉塔。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [SiglipEncoder.__init__](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L644) | 初始化SiglipEncoder的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.config、self.layers、self.gradient_checkpointing。 |
| [SiglipEncoder.forward](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L656) | 执行SigLIP图像/文本编码网络与兼容任务头；π0主要取其中视觉塔。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [SiglipTextTransformer](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L720) | SigLIP图像/文本编码网络与兼容任务头；π0主要取其中视觉塔。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [SiglipTextTransformer.__init__](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L724) | 初始化SiglipTextTransformer的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.config、self.embeddings、self.encoder、self.final_layer_norm、self.head。 |
| [SiglipTextTransformer.forward](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L741) | 执行SigLIP图像/文本编码网络与兼容任务头；π0主要取其中视觉塔。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [SiglipTextModel](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L796) | SigLIP图像/文本编码网络与兼容任务头；π0主要取其中视觉塔。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [SiglipTextModel.__init__](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L802) | 初始化SiglipTextModel的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.text_model。 |
| [SiglipTextModel.get_input_embeddings](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L810) | 返回底层组件引用，供模型组合或框架兼容接口使用，不会在这里执行一次完整推理。 |
| [SiglipTextModel.set_input_embeddings](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L815) | 替换底层组件引用；这属于模型组装接口，不是梯度更新。 |
| [SiglipTextModel.forward](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L823) | 执行SigLIP图像/文本编码网络与兼容任务头；π0主要取其中视觉塔。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [SiglipVisionTransformer](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L858) | SigLIP图像/文本编码网络与兼容任务头；π0主要取其中视觉塔。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [SiglipVisionTransformer.__init__](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L862) | 初始化SiglipVisionTransformer的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.config、self.embeddings、self.encoder、self.post_layernorm、self.use_head。 |
| [SiglipVisionTransformer.forward](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L880) | 执行SigLIP图像/文本编码网络与兼容任务头；π0主要取其中视觉塔。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [SiglipMultiheadAttentionPoolingHead](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L917) | SigLIP图像/文本编码网络与兼容任务头；π0主要取其中视觉塔。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [SiglipMultiheadAttentionPoolingHead.__init__](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L923) | 初始化SiglipMultiheadAttentionPoolingHead的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.probe、self.attention、self.layernorm、self.mlp。 |
| [SiglipMultiheadAttentionPoolingHead.forward](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L935) | 执行SigLIP图像/文本编码网络与兼容任务头；π0主要取其中视觉塔。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [SiglipVisionModel](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L954) | SigLIP图像/文本编码网络与兼容任务头；π0主要取其中视觉塔。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [SiglipVisionModel.__init__](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L961) | 初始化SiglipVisionModel的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.vision_model。 |
| [SiglipVisionModel.get_input_embeddings](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L971) | 返回底层组件引用，供模型组合或框架兼容接口使用，不会在这里执行一次完整推理。 |
| [SiglipVisionModel.forward](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L979) | 执行SigLIP图像/文本编码网络与兼容任务头；π0主要取其中视觉塔。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [SiglipModel](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L1017) | SigLIP图像/文本编码网络与兼容任务头；π0主要取其中视觉塔。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [SiglipModel.__init__](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L1023) | 初始化SiglipModel的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.text_model、self.vision_model、self.logit_scale、self.logit_bias。 |
| [SiglipModel.get_text_features](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L1059) | 本函数位于“第三方适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.text_model 追踪具体实现。 |
| [SiglipModel.get_image_features](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L1108) | 本函数位于“第三方适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.vision_model 追踪具体实现。 |
| [SiglipModel.forward](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L1162) | 执行SigLIP图像/文本编码网络与兼容任务头；π0主要取其中视觉塔。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [SiglipForImageClassification](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L1266) | SigLIP图像/文本编码网络与兼容任务头；π0主要取其中视觉塔。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [SiglipForImageClassification.__init__](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L1273) | 初始化SiglipForImageClassification的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.num_labels、self.vision_model、self.classifier。 |
| [SiglipForImageClassification.forward](../../../../../../../code/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py#L1297) | 执行SigLIP图像/文本编码网络与兼容任务头；π0主要取其中视觉塔。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
