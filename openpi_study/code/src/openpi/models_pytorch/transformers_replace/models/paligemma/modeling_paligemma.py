# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：第三方适配｜组合视觉塔、图文投影和语言模型，处理多模态输入及KV缓存。
# 阅读顺序：跟踪get_image_features、PaliGemmaMultiModalProjector与PaliGemmaModel.forward；再看缓存mask。
# 重点边界：理解图像特征怎样进入语言hidden space即可，不必先学完标准文本生成API。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
# coding=utf-8
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch PaliGemmamodel."""

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...cache_utils import Cache, HybridCache, StaticCache
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import LossKwargs, ModelOutput, auto_docstring, can_return_tuple, is_torchdynamo_compiling, logging
from ..auto import AutoModel
from .configuration_paligemma import PaliGemmaConfig


logger = logging.get_logger(__name__)


# 【PaligemmaModelOutputWithPast】组合视觉塔、图文投影和语言模型，处理多模态输入及KV缓存。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Paligemma outputs, with hidden states and attentions.
    """
)
class PaligemmaModelOutputWithPast(BaseModelOutputWithPast):
    r"""
    past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    image_hidden_states: Optional[torch.FloatTensor] = None


# 【PaliGemmaCausalLMOutputWithPast】组合视觉塔、图文投影和语言模型，处理多模态输入及KV缓存。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclass
@auto_docstring(
    custom_intro="""
    Base class for PaliGemma causal language model (or autoregressive) outputs.
    """
)
class PaliGemmaCausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.text_config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder after projecting last hidden state.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Union[list[torch.FloatTensor], Cache]] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None


# 【PaliGemmaMultiModalProjector】组合视觉塔、图文投影和语言模型，处理多模态输入及KV缓存。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
class PaliGemmaMultiModalProjector(nn.Module):
    # 【PaliGemmaMultiModalProjector.__init__】初始化PaliGemmaMultiModalProjector的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.linear。
    # 输入接口：config:PaliGemmaConfig。
    # 内部调用线索：super().__init__ → nn.Linear（含分支中的调用，实际路径由条件决定）。
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    # 【PaliGemmaMultiModalProjector.forward】执行组合视觉塔、图文投影和语言模型，处理多模态输入及KV缓存。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。
    # 输入接口：image_features。
    # 返回值可从这里追踪：hidden_states。
    def forward(self, image_features):
        hidden_states = self.linear(image_features)

        return hidden_states


# 【PaliGemmaPreTrainedModel】组合视觉塔、图文投影和语言模型，处理多模态输入及KV缓存。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@auto_docstring
class PaliGemmaPreTrainedModel(PreTrainedModel):
    config_class = PaliGemmaConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _no_split_modules = ["PaliGemmaMultiModalProjector"]
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    # 【PaliGemmaPreTrainedModel._init_weights】按层类型初始化参数，需与随后加载预训练权重的步骤区分。
    # 输入接口：module。
    # 内部调用线索：getattr → self.config.get_text_config → isinstance → module.weight.data.normal_ → module.bias.data.zero_（含分支中的调用，实际路径由条件决定）。
    def _init_weights(self, module):
        # important: this ported version of PaliGemmaisn't meant for training from scratch - only
        # inference and fine-tuning
        std = getattr(self.config, "initializer_range", self.config.get_text_config().initializer_range)

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()


# 【PaliGemmaModel】组合视觉塔、图文投影和语言模型，处理多模态输入及KV缓存。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@auto_docstring(
    custom_intro="""
    The Base Paligemma model which consists of a vision backbone and a language model withou language modeling head.,
    """
)
class PaliGemmaModel(PaliGemmaPreTrainedModel):
    _checkpoint_conversion_mapping = {"language_model.model": "language_model"}
    # we are filtering the logits/labels so we shouldn't divide the loss based on num_items_in_batch
    accepts_loss_kwargs = False

    # 【PaliGemmaModel.__init__】初始化PaliGemmaModel的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.vision_tower、self.multi_modal_projector、self.vocab_size、self.language_model、self.pad_token_id。
    # 输入接口：config:PaliGemmaConfig。
    # 内部调用线索：super().__init__ → AutoModel.from_config → PaliGemmaMultiModalProjector → self.post_init（含分支中的调用，实际路径由条件决定）。
    def __init__(self, config: PaliGemmaConfig):
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config=config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.text_config.vocab_size

        language_model = AutoModel.from_config(config=config.text_config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.post_init()

    # Copied from transformers.models.llava.modeling_llava.LlavaModel.get_input_embeddings with Llava->PaliGemma
    # 【PaliGemmaModel.get_input_embeddings】返回底层组件引用，供模型组合或框架兼容接口使用，不会在这里执行一次完整推理。
    # 返回值可从这里追踪：self.language_model.get_input_embeddings()。
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    # Copied from transformers.models.llava.modeling_llava.LlavaModel.set_input_embeddings with Llava->PaliGemma
    # 【PaliGemmaModel.set_input_embeddings】替换底层组件引用；这属于模型组装接口，不是梯度更新。
    # 输入接口：value。
    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    # 【PaliGemmaModel.set_decoder】替换底层组件引用；这属于模型组装接口，不是梯度更新。
    # 输入接口：decoder。
    def set_decoder(self, decoder):
        self.language_model = decoder

    # 【PaliGemmaModel.get_decoder】返回底层组件引用，供模型组合或框架兼容接口使用，不会在这里执行一次完整推理。
    # 返回值可从这里追踪：self.language_model。
    def get_decoder(self):
        return self.language_model

    # 【PaliGemmaModel._update_causal_mask】本函数位于“第三方适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 isinstance → torch.finfo → past_key_values.get_max_cache_shape 追踪具体实现。
    # 输入接口：attention_mask；token_type_ids；past_key_values；cache_position；input_tensor；is_training:Optional[bool]。
    # 返回值可从这里追踪：attention_mask / None。
    # 内部调用线索：isinstance → torch.finfo → past_key_values.get_max_cache_shape → attention_mask.dim → torch.full（含分支中的调用，实际路径由条件决定）。
    def _update_causal_mask(
        self,
        attention_mask,
        token_type_ids=None,
        past_key_values=None,
        cache_position=None,
        input_tensor=None,
        is_training: Optional[bool] = None,
    ):
        if self.config.text_config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None
        is_training = is_training if is_training is not None else self.training
        using_static_cache = isinstance(past_key_values, StaticCache)
        min_dtype = torch.finfo(self.dtype).min
        if input_tensor is None:
            input_tensor = attention_mask

        inputs_lead_dim, sequence_length = input_tensor.shape[:2]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        elif isinstance(past_key_values, HybridCache):
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else cache_position[0] + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            return attention_mask

        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=self.dtype, device=cache_position.device
        )
        # Causal diagonal mask only if training, otherwise attend to the whole prefix. Training-specific attn for prefix is handled below
        if sequence_length != 1:
            if is_training:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            else:
                causal_mask[:, :sequence_length] = 0.0

        causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(inputs_lead_dim, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]

            # First unmask prefix tokens during training
            if is_training:
                if token_type_ids is None:
                    raise ValueError("Token type ids must be provided during training")
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    token_type_ids[:, None, None, :].to(causal_mask.device) == 0, 0
                )

            # Then apply padding mask (will mask pad tokens)
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

        return causal_mask

    # 【PaliGemmaModel.get_image_features】本函数位于“第三方适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.vision_tower → self.multi_modal_projector 追踪具体实现。
    # 输入接口：pixel_values:torch.FloatTensor。
    # 返回值可从这里追踪：image_features。
    # 内部调用线索：self.vision_tower → self.multi_modal_projector（含分支中的调用，实际路径由条件决定）。
    def get_image_features(self, pixel_values: torch.FloatTensor):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        image_outputs = self.vision_tower(pixel_values)
        selected_image_feature = image_outputs.last_hidden_state
        image_features = self.multi_modal_projector(selected_image_feature)
        return image_features

    # 【PaliGemmaModel.forward】执行组合视觉塔、图文投影和语言模型，处理多模态输入及KV缓存。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。
    # 输入接口：input_ids:torch.LongTensor；pixel_values:torch.FloatTensor；attention_mask:Optional[torch.Tensor]；position_ids:Optional[torch.LongTensor]；past_key_values:Optional[Union[list[torch.FloatTensor], Cache]]；token_type_ids:Optional[torch.LongTensor]；cache_position:Optional[torch.LongTensor]；inputs_embeds:Optional[torch.FloatTensor]；其余见函数签名。
    # 返回类型：Union[tuple, PaligemmaModelOutputWithPast]；类型/shape约定需与调用方配套。
    # 内部调用线索：ValueError → input_ids.clone → self.get_input_embeddings() → self.get_input_embeddings → past_key_values.get_seq_length（含分支中的调用，实际路径由条件决定）。
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[list[torch.FloatTensor], Cache]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, PaligemmaModelOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        >>> model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma2-3b-mix-224")
        >>> processor = AutoProcessor.from_pretrained("google/paligemma2-3b-mix-224")

        >>> prompt = "Where is the cat standing?"
        >>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt,  return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs,)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Where is the cat standing?\nsnow"
        ```"""

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        is_training = token_type_ids is not None and labels is not None

        # Replace image id woth PAD if the image token if OOV, to avoid index-errors
        if input_ids is not None and self.config.image_token_id >= self.vocab_size:
            special_image_mask = input_ids == self.config.image_token_id
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0) + 1  # Paligemma positions are 1-indexed

        # Merge text and images
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)

            if input_ids is None:
                special_image_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            else:
                special_image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
                special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

            if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != image_features.numel():
                image_tokens_in_text = (special_image_mask).sum(dim=1).sum(dim=0)[0]
                raise ValueError(
                    f"Number of images does not match number of special image tokens in the input text. "
                    f"Got {image_tokens_in_text} image tokens in the text but {image_features.shape[0] * image_features.shape[1]} "
                    "tokens from image embeddings."
                )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        causal_mask = self._update_causal_mask(
            attention_mask, token_type_ids, past_key_values, cache_position, inputs_embeds, is_training
        )
        outputs = self.language_model(
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        return PaligemmaModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )


# 【KwargsForCausalLM】组合视觉塔、图文投影和语言模型，处理多模态输入及KV缓存。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


# 【PaliGemmaForConditionalGeneration】组合视觉塔、图文投影和语言模型，处理多模态输入及KV缓存。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@auto_docstring(
    custom_intro="""
    The Base Paligemma model which consists of a vision backbone and a language model without language modeling head.,
    """
)
class PaliGemmaForConditionalGeneration(PaliGemmaPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {
        "^language_model.model": "model.language_model",
        "^vision_tower": "model.vision_tower",
        "^multi_modal_projector": "model.multi_modal_projector",
        "^language_model.lm_head": "lm_head",
    }
    _tied_weights_keys = ["lm_head.weight"]

    # 【PaliGemmaForConditionalGeneration.__init__】初始化PaliGemmaForConditionalGeneration的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.model、self.lm_head。
    # 输入接口：config:PaliGemmaConfig。
    # 内部调用线索：super().__init__ → PaliGemmaModel → nn.Linear → self.post_init（含分支中的调用，实际路径由条件决定）。
    def __init__(self, config: PaliGemmaConfig):
        super().__init__(config)
        self.model = PaliGemmaModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    # 【PaliGemmaForConditionalGeneration.get_input_embeddings】返回底层组件引用，供模型组合或框架兼容接口使用，不会在这里执行一次完整推理。
    # 返回值可从这里追踪：self.model.get_input_embeddings()。
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    # 【PaliGemmaForConditionalGeneration.set_input_embeddings】替换底层组件引用；这属于模型组装接口，不是梯度更新。
    # 输入接口：value。
    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    # 【PaliGemmaForConditionalGeneration.get_output_embeddings】返回底层组件引用，供模型组合或框架兼容接口使用，不会在这里执行一次完整推理。
    # 返回值可从这里追踪：self.lm_head。
    def get_output_embeddings(self):
        return self.lm_head

    # 【PaliGemmaForConditionalGeneration.set_output_embeddings】替换底层组件引用；这属于模型组装接口，不是梯度更新。
    # 输入接口：new_embeddings。
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 【PaliGemmaForConditionalGeneration.set_decoder】替换底层组件引用；这属于模型组装接口，不是梯度更新。
    # 输入接口：decoder。
    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    # 【PaliGemmaForConditionalGeneration.get_decoder】返回底层组件引用，供模型组合或框架兼容接口使用，不会在这里执行一次完整推理。
    # 返回值可从这里追踪：self.model.get_decoder()。
    def get_decoder(self):
        return self.model.get_decoder()

    # 【PaliGemmaForConditionalGeneration.get_image_features】本函数位于“第三方适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.model.get_image_features 追踪具体实现。
    # 输入接口：pixel_values。
    # 返回值可从这里追踪：self.model.get_image_features(pixel_values)。
    def get_image_features(self, pixel_values):
        return self.model.get_image_features(pixel_values)

    # Make modules available throught conditional class for BC
    # 【PaliGemmaForConditionalGeneration.language_model】本函数位于“第三方适配”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。
    # 返回值可从这里追踪：self.model.language_model。
    @property
    def language_model(self):
        return self.model.language_model

    # 【PaliGemmaForConditionalGeneration.vision_tower】本函数位于“第三方适配”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。
    # 返回值可从这里追踪：self.model.vision_tower。
    @property
    def vision_tower(self):
        return self.model.vision_tower

    # 【PaliGemmaForConditionalGeneration.multi_modal_projector】本函数位于“第三方适配”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。
    # 返回值可从这里追踪：self.model.multi_modal_projector。
    @property
    def multi_modal_projector(self):
        return self.model.multi_modal_projector

    # 【PaliGemmaForConditionalGeneration.forward】执行组合视觉塔、图文投影和语言模型，处理多模态输入及KV缓存。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。
    # 输入接口：input_ids:torch.LongTensor；pixel_values:torch.FloatTensor；attention_mask:Optional[torch.Tensor]；position_ids:Optional[torch.LongTensor]；past_key_values:Optional[Union[list[torch.FloatTensor], Cache]]；token_type_ids:Optional[torch.LongTensor]；cache_position:Optional[torch.LongTensor]；inputs_embeds:Optional[torch.FloatTensor]；其余见函数签名。
    # 返回类型：Union[tuple, PaliGemmaCausalLMOutputWithPast]；类型/shape约定需与调用方配套。
    # 内部调用线索：self.model → isinstance → slice → self.lm_head → self.loss_function（含分支中的调用，实际路径由条件决定）。
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[list[torch.FloatTensor], Cache]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[tuple, PaliGemmaCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        >>> model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma2-3b-mix-224")
        >>> processor = AutoProcessor.from_pretrained("google/paligemma2-3b-mix-224")

        >>> prompt = "Where is the cat standing?"
        >>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt,  return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs,)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Where is the cat standing?\nsnow"
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return PaliGemmaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    # 【PaliGemmaForConditionalGeneration.prepare_inputs_for_generation】本函数位于“第三方适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 super().prepare_inputs_for_generation → model_inputs.get → isinstance 追踪具体实现。
    # 输入接口：input_ids；past_key_values；inputs_embeds；cache_position；position_ids；pixel_values；attention_mask；token_type_ids；其余见函数签名。
    # 返回值可从这里追踪：model_inputs。
    # 内部调用线索：super().prepare_inputs_for_generation → model_inputs.get → isinstance → self.model._update_causal_mask（含分支中的调用，实际路径由条件决定）。
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        pixel_values=None,
        attention_mask=None,
        token_type_ids=None,
        use_cache=True,
        logits_to_keep=None,
        labels=None,
        **kwargs,
    ):
        # Overwritten -- custom `position_ids` and `pixel_values` handling
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        # position_ids in Paligemma are 1-indexed
        if model_inputs.get("position_ids") is not None:
            model_inputs["position_ids"] += 1
        # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
        # Otherwise we need pixel values to be passed to model. NOTE: use_cache=False needs pixel_values always
        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
        is_training = token_type_ids is not None and labels is not None
        if cache_position[0] == 0 and isinstance(past_key_values, HybridCache):
            input_tensor = inputs_embeds if inputs_embeds is not None else input_ids
            causal_mask = self.model._update_causal_mask(
                attention_mask, token_type_ids, past_key_values, cache_position, input_tensor, is_training
            )
            model_inputs["attention_mask"] = causal_mask

        return model_inputs

    # 【PaliGemmaForConditionalGeneration._prepare_4d_causal_attention_mask_with_cache_position】本函数位于“第三方适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 attention_mask.dim → torch.finfo → torch.full 追踪具体实现。
    # 输入接口：attention_mask:torch.Tensor；sequence_length:int；target_length:int；dtype:torch.dtype；cache_position:torch.Tensor；batch_size:int。
    # 返回值可从这里追踪：causal_mask。
    # 内部调用线索：attention_mask.dim → torch.finfo → torch.full → torch.triu → torch.arange（含分支中的调用，实际路径由条件决定）。
    @staticmethod
    # Copied from transformers.models.gptj.modeling_gptj.GPTJModel._prepare_4d_causal_attention_mask_with_cache_position
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


__all__ = ["PaliGemmaForConditionalGeneration", "PaliGemmaPreTrainedModel", "PaliGemmaModel"]
