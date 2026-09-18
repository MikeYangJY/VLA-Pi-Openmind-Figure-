# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：核心模型对照｜π0/π0.5的PyTorch版本：forward计算flow loss，sample_actions进行Euler积分。
# 阅读顺序：先对照JAX版的embed_prefix/embed_suffix，再读forward、sample_actions和denoise_step；显存优化辅助函数后读。
# 重点边界：gradient checkpointing是重算激活省显存，不是保存模型权重；pi05分支的AdaRMS时间条件需追到Gemma层。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


# 【get_safe_dtype】本函数位于“核心模型对照”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。
# 输入接口：target_dtype；device_type。
# 返回值可从这里追踪：torch.float32 / torch.float64。
def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


# 【create_sinusoidal_pos_embedding】PyTorch版flow时间sin/cos编码，输入[B]，输出[B,D]，供后续动作/时间融合。
# 输入接口：time:torch.tensor；dimension:int；min_period:float；max_period:float；device。
# 返回类型：Tensor；类型/shape约定需与调用方配套。
# 内部调用线索：ValueError → get_safe_dtype → torch.linspace → torch.cat → torch.sin（含分支中的调用，实际路径由条件决定）。
def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


# 【sample_beta】本函数位于“核心模型对照”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 torch.as_tensor → torch.distributions.Beta → dist.sample 追踪具体实现。
# 输入接口：alpha；beta；bsize；device。
# 返回值可从这里追踪：dist.sample((bsize,))。
# 内部调用线索：torch.as_tensor → torch.distributions.Beta → dist.sample（含分支中的调用，实际路径由条件决定）。
def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


# 【make_att_2d_masks】PyTorch版分块注意力mask。cumsum给各token编号，比较编号得到能否注意，再排除padding。
# 输入接口：pad_masks；att_masks。
# 返回值可从这里追踪：att_2d_masks & pad_2d_masks。
# 内部调用线索：ValueError → torch.cumsum（含分支中的调用，实际路径由条件决定）。
def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


# 【PI0Pytorch】共享模型的PyTorch实现，forward是训练损失，sample_actions是推理动作。
class PI0Pytorch(nn.Module):
    # 【PI0Pytorch.__init__】初始化PI0Pytorch的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.config、self.pi05、self.paligemma_with_expert、self.action_in_proj、self.action_out_proj。
    # 输入接口：config。
    # 内部调用线索：super().__init__ → _gemma.get_config → PaliGemmaWithExpertModel → nn.Linear → torch.set_float32_matmul_precision（含分支中的调用，实际路径由条件决定）。
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)

        # 版本分支：此分支是π0.5；对应else为π0。
        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(config.action_dim, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        if config.pytorch_compile_mode is not None:
            self.sample_actions = torch.compile(self.sample_actions, mode=config.pytorch_compile_mode)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    # 【PI0Pytorch.gradient_checkpointing_enable】切换训练激活重算选项，在显存占用和额外计算之间取舍，不涉及磁盘存档。
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    # 【PI0Pytorch.gradient_checkpointing_disable】切换训练激活重算选项，在显存占用和额外计算之间取舍，不涉及磁盘存档。
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    # 【PI0Pytorch.is_gradient_checkpointing_enabled】本函数位于“核心模型对照”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。
    # 返回值可从这里追踪：self.gradient_checkpointing_enabled。
    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    # 【PI0Pytorch._apply_checkpoint】在训练且启用时用激活重算换显存；否则直接调用传入函数。这里的checkpoint不是磁盘权重存档。
    # 输入接口：func。
    # 返回值可从这里追踪：torch.utils.checkpoint.checkpoint(func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs) / func(*args, **kwargs)。
    # 内部调用线索：torch.utils.checkpoint.checkpoint → func（含分支中的调用，实际路径由条件决定）。
    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    # 【PI0Pytorch._prepare_attention_masks_4d】将布尔可见性[B,N,N]转成带head广播轴的加性mask[B,1,N,N]：可见为0，不可见为很大的负数。
    # 输入接口：att_2d_masks。
    # 返回值可从这里追踪：torch.where(att_2d_masks_4d, 0.0, -2.3819763e+38)。
    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    # 【PI0Pytorch._preprocess_observation】本函数位于“核心模型对照”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 _preprocessing.preprocess_observation_pytorch → observation.images.values → observation.image_masks.values 追踪具体实现。
    # 输入接口：observation；train。
    # 返回值可从这里追踪：(list(observation.images.values()), list(observation.image_masks.values()), observation.tokenized_prompt, obse…。
    # 内部调用线索：_preprocessing.preprocess_observation_pytorch → observation.images.values → observation.image_masks.values（含分支中的调用，实际路径由条件决定）。
    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    # 【PI0Pytorch.sample_noise】生成与动作形状相同的标准高斯噪声；不涉及机器人传感器噪声标定。
    # 输入接口：shape；device。
    # 返回值可从这里追踪：torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)。
    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    # 【PI0Pytorch.sample_time】从Beta(1.5,1)采样flow时间并缩放到接近(0,1]范围，决定训练样本噪声强度。
    # 输入接口：bsize；device。
    # 返回值可从这里追踪：time.to(dtype=torch.float32, device=device)。
    # 内部调用线索：sample_beta → time.to（含分支中的调用，实际路径由条件决定）。
    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    # 【PI0Pytorch.embed_prefix】图像经视觉塔得到patch embedding，语言查embedding并缩放；沿token轴拼接，并建立前缀有效性与注意力分块mask。
    # 输入接口：images；img_masks；lang_tokens；lang_masks。
    # 返回类型：tuple[torch.Tensor, torch.Tensor, torch.Tensor]；类型/shape约定需与调用方配套。
    # 内部调用线索：self._apply_checkpoint → embs.append → pad_masks.append → img_mask[:, None].expand → torch.cat（含分支中的调用，实际路径由条件决定）。
    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            # 【PI0Pytorch.embed_prefix.image_embed_func】本函数位于“核心模型对照”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.paligemma_with_expert.embed_image 追踪具体实现。
            # 输入接口：img。
            # 返回值可从这里追踪：self.paligemma_with_expert.embed_image(img)。
            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        # 【PI0Pytorch.embed_prefix.lang_embed_func】本函数位于“核心模型对照”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.paligemma_with_expert.embed_language_tokens → math.sqrt 追踪具体实现。
        # 输入接口：lang_tokens。
        # 返回值可从这里追踪：lang_emb * math.sqrt(lang_emb_dim)。
        # 内部调用线索：self.paligemma_with_expert.embed_language_tokens → math.sqrt（含分支中的调用，实际路径由条件决定）。
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    # 【PI0Pytorch.embed_suffix】把动作、可选连续state与flow时间变成专家输入；pi05分支通过time MLP提供AdaRMS条件。
    # 输入接口：state；noisy_actions；timestep。
    # 返回值可从这里追踪：(embs, pad_masks, att_masks, adarms_cond)。
    # 内部调用线索：state.to → self._apply_checkpoint → embs.append → torch.ones → pad_masks.append（含分支中的调用，实际路径由条件决定）。
    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # 版本分支：此分支是π0；对应else为π0.5。
        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            # 【PI0Pytorch.embed_suffix.state_proj_func】本函数位于“核心模型对照”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.state_proj 追踪具体实现。
            # 输入接口：state。
            # 返回值可从这里追踪：self.state_proj(state)。
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        # 【PI0Pytorch.embed_suffix.action_proj_func】本函数位于“核心模型对照”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.action_in_proj 追踪具体实现。
        # 输入接口：noisy_actions。
        # 返回值可从这里追踪：self.action_in_proj(noisy_actions)。
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        # 版本分支：此分支是π0；对应else为π0.5。
        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            # 【PI0Pytorch.embed_suffix.mlp_func】本函数位于“核心模型对照”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.action_time_mlp_in → F.silu → self.action_time_mlp_out 追踪具体实现。
            # 输入接口：action_time_emb。
            # 返回值可从这里追踪：self.action_time_mlp_out(x)。
            # 内部调用线索：self.action_time_mlp_in → F.silu → self.action_time_mlp_out（含分支中的调用，实际路径由条件决定）。
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            # 学习提示：π0这一分支已在动作token中融合时间，不使用AdaRMS时间条件。
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            # 【PI0Pytorch.embed_suffix.time_mlp_func】本函数位于“核心模型对照”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.time_mlp_in → F.silu → self.time_mlp_out 追踪具体实现。
            # 输入接口：time_emb。
            # 返回值可从这里追踪：F.silu(x)。
            # 内部调用线索：self.time_mlp_in → F.silu → self.time_mlp_out（含分支中的调用，实际路径由条件决定）。
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            # 学习提示：π0.5把时间信息作为归一化条件传下去；动作token本身不再拼接重复时间向量。
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    # 【PI0Pytorch.forward】训练前向：构造x_t与目标u_t，联合运行前缀/后缀，取动作token输出向量场，返回未归约MSE；训练循环再取平均并反传。
    # 输入接口：observation；actions；noise；time。
    # 返回类型：Tensor；类型/shape约定需与调用方配套。
    # 内部调用线索：self._preprocess_observation → self.sample_noise → self.sample_time → self.embed_prefix → self.embed_suffix（含分支中的调用，实际路径由条件决定）。
    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        # 学习提示：时间[B]扩成[B,1,1]，以便与动作[B,H,A]逐元素混合。
        time_expanded = time[:, None, None]
        # 学习提示：flow训练样本：t=1时是噪声，t=0时是真实动作，中间为线性插值。
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        # 学习提示：目标向量场是dx_t/dt=noise-actions；推理dt为负，所以最终从噪声走向动作。
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        # 【PI0Pytorch.forward.forward_func】本函数位于“核心模型对照”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.paligemma_with_expert.forward 追踪具体实现。
        # 输入接口：prefix_embs；suffix_embs；att_2d_masks_4d；position_ids；adarms_cond。
        # 返回值可从这里追踪：suffix_out。
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        # 【PI0Pytorch.forward.action_out_proj_func】本函数位于“核心模型对照”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.action_out_proj 追踪具体实现。
        # 输入接口：suffix_out。
        # 返回值可从这里追踪：self.action_out_proj(suffix_out)。
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        # 学习提示：返回[B,H,A]的逐元素误差；最终标量平均在PyTorch训练循环里完成。
        return F.mse_loss(u_t, v_t, reduction="none")

    # 【PI0Pytorch.sample_actions】无梯度推理：缓存前缀K/V，从噪声开始循环denoise_step与Euler更新，输出整个动作块。
    # 输入接口：device；observation；noise；num_steps。
    # 返回类型：Tensor；类型/shape约定需与调用方配套。
    # 内部调用线索：self.sample_noise → self._preprocess_observation → self.embed_prefix → make_att_2d_masks → torch.cumsum（含分支中的调用，实际路径由条件决定）。
    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # 学习提示：负步长：从t=1积分到t=0。num_steps是去噪/积分次数，不是action_horizon。
        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            # 学习提示：按当前向量场更新整个动作块，下一轮继续细化。
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    # 【PI0Pytorch.denoise_step】仅重算当前动作后缀，拼接对前缀缓存的注意力mask，得到此flow时间的v_t。
    # 输入接口：state；prefix_pad_masks；past_key_values；x_t；timestep。
    # 返回值可从这里追踪：self.action_out_proj(suffix_out)。
    # 内部调用线索：self.embed_suffix → prefix_pad_masks[:, None, :].expand → make_att_2d_masks → torch.cat → torch.sum（含分支中的调用，实际路径由条件决定）。
    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
