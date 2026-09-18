# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：核心模型｜π0和π0.5共用的JAX实现，串起图像/语言条件、动作专家、flow训练和动作采样。
# 阅读顺序：先读compute_loss与sample_actions，再读embed_prefix、embed_suffix和make_attn_mask。
# 重点边界：t=1是噪声，t=0是真实动作；pi05开关改变状态入口和时间条件，不代表论文全部训练流程都在此实现。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


# 【make_attn_mask】把token有效性mask和分块边界mask组成[B,N,N]可见性矩阵；前缀彼此可见，动作可读前缀，但前缀不能偷看带噪动作。
# 输入接口：input_mask；mask_ar。
# 返回值可从这里追踪：jnp.logical_and(attn_mask, valid_mask)。
# 内部调用线索：jnp.broadcast_to → jnp.cumsum → jnp.logical_and（含分支中的调用，实际路径由条件决定）。
def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    # 学习提示：每个True开始一个新注意力块；同块token双向可见，后块可看前块。
    cumsum = jnp.cumsum(mask_ar, axis=1)
    # 学习提示：比较key块号与query块号；[B,query,key]中的True表示该query允许读该key。
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


# 【posemb_sincos】把一个标量flow时间编码成不同频率的sin/cos向量；时间不是机器人时钟，而是噪声到动作的插值位置。
# 输入接口：pos:at.Real[at.Array, ' b']；embedding_dim:int；min_period:float；max_period:float。
# 返回类型：at.Float[at.Array, 'b {embedding_dim}']；类型/shape约定需与调用方配套。
# 内部调用线索：ValueError → jnp.linspace → jnp.einsum → jnp.concatenate → jnp.sin（含分支中的调用，实际路径由条件决定）。
@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


# 【Pi0】π0/π0.5的共享flow模型；Pi0Config.pi05决定具体分支。
class Pi0(_model.BaseModel):
    # 【Pi0.__init__】初始化Pi0的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.pi05、self.PaliGemma、self.action_in_proj、self.time_mlp_in、self.time_mlp_out。
    # 输入接口：config:pi0_config.Pi0Config；rngs:nnx.Rngs。
    # 内部调用线索：super().__init__ → _gemma.get_config → nnx_bridge.ToNNX → _gemma.Module → llm.lazy_init（含分支中的调用，实际路径由条件决定）。
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        # 版本分支：此分支是π0.5；对应else为π0。
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    # 【Pi0.embed_prefix】把多相机图像经SigLIP变成patch token，再拼接语言token。返回[B,P,D_vlm]及有效性/分块mask；π0.5的离散状态已经在语言token里。
    # 输入接口：obs:_model.Observation。
    # 返回类型：tuple[at.Float[at.Array, 'b s emb'], at.Bool[at.Array, 'b s'], at.Bool[at.Array, ' s']]；类型/shape约定需与调用方配套。
    # 内部调用线索：self.PaliGemma.img → tokens.append → input_mask.append → einops.repeat → self.PaliGemma.llm（含分支中的调用，实际路径由条件决定）。
    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    # 【Pi0.embed_suffix】把带噪动作[B,H,A]投影到专家宽度。π0额外加连续state token并拼接时间embedding；π0.5不加该state token，时间走AdaRMS条件。
    # 输入接口：obs:_model.Observation；noisy_actions:_model.Actions；timestep:at.Float[at.Array, ' b']。
    # 返回类型：tuple[at.Float[at.Array, 'b s emb'], at.Bool[at.Array, 'b s'], at.Bool[at.Array, ' s'], at.Float[at.Array, 'b emb'] | No…；类型/shape约定需与调用方配套。
    # 内部调用线索：self.state_proj → tokens.append → input_mask.append → jnp.ones → self.action_in_proj（含分支中的调用，实际路径由条件决定）。
    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        # 版本分支：此分支是π0；对应else为π0.5。
        if not self.pi05:
            # add a single state token
            # 学习提示：π0：连续state先线性投影，再加一个token轴，成为[B,1,D_expert]。
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        # 版本分支：此分支是π0.5；对应else为π0。
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            # 学习提示：π0.5把时间信息作为归一化条件传下去；动作token本身不再拼接重复时间向量。
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            # 学习提示：π0这一分支已在动作token中融合时间，不使用AdaRMS时间条件。
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        # 学习提示：整段动作属于同一可互看的块；并非H个动作按时间逐token自回归生成。
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    # 【Pi0.compute_loss】训练：随机采样噪声和时间，把真实动作变为x_t，预测其flow向量场v_t，对目标noise-actions计算MSE。返回每个样本、每个预测时刻的误差。
    # 输入接口：rng:at.KeyArrayLike；observation:_model.Observation；actions:_model.Actions；train:bool。
    # 返回类型：at.Float[at.Array, '*b ah']；类型/shape约定需与调用方配套。
    # 内部调用线索：jax.random.split → _model.preprocess_observation → jax.random.normal → jax.random.beta → self.embed_prefix（含分支中的调用，实际路径由条件决定）。
    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        # 学习提示：[B]→[B,1,1]：让每个样本的同一个噪声时间广播到全部H步和A个动作坐标。
        time_expanded = time[..., None, None]
        # 学习提示：flow训练样本：t=1时是噪声，t=0时是真实动作，中间为线性插值。
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        # 学习提示：目标向量场是dx_t/dt=noise-actions；推理dt为负，所以最终从噪声走向动作。
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        # 学习提示：只按有效token累计位置，padding不会推动有效序列的位置编号。
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        # 学习提示：只取最后H个动作token，再把专家hidden width投影回A维，得到[B,H,A]向量场。
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        # 学习提示：这里只平均动作坐标维，保留[B,H]；训练脚本会进一步取全局均值。
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    # 【Pi0.sample_actions】推理：固定当前观测，先缓存图像/语言前缀K/V，再从高斯噪声向t=0做多次Euler更新。返回[B,H,A]动作块，执行由外部客户端负责。
    # 输入接口：rng:at.KeyArrayLike；observation:_model.Observation；num_steps:int | at.Int[at.Array, '']；noise:at.Float[at.Array, 'b ah ad'] | None。
    # 返回类型：_model.Actions；类型/shape约定需与调用方配套。
    # 内部调用线索：_model.preprocess_observation → jax.random.normal → self.embed_prefix → make_attn_mask → jnp.cumsum（含分支中的调用，实际路径由条件决定）。
    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        # 学习提示：负步长：从t=1积分到t=0。num_steps是去噪/积分次数，不是action_horizon。
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        # 【Pi0.sample_actions.step】一次flow积分：用当前x_t与time构造后缀，读取不变的前缀缓存，预测v_t并更新x_t和time。不是环境里执行了一步机器人动作。
        # 输入接口：carry。
        # 返回值可从这里追踪：(x_t + dt * v_t, time + dt)。
        # 内部调用线索：self.embed_suffix → jnp.broadcast_to → make_attn_mask → einops.repeat → jnp.concatenate（含分支中的调用，实际路径由条件决定）。
        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            # 学习提示：只取最后H个动作token，再把专家hidden width投影回A维，得到[B,H,A]向量场。
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            # 学习提示：Euler积分一步；更新的是整段预测动作，不是机器人此刻已经执行的轨迹。
            return x_t + dt * v_t, time + dt

        # 【Pi0.sample_actions.cond】判断积分是否尚未到达t=0；保留半个步长容差以避免浮点误差多算或少算一步。
        # 输入接口：carry。
        # 返回值可从这里追踪：time >= -dt / 2。
        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
