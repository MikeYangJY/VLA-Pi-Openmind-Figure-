# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：共享模型底层｜用不同参数分支处理VLM前缀与动作后缀，并通过注意力交换信息。
# 阅读顺序：Config决定宽度；Attention计算Q/K/V；Block加入归一化、残差和MLP；Module逐层运行并管理缓存。
# 重点边界：两个分支可以有不同hidden width，但注意力头维必须兼容；分支交互不等于先生成语言再执行。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
# Copyright 2024 Big Vision Authors.
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

"""Gemma adaptation for Pi, taken from big_vision.

We follow this einsum axis naming convention:
  B: batch
  T: query length
  S: k/v length
  N: num query heads
  K: num k/v heads
  G: num query heads per k/v head
  H: head dim
  D: d_model ("features")
"""

from collections.abc import Sequence
import dataclasses
from typing import Literal, TypeAlias

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding

PALIGEMMA_VOCAB_SIZE = 257_152


# 【Config】用不同参数分支处理VLM前缀与动作后缀，并通过注意力交换信息。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclasses.dataclass
class Config:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    lora_configs: dict[str, lora.LoRAConfig] = dataclasses.field(default_factory=dict)


Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora"]


# 【get_config】按名称返回配置；名称不匹配时进入错误处理，配置对象随后控制模型或实验构建。
# 输入接口：variant:Variant。
# 返回类型：Config；类型/shape约定需与调用方配套。
# 内部调用线索：Config → lora.LoRAConfig → ValueError（含分支中的调用，实际路径由条件决定）。
def get_config(variant: Variant) -> Config:
    """Returns config for specified gemma variant."""
    if variant == "dummy":
        return Config(
            width=64,
            depth=4,
            mlp_dim=128,
            num_heads=8,
            num_kv_heads=1,
            head_dim=16,
        )
    if variant == "gemma_300m":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b_lora":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=16, alpha=16.0), "ffn": lora.LoRAConfig(rank=16, alpha=16.0)},
        )
    if variant == "gemma_300m_lora":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=32, alpha=32.0), "ffn": lora.LoRAConfig(rank=32, alpha=32.0)},
        )
    raise ValueError(f"Unknown variant: {variant}")


# 【RMSNorm】按均方根归一化；可通过条件产生scale/shift/gate。
@at.typecheck
class RMSNorm(nn.Module):
    # 【RMSNorm.__call__】按最后一维均方根归一化。无cond时用常规可学习scale；有cond时生成scale/shift/gate，实现时间条件调制。
    # 输入接口：x；cond。
    # 返回值可从这里追踪：(normed_inputs.astype(dtype), None) / (normed_inputs.astype(dtype), gate)。
    # 内部调用线索：jnp.mean → jnp.square → x.astype → jnp.asarray → jnp.reciprocal（含分支中的调用，实际路径由条件决定）。
    @nn.compact
    def __call__(self, x, cond):
        dtype = x.dtype  # original dtype, could be half-precision
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)  # compute variance in float32
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))  # compute normalization in float32
        if cond is None:
            # regular RMSNorm
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
            normed_inputs = normed_inputs * (
                1 + scale
            )  # scale by learned parameter in float32 (matches Flax implementation)
            return normed_inputs.astype(dtype), None  # return in original dtype

        # adaptive RMSNorm
        modulation = nn.Dense(x.shape[-1] * 3, kernel_init=nn.initializers.zeros, dtype=dtype)(cond)
        # 学习提示：条件向量生成三组参数：缩放、平移，以及残差分支贡献的gate。
        scale, shift, gate = jnp.split(modulation[:, None, :], 3, axis=-1)
        # 学习提示：AdaRMS把flow时间变成特征调制，告诉专家当前噪声水平。
        normed_inputs = normed_inputs * (1 + scale) + shift  # scale and shift in float32
        return normed_inputs.astype(dtype), gate


# 【Embedder】用不同参数分支处理VLM前缀与动作后缀，并通过注意力交换信息。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@at.typecheck
class Embedder(nn.Module):
    """Embedder module."""

    vocab_size: int
    embed_dim: int

    # 【Embedder.setup】建立该模块用到的参数和子层，之后前向调用重复使用它们。
    # 内部调用线索：self.param → nn.initializers.normal（含分支中的调用，实际路径由条件决定）。
    def setup(self):
        self.input_embedding_table = self.param(
            "input_embedding",
            nn.initializers.normal(),
            (self.vocab_size, self.embed_dim),
        )

    # 【Embedder.encode】用token ID索引embedding矩阵，并按隐藏维度缩放嵌入。
    # 输入接口：x。
    # 返回值可从这里追踪：x。
    # 内部调用线索：jnp.sqrt(self.embed_dim).astype → jnp.sqrt（含分支中的调用，实际路径由条件决定）。
    def encode(self, x):
        x = self.input_embedding_table[(x,)]
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
        return x

    # 【Embedder.decode】用embedding矩阵的转置把隐藏表示投影回词表分数；动作flow头不是在这里输出关节值。
    # 输入接口：x。
    # 返回值可从这里追踪：jnp.dot(x, self.input_embedding_table.T)。
    def decode(self, x):
        return jnp.dot(x, self.input_embedding_table.T)


# 【Attention】将每个token的信息按相关性加权汇集，Q/K/V分别用于查询、匹配和内容聚合。
@at.typecheck
class Attention(nn.Module):
    """Attention module."""

    configs: Sequence[Config]

    # 【Attention.__call__】对各参数分支分别生成Q/K/V，统一注意力头尺寸，在token维上交互，再拆回分支并投影；可读取已有前缀缓存。
    # 输入接口：xs；positions；attn_mask；kv_cache。
    # 返回值可从这里追踪：(out, (k, v))。
    # 内部调用线索：all → next → lora.Einsum → _name → nn.initializers.lecun_normal（含分支中的调用，实际路径由条件决定）。
    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache):
        # all experts must share the same head dim, num heads, and num kv heads for self-attention to work
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)

        dtype = next(x.dtype for x in xs if x is not None)  # original dtype, could be half-precision

        qkvs = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue
            if config.num_kv_heads == config.num_heads:
                qkv_einsum = lora.Einsum(
                    shape=(3, config.num_heads, config.width, config.head_dim),
                    name=_name("qkv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))
            else:
                q_einsum = lora.Einsum(
                    shape=(config.num_heads, config.width, config.head_dim),
                    name=_name("q_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                    lora_config=config.lora_configs.get("attn"),
                )
                q = q_einsum("BTD,NDH->BTNH", x)
                kv_einsum = lora.Einsum(
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),
                    name=_name("kv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                k, v = kv_einsum("BSD,2KDH->2BSKH", x)
                qkvs.append((q, k, v))

        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

        q = _apply_rope(q, positions=positions)
        q *= self.configs[0].head_dim ** -0.5

        k = _apply_rope(k, positions=positions)

        # should still be half-precision here (if input was half-precision)
        assert q.dtype == k.dtype == v.dtype == dtype

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = jnp.concatenate([cache_k, k], axis=1)
            v = jnp.concatenate([cache_v, v], axis=1)

        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)

        if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(
                f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
            )

        # big_neg = jnp.finfo(logits.dtype).min
        big_neg = -2.3819763e38  # See gemma/modules.py
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)

        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)

        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        out = []
        start = 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                end = start + x.shape[1]
                out_einsum = lora.Einsum(
                    shape=(config.num_heads, config.head_dim, config.width),
                    name=_name("attn_vec_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    lora_config=config.lora_configs.get("attn"),
                )
                out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
                start = end
            else:
                out.append(None)

        return out, (k, v)


# 【FeedForward】逐token的非线性特征变换，不在token间直接交换信息。
@at.typecheck
class FeedForward(nn.Module):
    """Feed forward module."""

    features: int
    hidden_dim: int

    # 【FeedForward.__call__】执行逐token的非线性特征变换，不在token间直接交换信息。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。
    # 输入接口：x。
    # 返回值可从这里追踪：outputs。
    # 内部调用线索：self.param('gating_einsum', nn.initializers.lecun_normal(in_axis=-2, o… → self.param → nn.initializers.lecun_normal → jnp.dot → nn.gelu（含分支中的调用，实际路径由条件决定）。
    @nn.compact
    def __call__(self, x):
        dtype = x.dtype  # original dtype, could be half-precision
        w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        ).astype(dtype)
        ff_gate = jnp.dot(x, w_gating[0])
        gate_value = nn.gelu(ff_gate)

        ff1 = jnp.dot(x, w_gating[1])
        activations = gate_value * ff1

        w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        ).astype(dtype)
        outputs = jnp.dot(activations, w_linear)
        assert outputs.dtype == dtype
        return outputs


# 【Block】归一化、注意力、残差与前馈组成的一层Transformer。
@at.typecheck
class Block(nn.Module):
    """Transformer block."""

    configs: tuple[Config, ...]

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()

    # 【Block.__call__】执行归一化、注意力、残差与前馈组成的一层Transformer。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。
    # 输入接口：xs；kv_cache；positions；attn_mask；adarms_cond；deterministic。
    # 返回值可从这里追踪：(xs, kv_cache)。
    # 内部调用线索：sharding.activation_sharding_constraint → nn.Dropout → Attention → RMSNorm(name=_name('pre_attention_norm', i)) → RMSNorm（含分支中的调用，实际路径由条件决定）。
    @nn.compact
    def __call__(self, xs, kv_cache, positions, attn_mask, adarms_cond, deterministic=True):  # noqa: FBT002
        xs = sharding.activation_sharding_constraint(xs)
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x

        attn = Attention(configs=self.configs, name="attn")

        pre_attn = []
        gates = []
        for i, x in enumerate(xs):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_attention_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
            pre_attn.append(x)
            gates.append(gate if x is not None else None)

        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache)
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        post_attn = sharding.activation_sharding_constraint(post_attn)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        out = []
        gates = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_ffw_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
                x = lora.FeedForward(  # noqa: PLW2901
                    features=config.width,
                    hidden_dim=config.mlp_dim,
                    name=_name("mlp", i),
                    lora_config=config.lora_configs.get("ffn"),
                )(x)
            out.append(x)
            gates.append(gate if x is not None else None)

        out = sharding.activation_sharding_constraint(out)
        out = jax.tree.map(lambda x: drop(x, deterministic), out)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, out, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        return xs, kv_cache


KVCache: TypeAlias = tuple[at.Float[at.Array, "l b _t _k _h"], at.Float[at.Array, "l b _t _v _h"]]


# 【Module】用不同参数分支处理VLM前缀与动作后缀，并通过注意力交换信息。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@at.typecheck
class Module(nn.Module):
    """Transformer model, supporting a mixture of different weights for different tokens."""

    configs: Sequence[Config]  # list of configs, one for each expert
    embed_dtype: str

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()  # Every float is dropped independently.
    adarms: bool = False

    # 【Module.setup】建立该模块用到的参数和子层，之后前向调用重复使用它们。
    # 内部调用线索：all → Embedder → nn.remat → nn.scan(block_cls, variable_axes={'params': 0}, split_rngs={'params': … → nn.scan（含分支中的调用，实际路径由条件决定）。
    def setup(self):
        # all experts must have the same depth
        assert all(config.depth == self.configs[0].depth for config in self.configs)

        self.embedder = Embedder(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            embed_dim=self.configs[0].width,  # embedder for first expert only
            name="embedder",
        )
        block_cls = nn.remat(
            Block,
            prevent_cse=False,
            static_argnums=(5,),  # 0=self, 6=deterministic
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(
                0,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
            ),  # 0=kv_cache, 1=positions, 2=mask, 3=adarms_cond, 4=deterministic
            length=self.configs[0].depth,
        )(
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
        )
        self.final_norms = [RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))]

    # 【Module.embed】本函数位于“共享模型底层”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.embedder.encode(tokens).astype → self.embedder.encode 追踪具体实现。
    # 输入接口：tokens:at.Int[at.Array, 'b t']。
    # 返回类型：at.Float[at.Array, 'b t d']；类型/shape约定需与调用方配套。
    # 内部调用线索：self.embedder.encode(tokens).astype → self.embedder.encode（含分支中的调用，实际路径由条件决定）。
    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    # 【Module.__call__】执行用不同参数分支处理VLM前缀与动作后缀，并通过注意力交换信息。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。
    # 输入接口：embedded:Sequence[at.Float[at.Array, 'b _t _d'] | None]；positions:at.Int[at.Array, 'b t']；mask:at.Bool[at.Array, 'b t s']；adarms_cond:Sequence[at.Float[at.Array, 'b _d'] | None] | None；kv_cache:KVCache | None；deterministic:bool。
    # 返回类型：tuple[Sequence[at.Float[at.Array, 'b _t _d'] | None], KVCache]；类型/shape约定需与调用方配套。
    # 内部调用线索：jax.tree.map → e.astype → jnp.asarray → self.layers → all（含分支中的调用，实际路径由条件决定）。
    @at.typecheck
    def __call__(
        self,
        # list of token arrays, one for each expert, or None if that expert should not be run
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],
        mask: at.Bool[at.Array, "b t s"],
        adarms_cond: Sequence[at.Float[at.Array, "b _d"] | None] | None = None,
        *,
        kv_cache: KVCache | None = None,
        deterministic: bool = True,
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache]:
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        embedded, kv_cache = self.layers(embedded, kv_cache, positions, mask, adarms_cond, deterministic)

        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)

        return [
            f(e, a)[0] if e is not None else e for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ], kv_cache

    # 【Module.init】本函数位于“共享模型底层”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.embed → jnp.zeros → self 追踪具体实现。
    # 输入接口：use_adarms:Sequence[bool]。
    # 内部调用线索：self.embed → jnp.zeros → self（含分支中的调用，实际路径由条件决定）。
    def init(self, use_adarms: Sequence[bool]):
        """Convenience method for initializing all parameters, necessary due to the quirks of linen."""
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            adarms_cond=[jnp.zeros((1, c.width)) if u else None for u, c in zip(use_adarms, self.configs, strict=True)],
        )


# 【_apply_rope】对Q/K向量分量施加随token位置变化的旋转，让注意力感知相对位置；不改变序列中token的物理排列。
# 输入接口：x；positions；max_wavelength。
# 返回值可从这里追踪：res.astype(x.dtype)。
# 内部调用线索：jnp.arange → jnp.sin → jnp.cos → jnp.split → jnp.concatenate（含分支中的调用，实际路径由条件决定）。
def _apply_rope(x, *, positions, max_wavelength=10_000):
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
    freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2, dtype=jnp.float32)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    assert radians.dtype == jnp.float32
    # radians.shape = [...,L,1,d=D/2]
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    x1, x2 = jnp.split(x, 2, axis=-1)
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    assert res.dtype == jnp.float32
    # The original bigvision impl allows RoPE to upcast to float32. It is then immediately downcast again to the cache
    # dtype when in inference mode (but not in training mode). I don't think any of this was intentional. Based on the
    # original DeepMind impl, as well as the widely-used transformers impl, it is ok to always downcast back to bfloat16
    # here.
    return res.astype(x.dtype)


# 【_name】本函数位于“共享模型底层”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。
# 输入接口：name；i。
# 返回值可从这里追踪：name / f'{name}_{i}'。
def _name(name, i):
    # we name layers like this because we want the first expert's weights to have no suffix (e.g., "attn"), so that they
    # can be loaded seamlessly from the existing PaliGemma checkpoint. subsequent experts will have a suffix (e.g.,
    # "attn_1") and their weights will be initialized from scratch. in practice, we only use two experts -- PaliGemma,
    # and the action expert.
    if i == 0:
        return name
    return f"{name}_{i}"


# 【_gated_residual】把子层输出加回输入；存在gate时先调节子层贡献，这是AdaRMS相关残差控制。
# 输入接口：x；y；gate。
# 返回值可从这里追踪：None / x + y。
def _gated_residual(x, y, gate):
    assert (x is None) == (y is None)
    if x is None:
        return None
    if gate is None:
        return x + y
    return x + y * gate
