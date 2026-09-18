# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：视觉编码器｜把每个相机图像切成patch并编码为视觉token，为VLM提供前缀条件。
# 阅读顺序：从Module与decode_variant找到尺寸，再读_Module.__call__的patch/位置编码，最后看Encoder和池化选项。
# 重点边界：π0使用pool_type=none保留空间token；不要把整张图只想成一个向量。
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

"""A refactored and simplified ViT adoptation for Pi, taken from big_vision."""

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import openpi.training.sharding as sharding


# 【posemb_sincos_2d】本函数位于“视觉编码器”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 jnp.arange → jnp.einsum → y.flatten 追踪具体实现。
# 输入接口：h；w；width；temperature；dtype。
# 返回值可从这里追踪：jnp.asarray(pe, dtype)[None, :, :]。
# 内部调用线索：jnp.arange → jnp.einsum → y.flatten → x.flatten → jnp.concatenate（含分支中的调用，实际路径由条件决定）。
def posemb_sincos_2d(h, w, width, temperature=10_000.0, dtype=jnp.float32):
    """Follows the MoCo v3 logic."""
    y, x = jnp.mgrid[:h, :w]

    assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
    omega = jnp.arange(width // 4) / (width // 4 - 1)
    omega = 1.0 / (temperature**omega)
    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)
    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
    return jnp.asarray(pe, dtype)[None, :, :]


# 【get_posemb】本函数位于“视觉编码器”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.param → nn.initializers.normal → np.sqrt 追踪具体实现。
# 输入接口：typ；seqshape；width；name；dtype。
# 返回值可从这里追踪：self.param(name, nn.initializers.normal(stddev=1 / np.sqrt(width)), (1, np.prod(seqshape), width), dtype) / posemb_sincos_2d(*seqshape, width, dtype=dtype)。
# 内部调用线索：self.param → nn.initializers.normal → np.sqrt → np.prod → posemb_sincos_2d（含分支中的调用，实际路径由条件决定）。
def get_posemb(self, typ, seqshape, width, name, dtype=jnp.float32):
    if typ == "learn":
        return self.param(
            name,
            nn.initializers.normal(stddev=1 / np.sqrt(width)),
            (1, np.prod(seqshape), width),
            dtype,
        )
    if typ == "sincos2d":
        return posemb_sincos_2d(*seqshape, width, dtype=dtype)
    raise ValueError(f"Unknown posemb type: {typ}")


# 【MlpBlock】视觉Transformer中的逐token前馈网络。
class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int | None = None  # Defaults to 4x input dim
    dropout: float = 0.0
    dtype_mm: str = "float32"

    # 【MlpBlock.__call__】执行视觉Transformer中的逐token前馈网络。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。
    # 输入接口：x；deterministic。
    # 返回值可从这里追踪：nn.Dense(d, dtype=self.dtype_mm, **inits)(x)。
    # 内部调用线索：nn.initializers.xavier_uniform → nn.initializers.normal → nn.Dense(self.mlp_dim or 4 * d, dtype=self.dtype_mm, **inits) → nn.Dense → nn.gelu（含分支中的调用，实际路径由条件决定）。
    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        """Applies Transformer MlpBlock module."""
        inits = {
            "kernel_init": nn.initializers.xavier_uniform(),
            "bias_init": nn.initializers.normal(stddev=1e-6),
        }

        _, _, d = x.shape  # n,l,d
        x = nn.Dense(self.mlp_dim or 4 * d, dtype=self.dtype_mm, **inits)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic)
        return nn.Dense(d, dtype=self.dtype_mm, **inits)(x)


# 【Encoder1DBlock】一次视觉token注意力与前馈更新。
class Encoder1DBlock(nn.Module):
    """Single transformer encoder block (MHSA + MLP)."""

    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0
    dtype_mm: str = "float32"

    # 【Encoder1DBlock.__call__】执行一次视觉token注意力与前馈更新。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。
    # 输入接口：x；deterministic。
    # 返回值可从这里追踪：(x, out)。
    # 内部调用线索：sharding.activation_sharding_constraint → nn.LayerNorm(dtype=self.dtype_mm) → nn.LayerNorm → nn.MultiHeadDotProductAttention(num_heads=self.num_heads, kernel_init=… → nn.MultiHeadDotProductAttention（含分支中的调用，实际路径由条件决定）。
    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        out = {}
        x = sharding.activation_sharding_constraint(x)
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = out["sa"] = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype=self.dtype_mm,
        )(y, y)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+sa"] = x + y

        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = out["mlp"] = MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
        )(y, deterministic)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+mlp"] = x + y
        x = sharding.activation_sharding_constraint(x)
        return x, out


# 【Encoder】按顺序堆叠视觉编码层。
class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    depth: int
    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0
    scan: bool = False
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"

    # 【Encoder.__call__】执行按顺序堆叠视觉编码层。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。
    # 输入接口：x；deterministic。
    # 返回值可从这里追踪：(nn.LayerNorm(name='encoder_norm', dtype=self.dtype_mm)(x), out)。
    # 内部调用线索：nn.remat → getattr → nn.scan(block, variable_axes={'params': 0}, split_rngs={'params': True… → nn.scan → jax.tree.map（含分支中的调用，实际路径由条件决定）。
    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        out = {}

        if self.scan:
            block = nn.remat(
                Encoder1DBlock,
                prevent_cse=False,
                static_argnums=(2,),  # 0=self, 2=deterministic
                policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
            )
            x, scan_out = nn.scan(
                block,
                variable_axes={"params": 0},
                split_rngs={"params": True, "dropout": True},
                in_axes=nn.broadcast,
                length=self.depth,
            )(
                name="encoderblock",
                dtype_mm=self.dtype_mm,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
            )(x, deterministic)
            for lyr in range(self.depth):
                out[f"block{lyr:02d}"] = jax.tree.map(lambda o, lyr=lyr: o[lyr], scan_out)
        else:
            # Input Encoder
            for lyr in range(self.depth):
                block_cur = Encoder1DBlock(
                    name=f"encoderblock_{lyr}",
                    dtype_mm=self.dtype_mm,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                )
                x, out[f"block{lyr:02d}"] = block_cur(x, deterministic)
            out["pre_ln"] = x  # Alias for last block, but without the number in it.

        return nn.LayerNorm(name="encoder_norm", dtype=self.dtype_mm)(x), out


# 【MAPHead】把每个相机图像切成patch并编码为视觉token，为VLM提供前缀条件。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
class MAPHead(nn.Module):
    """Multihead Attention Pooling."""

    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dtype_mm: str = "float32"

    # 【MAPHead.__call__】执行把每个相机图像切成patch并编码为视觉token，为VLM提供前缀条件。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。
    # 输入接口：x。
    # 返回值可从这里追踪：x[:, 0]。
    # 内部调用线索：self.param → nn.initializers.xavier_uniform → jnp.tile → nn.MultiHeadDotProductAttention(num_heads=self.num_heads, dtype=self.d… → nn.MultiHeadDotProductAttention（含分支中的调用，实际路径由条件决定）。
    @nn.compact
    def __call__(self, x):
        n, _, d = x.shape  # n,l,d
        probe = self.param("probe", nn.initializers.xavier_uniform(), (1, 1, d), x.dtype)
        probe = jnp.tile(probe, [n, 1, 1])

        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=self.dtype_mm,
            kernel_init=nn.initializers.xavier_uniform(),
        )(probe, x)

        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        x = x + MlpBlock(mlp_dim=self.mlp_dim, dtype=self.dtype_mm)(y)
        return x[:, 0]


# 【_Module】把每个相机图像切成patch并编码为视觉token，为VLM提供前缀条件。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
class _Module(nn.Module):
    """ViT model."""

    num_classes: int | None = None
    patch_size: Sequence[int] = (16, 16)
    width: int = 768
    depth: int = 12
    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    posemb: str = "learn"  # Can also be "sincos2d"
    rep_size: int | bool = False
    dropout: float = 0.0
    pool_type: str = "gap"  # Can also be "map" or "tok"
    head_zeroinit: bool = True
    scan: bool = False
    # or "dots_with_no_batch_dims_saveable" for more speed (memory costly)
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"

    # 【_Module.__call__】执行把每个相机图像切成patch并编码为视觉token，为VLM提供前缀条件。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。
    # 输入接口：image；train。
    # 返回值可从这里追踪：(x, out)。
    # 内部调用线索：jnp.asarray → nn.Conv(self.width, self.patch_size, strides=self.patch_size, padding=… → nn.Conv → jnp.reshape → get_posemb（含分支中的调用，实际路径由条件决定）。
    @nn.compact
    def __call__(self, image, *, train=False):
        out = {}

        # Kevin edit: do patch extraction and posemb in float32,
        # because I feel like it's a bit safer.
        image = jnp.asarray(image, jnp.float32)

        # Patch extraction
        x = out["stem"] = nn.Conv(
            self.width,
            self.patch_size,
            strides=self.patch_size,
            padding="VALID",
            name="embedding",
            dtype=jnp.float32,
        )(image)

        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])

        # Add posemb before adding extra token.
        x = out["with_posemb"] = x + get_posemb(self, self.posemb, (h, w), c, "pos_embedding", jnp.float32)

        if self.pool_type == "tok":
            cls = self.param("cls", nn.initializers.zeros, (1, 1, c), x.dtype)
            x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)

        n, _, c = x.shape  # n,l,d
        x = nn.Dropout(rate=self.dropout)(x, not train)

        # Kevin edit: now cast back to dtype_mm (potentially half precision)
        x = x.astype(self.dtype_mm)

        x, out["encoder"] = Encoder(
            depth=self.depth,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            scan=self.scan,
            remat_policy=self.remat_policy,
            dtype_mm=self.dtype_mm,
            name="Transformer",
        )(x, deterministic=not train)
        encoded = out["encoded"] = x

        if self.pool_type == "map":
            x = out["head_input"] = MAPHead(
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dtype=self.dtype_mm,
            )(x)
        elif self.pool_type == "gap":
            x = out["head_input"] = jnp.mean(x, axis=1)
        elif self.pool_type == "0":
            x = out["head_input"] = x[:, 0]
        elif self.pool_type == "tok":
            x = out["head_input"] = x[:, 0]
            encoded = encoded[:, 1:]
        elif self.pool_type == "none":
            pass
        else:
            raise ValueError(f"Unknown pool type: '{self.pool_type}'")

        x_2d = jnp.reshape(encoded, [n, h, w, -1])

        if self.rep_size:
            rep_size = self.width if self.rep_size is True else self.rep_size
            hid = nn.Dense(rep_size, dtype=self.dtype_mm, name="pre_logits")
            # NOTE: In the past we did not include tanh in pre_logits.
            # For few-shot, it should not matter much, as it whitens anyways.
            x_2d = nn.tanh(hid(x_2d))
            x = nn.tanh(hid(x))

        out["pre_logits_2d"] = x_2d
        out["pre_logits"] = x

        if self.num_classes:
            kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
            head = nn.Dense(self.num_classes, dtype=self.dtype_mm, name="head", **kw)
            x_2d = out["logits_2d"] = head(x_2d)
            x = out["logits"] = head(x)

        return x, out


# 【Module】本函数位于“视觉编码器”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 _Module → decode_variant 追踪具体实现。
# 输入接口：num_classes；variant。
# 返回值可从这里追踪：_Module(num_classes, **{**decode_variant(variant), **kw})。
# 内部调用线索：_Module → decode_variant（含分支中的调用，实际路径由条件决定）。
def Module(num_classes=None, *, variant=None, **kw):  # pylint: disable=invalid-name  # noqa: N802
    """Factory function, because linen really don't like what I'm doing!"""
    return _Module(num_classes, **{**decode_variant(variant), **kw})


# 【decode_variant】本函数位于“视觉编码器”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 variant.split 追踪具体实现。
# 输入接口：variant。
# 返回值可从这里追踪：{} / {'width': {'mu': 32, 'Ti': 192, 'S': 384, 'M': 512, 'B': 768, 'L': 1024, 'So400m': 1152, 'H': 1280, 'g': 1408,…。
def decode_variant(variant):
    """Converts a string like "B" or "B/32" into a params dict."""
    if variant is None:
        return {}

    v, patch = variant, {}
    if "/" in variant:
        v, patch = variant.split("/")
        patch = {"patch_size": (int(patch), int(patch))}

    return {
        # pylint:disable=line-too-long
        # Reference: Table 2 of https://arxiv.org/abs/2106.04560.
        "width": {
            "mu": 32,
            "Ti": 192,
            "S": 384,
            "M": 512,
            "B": 768,
            "L": 1024,
            "So400m": 1152,
            "H": 1280,
            "g": 1408,
            "g-opt": 1536,
            "G": 1664,
            "G-opt": 1536,
            "e": 1792,
        }[v],
        "depth": {
            "mu": 1,
            "Ti": 12,
            "S": 12,
            "M": 12,
            "B": 12,
            "L": 24,
            "So400m": 27,
            "H": 32,
            "g": 40,
            "g-opt": 40,
            "G": 48,
            "G-opt": 48,
            "e": 56,
        }[v],
        "mlp_dim": {
            "mu": 128,
            "Ti": 768,
            "S": 1536,
            "M": 2048,
            "B": 3072,
            "L": 4096,
            "So400m": 4304,
            "H": 5120,
            "g": 6144,
            "g-opt": 6144,
            "G": 8192,
            "G-opt": 8192,
            "e": 15360,
        }[v],
        "num_heads": {
            "mu": 2,
            "Ti": 3,
            "S": 6,
            "M": 8,
            "B": 12,
            "L": 16,
            "So400m": 16,
            "H": 16,
            "g": 16,
            "g-opt": 16,
            "G": 16,
            "G-opt": 16,
            "e": 16,
        }[v],
        # pylint:enable=line-too-long
        **patch,
    }
