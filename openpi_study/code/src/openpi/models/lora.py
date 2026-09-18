# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：可选微调｜为线性/Einsum和前馈层添加低秩可训练分支，以降低微调参数量。
# 阅读顺序：LoRAConfig决定rank与缩放；setup创建低秩矩阵；__call__将低秩更新叠加到原运算。
# 重点边界：LoRA是参数更新方式，不是KI；还要配合freeze_filter控制哪些参数实际训练。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import math
import re

import flax.linen as nn
import flax.struct as struct
import jax.numpy as jnp

import openpi.shared.array_typing as at


# 【LoRAConfig】为线性/Einsum和前馈层添加低秩可训练分支，以降低微调参数量。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@struct.dataclass
class LoRAConfig:
    """Configuration for LoRA."""

    # LoRA rank.
    rank: int
    # LoRA scaling factor.
    alpha: float = 1.0
    # Initialization function for LoRA parameters.
    init_fn: nn.initializers.Initializer = nn.initializers.normal(stddev=0.01)
    # Enable rank-stabilized LoRA: https://arxiv.org/pdf/2312.03732
    rslora: bool = False
    # Axes in the weight to apply LoRA to. Should typically be the last two axes.
    axes: tuple[int, int] = (-2, -1)
    # Axis label which is used by LoRA in einsum equations. Must not be present in the original equation.
    label: str = "L"

    # 【LoRAConfig.scaling_value】本函数位于“可选微调”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 math.sqrt 追踪具体实现。
    # 返回类型：float；类型/shape约定需与调用方配套。
    @property
    def scaling_value(self) -> float:
        return self.alpha / math.sqrt(self.rank) if self.rslora else self.alpha / self.rank


# 【Einsum】为线性/Einsum和前馈层添加低秩可训练分支，以降低微调参数量。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
class Einsum(nn.Module):
    """Einsum with LoRA support. Can be used as a drop-in replacement for the Gemma Einsum."""

    # Shape of the weight.
    shape: tuple[int, ...]
    # Initialization function for the weight.
    init_fn: nn.initializers.Initializer = nn.initializers.zeros
    # If not None, apply LoRA to the weight.
    lora_config: LoRAConfig | None = None

    # 【Einsum.setup】建立该模块用到的参数和子层，之后前向调用重复使用它们。
    def setup(self):
        self.w = self.param("w", self.init_fn, self.shape)

        if config := self.lora_config:
            # Setup LoRA parameters.
            shape_a, shape_b = list(self.shape), list(self.shape)
            shape_a[config.axes[1]] = config.rank
            shape_b[config.axes[0]] = config.rank
            self.w_a = self.param("lora_a", config.init_fn, shape_a)
            self.w_b = self.param("lora_b", config.init_fn, shape_b)

    # 【Einsum.__call__】执行为线性/Einsum和前馈层添加低秩可训练分支，以降低微调参数量。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。
    # 输入接口：eqn:str；x。
    # 返回值可从这里追踪：result。
    # 内部调用线索：jnp.einsum → self.w.astype → self._make_lora_eqns → self.w_a.astype → self.w_b.astype（含分支中的调用，实际路径由条件决定）。
    @nn.compact
    def __call__(self, eqn: str, x):
        dtype = x.dtype  # original dtype, could be half-precision
        result = jnp.einsum(eqn, x, self.w.astype(dtype))

        if config := self.lora_config:
            eqn_a, eqn_b = self._make_lora_eqns(eqn)
            lora = jnp.einsum(eqn_a, x, self.w_a.astype(dtype))
            lora = jnp.einsum(eqn_b, lora, self.w_b.astype(dtype))
            result = result + lora * config.scaling_value

        return result

    # 【Einsum._make_lora_eqns】本函数位于“可选微调”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 ValueError → re.match → m.groups 追踪具体实现。
    # 输入接口：eqn:str。
    # 返回类型：tuple[str, str]；类型/shape约定需与调用方配套。
    # 内部调用线索：ValueError → re.match → m.groups → rhs.replace → out.replace（含分支中的调用，实际路径由条件决定）。
    def _make_lora_eqns(self, eqn: str) -> tuple[str, str]:
        if "L" in eqn:
            raise ValueError(f"L already in eqn: {eqn}")
        if not (m := re.match("(.*),(.*)->(.*)", eqn)):
            raise ValueError(f"Unsupported einsum eqn: {eqn}")
        lhs, rhs, out = m.groups()

        assert self.lora_config is not None
        a_label, b_label = (rhs[x] for x in self.lora_config.axes)
        label = self.lora_config.label

        a_rhs = rhs.replace(b_label, label)
        a_out = out.replace(b_label, label)
        eqn_a = f"{lhs},{a_rhs}->{a_out}"

        b_rhs = rhs.replace(a_label, label)
        eqn_b = f"{a_out},{b_rhs}->{out}"

        return eqn_a, eqn_b


# 【FeedForward】逐token的非线性特征变换，不在token间直接交换信息。
class FeedForward(nn.Module):
    """Feed forward module."""

    features: int
    hidden_dim: int
    # If not None, apply LoRA to the weight.
    lora_config: LoRAConfig | None = None

    # 【FeedForward.setup】建立该模块用到的参数和子层，之后前向调用重复使用它们。
    # 内部调用线索：self.param → nn.initializers.lecun_normal（含分支中的调用，实际路径由条件决定）。
    def setup(self):
        self.w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        )
        self.w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        )
        self.w_gating_lora = None
        self.w_linear_lora = None
        if self.lora_config:
            # Setup LoRA parameters.
            # TODO: follow up with a simplified init_fn api.
            self.w_gating_lora = (
                self.param("gating_einsum_lora_a", self.lora_config.init_fn, (2, self.features, self.lora_config.rank)),
                self.param(
                    "gating_einsum_lora_b", self.lora_config.init_fn, (2, self.lora_config.rank, self.hidden_dim)
                ),
            )
            self.w_linear_lora = (
                self.param("linear_lora_a", self.lora_config.init_fn, (self.hidden_dim, self.lora_config.rank)),
                self.param("linear_lora_b", self.lora_config.init_fn, (self.lora_config.rank, self.features)),
            )

    # 【FeedForward.__call__】执行逐token的非线性特征变换，不在token间直接交换信息。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。
    # 输入接口：x。
    # 返回值可从这里追踪：outputs。
    # 内部调用线索：self._dot → nn.gelu（含分支中的调用，实际路径由条件决定）。
    @nn.compact
    def __call__(self, x):
        dtype = x.dtype  # original dtype, could be half-precision
        ff_gate = self._dot(
            x,
            self.w_gating[0],
            None if self.w_gating_lora is None else (self.w_gating_lora[0][0], self.w_gating_lora[1][0]),
        )
        gate_value = nn.gelu(ff_gate)

        ff1 = self._dot(
            x,
            self.w_gating[1],
            None if self.w_gating_lora is None else (self.w_gating_lora[0][1], self.w_gating_lora[1][1]),
        )
        activations = gate_value * ff1

        outputs = self._dot(activations, self.w_linear, self.w_linear_lora)
        assert outputs.dtype == dtype
        return outputs

    # 【FeedForward._dot】本函数位于“可选微调”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 jnp.dot → w.astype → lora_weights[0].astype 追踪具体实现。
    # 输入接口：x:at.Array；w:at.Array；lora_weights:tuple[at.Array, at.Array] | None。
    # 返回类型：at.Array；类型/shape约定需与调用方配套。
    # 内部调用线索：jnp.dot → w.astype → lora_weights[0].astype → lora_weights[1].astype（含分支中的调用，实际路径由条件决定）。
    def _dot(self, x: at.Array, w: at.Array, lora_weights: tuple[at.Array, at.Array] | None) -> at.Array:
        base = jnp.dot(x, w.astype(x.dtype))
        if lora_weights is None:
            return base
        return base + jnp.dot(jnp.dot(x, lora_weights[0].astype(x.dtype)), lora_weights[1].astype(x.dtype))
