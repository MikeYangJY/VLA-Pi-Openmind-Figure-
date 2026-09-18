# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：核心配置｜决定模型大小、动作维数、预测长度和pi05开关，并构造模型。
# 阅读顺序：从Pi0Config字段到__post_init__，再到inputs_spec和create；LoRA相关freeze_filter最后读。
# 重点边界：默认action_dim=32是统一模型宽度，不代表每台机器人都有32个关节；action_horizon=50是预测步数。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


# 【Pi0Config】决定模型大小、动作维数、预测长度和pi05开关，并构造模型。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    # 字段含义：模型计算/参数精度配置，与动作物理单位无关。
    dtype: str = "bfloat16"
    # 字段含义：VLM语言主干尺寸/LoRA变体。
    paligemma_variant: _gemma.Variant = "gemma_2b"
    # 字段含义：连续动作专家尺寸/LoRA变体。
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    # 字段含义：模型统一动作坐标宽度；平台实际动作维数可更小。
    action_dim: int = 32
    # 字段含义：一次预测未来多少个控制时刻，与一次任务持续多久不同。
    action_horizon: int = 50
    # 字段含义：文本/离散状态序列的最大token长度，超长会截断。
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    # 字段含义：选择π0.5分支；False表示π0。
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    # 字段含义：是否把state离散后编码进语言前缀。
    discrete_state_input: bool = None  # type: ignore

    # 字段含义：PyTorch推理编译选项；初读算法时可忽略性能调优细节。
    pytorch_compile_mode: str | None = "max-autotune"

    # 【Pi0Config.__post_init__】根据pi05开关补默认值：π0通常48个文本token，π0.5通常200；离散state默认跟随pi05。还校验PyTorch编译模式。
    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    # 【Pi0Config.model_type】暴露模型类型，让数据变换和上层工厂选择匹配的处理路径。
    # 返回类型：_model.ModelType；类型/shape约定需与调用方配套。
    @property
    @override
    def model_type(self) -> _model.ModelType:
        # 版本分支：此分支是π0.5；对应else为π0。
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    # 【Pi0Config.create】按当前配置构造所需对象或数据管线；创建结果交给上层脚本继续使用。
    # 输入接口：rng:at.KeyArrayLike。
    # 返回类型：'Pi0'；类型/shape约定需与调用方配套。
    # 内部调用线索：Pi0 → nnx.Rngs（含分支中的调用，实际路径由条件决定）。
    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    # 【Pi0Config.inputs_spec】构造只含shape/dtype的信息，规定相机、state、token和动作的维度；这是接口模板，不会采集真机数据。
    # 输入接口：batch_size:int。
    # 返回类型：tuple[_model.Observation, _model.Actions]；类型/shape约定需与调用方配套。
    # 内部调用线索：jax.ShapeDtypeStruct → at.disable_typechecking → _model.Observation（含分支中的调用，实际路径由条件决定）。
    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    # 【Pi0Config.get_freeze_filter】根据LoRA分支选出不更新的参数；没有LoRA时返回不冻结。训练脚本再从中推导可训练参数。
    # 返回类型：nnx.filterlib.Filter；类型/shape约定需与调用方配套。
    # 内部调用线索：nnx_utils.PathRegex → filters.append → nnx.Not → nnx.All（含分支中的调用，实际路径由条件决定）。
    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
