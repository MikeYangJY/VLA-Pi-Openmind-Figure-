# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：参数更新｜定义学习率日程与优化器配置，把梯度变成参数更新。
# 阅读顺序：从create_optimizer追到AdamW.create及CosineDecaySchedule；区分学习率、梯度裁剪和权重衰减。
# 重点边界：优化器不决定训练目标；flow loss是在模型中定义的。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import dataclasses
from typing import Protocol, runtime_checkable

import jax.numpy as jnp
import optax

import openpi.shared.array_typing as at


# 【LRScheduleConfig】定义学习率日程与优化器配置，把梯度变成参数更新。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@runtime_checkable
class LRScheduleConfig(Protocol):
    # 【LRScheduleConfig.create】按当前配置构造所需对象或数据管线；创建结果交给上层脚本继续使用。
    # 返回类型：optax.Schedule；类型/shape约定需与调用方配套。
    def create(self) -> optax.Schedule: ...


# 【CosineDecaySchedule】定义学习率日程与优化器配置，把梯度变成参数更新。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclasses.dataclass(frozen=True)
class CosineDecaySchedule(LRScheduleConfig):
    """Cosine decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 2.5e-5
    decay_steps: int = 30_000
    decay_lr: float = 2.5e-6

    # 【CosineDecaySchedule.create】按当前配置构造所需对象或数据管线；创建结果交给上层脚本继续使用。
    # 返回类型：optax.Schedule；类型/shape约定需与调用方配套。
    def create(self) -> optax.Schedule:
        return optax.warmup_cosine_decay_schedule(
            init_value=self.peak_lr / (self.warmup_steps + 1),
            peak_value=self.peak_lr,
            warmup_steps=self.warmup_steps,
            decay_steps=self.decay_steps,
            end_value=self.decay_lr,
        )


# 【RsqrtDecaySchedule】定义学习率日程与优化器配置，把梯度变成参数更新。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclasses.dataclass(frozen=True)
class RsqrtDecaySchedule(LRScheduleConfig):
    """Inverse square root decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 5e-5
    timescale: float = 10_000

    # 【RsqrtDecaySchedule.create】按当前配置构造所需对象或数据管线；创建结果交给上层脚本继续使用。
    # 返回类型：optax.Schedule；类型/shape约定需与调用方配套。
    # 内部调用线索：optax.join_schedules → optax.linear_schedule → jnp.sqrt（含分支中的调用，实际路径由条件决定）。
    def create(self) -> optax.Schedule:
        return optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=self.peak_lr / (self.warmup_steps + 1),
                    end_value=self.peak_lr,
                    transition_steps=self.warmup_steps,
                ),
                lambda step: self.peak_lr / jnp.sqrt((self.timescale + step) / self.timescale),
            ],
            [self.warmup_steps],
        )


# 【OptimizerConfig】定义学习率日程与优化器配置，把梯度变成参数更新。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@runtime_checkable
class OptimizerConfig(Protocol):
    # 【OptimizerConfig.create】按当前配置构造所需对象或数据管线；创建结果交给上层脚本继续使用。
    # 输入接口：lr:optax.ScalarOrSchedule；weight_decay_mask:at.PyTree | None。
    # 返回类型：optax.GradientTransformation；类型/shape约定需与调用方配套。
    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation: ...


# 【AdamW】定义学习率日程与优化器配置，把梯度变成参数更新。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclasses.dataclass(frozen=True)
class AdamW(OptimizerConfig):
    """AdamW optimizer."""

    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    # Changing this to 0 can cause out-of-memory errors for some reason, so we set it to a negligible value.
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.0

    # 【AdamW.create】按当前配置构造所需对象或数据管线；创建结果交给上层脚本继续使用。
    # 输入接口：lr:optax.ScalarOrSchedule；weight_decay_mask:at.PyTree | None。
    # 返回类型：optax.GradientTransformation；类型/shape约定需与调用方配套。
    # 内部调用线索：optax.adamw → optax.chain → optax.clip_by_global_norm（含分支中的调用，实际路径由条件决定）。
    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        tx = optax.adamw(
            lr, b1=self.b1, b2=self.b2, eps=self.eps, weight_decay=self.weight_decay, mask=weight_decay_mask
        )

        return optax.chain(optax.clip_by_global_norm(self.clip_gradient_norm), tx)


# 【SGD】定义学习率日程与优化器配置，把梯度变成参数更新。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclasses.dataclass(frozen=True)
class SGD(OptimizerConfig):
    """SGD optimizer."""

    lr: float = 5e-5
    momentum: float = 0.9
    nesterov: bool = False

    # 【SGD.create】按当前配置构造所需对象或数据管线；创建结果交给上层脚本继续使用。
    # 输入接口：lr:optax.ScalarOrSchedule；weight_decay_mask:at.PyTree | None。
    # 返回类型：optax.GradientTransformation；类型/shape约定需与调用方配套。
    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        assert weight_decay_mask is None, "Weight decay is not supported for SGD"
        return optax.sgd(lr, momentum=self.momentum, nesterov=self.nesterov)


# 【create_optimizer】把学习率日程和优化器配置组装成更新规则，按需加权重衰减mask。
# 输入接口：optimizer:OptimizerConfig；lr_schedule:LRScheduleConfig；weight_decay_mask:at.PyTree | None。
# 返回类型：optax.GradientTransformation；类型/shape约定需与调用方配套。
# 内部调用线索：lr_schedule.create → optimizer.create（含分支中的调用，实际路径由条件决定）。
def create_optimizer(
    optimizer: OptimizerConfig, lr_schedule: LRScheduleConfig, weight_decay_mask: at.PyTree | None = None
) -> optax.GradientTransformation:
    lr = lr_schedule.create()
    return optimizer.create(lr, weight_decay_mask=weight_decay_mask)
