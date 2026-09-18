# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：推理主入口｜把原始观测、输入变换、模型采样和输出变换封装成统一infer接口。
# 阅读顺序：infer：输入复制→变换→加batch轴→构造Observation→sample_actions→去batch轴→反变换；Recorder额外保存输入输出。
# 重点边界：返回的是一段动作；Policy本身不会驱动真实电机。记录的infer_ms也不自动包含完整网络/机器人闭环延迟。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


# 【Policy】可调用的推理管线：变换、模型、逆变换。
class Policy(BasePolicy):
    # 【Policy.__init__】初始化Policy的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._model、self._input_transform、self._output_transform、self._sample_kwargs、self._metadata。
    # 输入接口：model:_model.BaseModel；rng:at.KeyArrayLike | None；transforms:Sequence[_transforms.DataTransformFn]；output_transforms:Sequence[_transforms.DataTransformFn]；sample_kwargs:dict[str, Any] | None；metadata:dict[str, Any] | None；pytorch_device:str；is_pytorch:bool。
    # 内部调用线索：_transforms.compose → self._model.to → self._model.eval → nnx_utils.module_jit → jax.random.key（含分支中的调用，实际路径由条件决定）。
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    # 【Policy.infer】输入单条观测，做输入变换并加batch轴，调用sample_actions，再去batch轴和做输出逆变换；同时附模型调用计时。
    # 输入接口：obs:dict；noise:np.ndarray | None。
    # 返回类型：dict；类型/shape约定需与调用方配套。
    # 内部调用线索：jax.tree.map → self._input_transform → jnp.asarray → jax.random.split → torch.from_numpy(np.array(x)).to（含分支中的调用，实际路径由条件决定）。
    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            # 学习提示：JAX随机数需要显式拆key；更新保存的key，避免每次推理重复同一噪声。
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        # 学习提示：平台字段已经经过变换；这里才进入模型统一Observation契约。
        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        # 学习提示：反归一化、动作坐标还原及平台维度裁剪发生在模型采样之后。
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    # 【Policy.metadata】本函数位于“推理主入口”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。
    # 返回类型：dict[str, Any]；类型/shape约定需与调用方配套。
    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


# 【PolicyRecorder】不改变策略选择，额外把输入输出保存供调试。
class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    # 【PolicyRecorder.__init__】初始化PolicyRecorder的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._policy、self._record_dir、self._record_step。
    # 输入接口：policy:_base_policy.BasePolicy；record_dir:str。
    # 内部调用线索：logging.info → pathlib.Path → self._record_dir.mkdir（含分支中的调用，实际路径由条件决定）。
    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    # 【PolicyRecorder.infer】照常调用内部策略，再保存这一时刻的输入输出；不改变动作选择。
    # 输入接口：obs:dict。
    # 返回类型：dict；类型/shape约定需与调用方配套。
    # 内部调用线索：self._policy.infer → flax.traverse_util.flatten_dict → np.save → np.asarray（含分支中的调用，实际路径由条件决定）。
    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
