# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：模型初始化｜把预训练checkpoint或PaliGemma权重映射进当前模型参数树。
# 阅读顺序：先看WeightLoader接口，再看CheckpointWeightLoader与PaliGemmaWeightLoader的加载/合并规则。
# 重点边界：只匹配部分权重时其余参数需要初始化；参数树路径和形状必须正确。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


# 【WeightLoader】把预训练checkpoint或PaliGemma权重映射进当前模型参数树。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@runtime_checkable
class WeightLoader(Protocol):
    # 【WeightLoader.load】读取或合并外部资源/参数；检查路径、结构与调用方期待是否一致。
    # 输入接口：params:at.Params。
    # 返回类型：at.Params；类型/shape约定需与调用方配套。
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


# 【NoOpWeightLoader】把预训练checkpoint或PaliGemma权重映射进当前模型参数树。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    # 【NoOpWeightLoader.load】读取或合并外部资源/参数；检查路径、结构与调用方期待是否一致。
    # 输入接口：params:at.Params。
    # 返回类型：at.Params；类型/shape约定需与调用方配套。
    def load(self, params: at.Params) -> at.Params:
        return params


# 【CheckpointWeightLoader】把预训练checkpoint或PaliGemma权重映射进当前模型参数树。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    # 【CheckpointWeightLoader.load】读取或合并外部资源/参数；检查路径、结构与调用方期待是否一致。
    # 输入接口：params:at.Params。
    # 返回类型：at.Params；类型/shape约定需与调用方配套。
    # 内部调用线索：_model.restore_params → download.maybe_download → _merge_params（含分支中的调用，实际路径由条件决定）。
    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


# 【PaliGemmaWeightLoader】把预训练checkpoint或PaliGemma权重映射进当前模型参数树。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    # 【PaliGemmaWeightLoader.load】读取或合并外部资源/参数；检查路径、结构与调用方期待是否一致。
    # 输入接口：params:at.Params。
    # 返回类型：at.Params；类型/shape约定需与调用方配套。
    # 内部调用线索：download.maybe_download → path.open → np.load → flax.traverse_util.unflatten_dict → _merge_params（含分支中的调用，实际路径由条件决定）。
    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


# 【_merge_params】本函数位于“模型初始化”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 flax.traverse_util.flatten_dict → flat_loaded.items → v.astype 追踪具体实现。
# 输入接口：loaded_params:at.Params；params:at.Params；missing_regex:str。
# 返回类型：at.Params；类型/shape约定需与调用方配套。
# 内部调用线索：flax.traverse_util.flatten_dict → flat_loaded.items → v.astype → flat_loaded.clear → re.compile（含分支中的调用，实际路径由条件决定）。
def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
