# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：数据主线｜一组可组合的小变换，处理字段映射、图像、归一化、动作相对化、token与补维。
# 阅读顺序：沿Group→Repack→Normalize→TokenizePrompt→PadStatesAndActions读；输出按反向语义还原。FAST专属类可跳过。
# 重点边界：变换顺序影响物理含义；动作时间维与动作坐标维不能混淆。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


# 【DataTransformFn】一组可组合的小变换，处理字段映射、图像、归一化、动作相对化、token与补维。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@runtime_checkable
class DataTransformFn(Protocol):
    # 【DataTransformFn.__call__】执行一组可组合的小变换，处理字段映射、图像、归一化、动作相对化、token与补维。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。
    # 输入接口：data:DataDict。
    # 返回类型：DataDict；类型/shape约定需与调用方配套。
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


# 【Group】输入/输出变换的有序集合。
@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    # 【Group.push】输入变换追加到末尾，输出逆变换放到开头；这样嵌套的数据表示可以按相反顺序还原。
    # 输入接口：inputs:Sequence[DataTransformFn]；outputs:Sequence[DataTransformFn]。
    # 返回类型：'Group'；类型/shape约定需与调用方配套。
    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


# 【CompositeTransform】一组可组合的小变换，处理字段映射、图像、归一化、动作相对化、token与补维。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    # 【CompositeTransform.__call__】依次调用每个transform，并把上一步字典交给下一步；顺序就是数据管线的语义。
    # 输入接口：data:DataDict。
    # 返回类型：DataDict；类型/shape约定需与调用方配套。
    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


# 【compose】本函数位于“数据主线”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 CompositeTransform 追踪具体实现。
# 输入接口：transforms:Sequence[DataTransformFn]。
# 返回类型：DataTransformFn；类型/shape约定需与调用方配套。
def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


# 【RepackTransform】按给定键路径映射重组样本。
@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    # 【RepackTransform.__call__】先展平原字典，再按structure映射取值并组回目标结构；主要重命名/重排，不做数值学习。
    # 输入接口：data:DataDict。
    # 返回类型：DataDict；类型/shape约定需与调用方配套。
    # 内部调用线索：flatten_dict → jax.tree.map（含分支中的调用，实际路径由条件决定）。
    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


# 【InjectDefaultPrompt】一组可组合的小变换，处理字段映射、图像、归一化、动作相对化、token与补维。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    # 【InjectDefaultPrompt.__call__】只有输入没有prompt且配置给了默认指令时才补入，不覆盖已有用户指令。
    # 输入接口：data:DataDict。
    # 返回类型：DataDict；类型/shape约定需与调用方配套。
    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


# 【Normalize】把状态和动作映射到统一数值尺度。
@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    # 字段含义：状态/动作的数值统计，需与训练变换对应。
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    # 【Normalize.__post_init__】本函数位于“数据主线”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 _assert_quantile_stats 追踪具体实现。
    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    # 【Normalize.__call__】执行把状态和动作映射到统一数值尺度。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。
    # 输入接口：data:DataDict。
    # 返回类型：DataDict；类型/shape约定需与调用方配套。
    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    # 【Normalize._normalize】用训练数据的mean/std做z-score；1e-6避免除零，不是可训练参数。
    # 输入接口：x；stats:NormStats。
    # 返回值可从这里追踪：(x - mean) / (std + 1e-06)。
    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        # 学习提示：按每个坐标单独缩放，避免量纲差异主导损失。
        return (x - mean) / (std + 1e-6)

    # 【Normalize._normalize_quantile】用q01/q99把数据线性映射到约[-1,1]；超出分位数的数据仍可能超出这个区间。
    # 输入接口：x；stats:NormStats。
    # 返回值可从这里追踪：(x - q01) / (q99 - q01 + 1e-06) * 2.0 - 1.0。
    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        # 学习提示：用1%/99%分位数映射到[-1,1]附近，降低极端值对尺度估计的影响。
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


# 【Unnormalize】把输出恢复到真实动作尺度。
@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    # 字段含义：状态/动作的数值统计，需与训练变换对应。
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    # 【Unnormalize.__post_init__】本函数位于“数据主线”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 _assert_quantile_stats 追踪具体实现。
    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    # 【Unnormalize.__call__】执行把输出恢复到真实动作尺度。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。
    # 输入接口：data:DataDict。
    # 返回类型：DataDict；类型/shape约定需与调用方配套。
    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    # 【Unnormalize._unnormalize】逆转z-score，把模型输出恢复到训练动作的数值单位；补齐维数采用中性统计。
    # 输入接口：x；stats:NormStats。
    # 返回值可从这里追踪：x * (std + 1e-06) + mean。
    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    # 【Unnormalize._unnormalize_quantile】逆转分位数映射；真实维度按q01/q99恢复，额外padding维度单独保留。
    # 输入接口：x；stats:NormStats。
    # 返回值可从这里追踪：np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-06) + q01, x[..., dim:]], axis=-1) / (x + 1.0) / 2.0 * (q99 - q01 + 1e-06) + q01。
    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


# 【ResizeImages】一组可组合的小变换，处理字段映射、图像、归一化、动作相对化、token与补维。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    # 【ResizeImages.__call__】逐相机等比例缩放并补边到统一输入分辨率。
    # 输入接口：data:DataDict。
    # 返回类型：DataDict；类型/shape约定需与调用方配套。
    # 内部调用线索：image_tools.resize_with_pad → data['image'].items（含分支中的调用，实际路径由条件决定）。
    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


# 【SubsampleActions】一组可组合的小变换，处理字段映射、图像、归一化、动作相对化、token与补维。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    # 【SubsampleActions.__call__】沿时间维按stride取未来动作，不是减少动作坐标数量。
    # 输入接口：data:DataDict。
    # 返回类型：DataDict；类型/shape约定需与调用方配套。
    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


# 【DeltaActions】把指定坐标的绝对动作变成相对当前状态的增量。
@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    # 【DeltaActions.__call__】只在mask选中的动作维上减去当前state，得到相对目标；夹爪等未选维度不变。
    # 输入接口：data:DataDict。
    # 返回类型：DataDict；类型/shape约定需与调用方配套。
    # 内部调用线索：np.asarray → np.expand_dims → np.where（含分支中的调用，实际路径由条件决定）。
    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


# 【AbsoluteActions】把增量动作还原成对应绝对目标。
@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    # 【AbsoluteActions.__call__】将选中维度的相对动作加回当前state，恢复环境期望的绝对目标。
    # 输入接口：data:DataDict。
    # 返回类型：DataDict；类型/shape约定需与调用方配套。
    # 内部调用线索：np.asarray → np.expand_dims → np.where（含分支中的调用，实际路径由条件决定）。
    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


# 【TokenizePrompt】一组可组合的小变换，处理字段映射、图像、归一化、动作相对化、token与补维。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    # 字段含义：是否把state离散后编码进语言前缀。
    discrete_state_input: bool = False

    # 【TokenizePrompt.__call__】取prompt并按配置决定是否把state也交给PaligemmaTokenizer，添加tokenized_prompt与mask。
    # 输入接口：data:DataDict。
    # 返回类型：DataDict；类型/shape约定需与调用方配套。
    # 内部调用线索：data.pop → ValueError → data.get → isinstance → prompt.item（含分支中的调用，实际路径由条件决定）。
    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


# 【TokenizeFASTInputs】一组可组合的小变换，处理字段映射、图像、归一化、动作相对化、token与补维。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    # 【TokenizeFASTInputs.__call__】执行一组可组合的小变换，处理字段映射、图像、归一化、动作相对化、token与补维。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。
    # 输入接口：data:DataDict。
    # 返回类型：DataDict；类型/shape约定需与调用方配套。
    # 内部调用线索：data.pop → ValueError → isinstance → prompt.item → data.get（含分支中的调用，实际路径由条件决定）。
    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


# 【ExtractFASTActions】一组可组合的小变换，处理字段映射、图像、归一化、动作相对化、token与补维。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    # 字段含义：一次预测未来多少个控制时刻，与一次任务持续多久不同。
    action_horizon: int
    # 字段含义：模型统一动作坐标宽度；平台实际动作维数可更小。
    action_dim: int

    # 【ExtractFASTActions.__call__】执行一组可组合的小变换，处理字段映射、图像、归一化、动作相对化、token与补维。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。
    # 输入接口：data:DataDict。
    # 返回类型：DataDict；类型/shape约定需与调用方配套。
    # 内部调用线索：data.pop → self.tokenizer.extract_actions → tokens.astype（含分支中的调用，实际路径由条件决定）。
    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


# 【PromptFromLeRobotTask】一组可组合的小变换，处理字段映射、图像、归一化、动作相对化、token与补维。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    # 【PromptFromLeRobotTask.__call__】把数据集任务索引映射成语言指令，让每个训练样本带prompt。
    # 输入接口：data:DataDict。
    # 返回类型：DataDict；类型/shape约定需与调用方配套。
    # 内部调用线索：ValueError → self.tasks.get（含分支中的调用，实际路径由条件决定）。
    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


# 【PadStatesAndActions】一组可组合的小变换，处理字段映射、图像、归一化、动作相对化、token与补维。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    # 【PadStatesAndActions.__call__】把最后一维补到模型action_dim；训练含actions时一起补，推理只有state也可以处理。
    # 输入接口：data:DataDict。
    # 返回类型：DataDict；类型/shape约定需与调用方配套。
    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


# 【flatten_dict】本函数位于“数据主线”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 traverse_util.flatten_dict 追踪具体实现。
# 输入接口：tree:at.PyTree。
# 返回类型：dict；类型/shape约定需与调用方配套。
def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


# 【unflatten_dict】本函数位于“数据主线”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 traverse_util.unflatten_dict 追踪具体实现。
# 输入接口：tree:dict。
# 返回类型：at.PyTree；类型/shape约定需与调用方配套。
def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


# 【transform_dict】本函数位于“数据主线”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 flatten_dict → re.compile → patterns.items 追踪具体实现。
# 输入接口：patterns:Mapping[str, str | None]；tree:at.PyTree。
# 返回类型：at.PyTree；类型/shape约定需与调用方配套。
# 内部调用线索：flatten_dict → re.compile → patterns.items → compiled.items → pattern.fullmatch（含分支中的调用，实际路径由条件决定）。
def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


# 【apply_tree】按参考统计/结构逐叶应用函数，可严格检查缺失字段；用于批量归一化嵌套数据。
# 输入接口：tree:at.PyTree[T]；selector:at.PyTree[S]；fn:Callable[[T, S], T]；strict:bool。
# 返回类型：at.PyTree[T]；类型/shape约定需与调用方配套。
# 内部调用线索：flatten_dict → ValueError → unflatten_dict → transform → tree.items（含分支中的调用，实际路径由条件决定）。
def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    # 【apply_tree.transform】本函数位于“数据主线”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 fn 追踪具体实现。
    # 输入接口：k:str；v:T。
    # 返回类型：T；类型/shape约定需与调用方配套。
    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


# 【pad_to_dim】沿指定维度补常数，使输入适配统一长度；shape相容不代表不同机器人的动作语义相同。
# 输入接口：x:np.ndarray；target_dim:int；axis:int；value:float。
# 返回类型：np.ndarray；类型/shape约定需与调用方配套。
def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


# 【make_bool_mask】把正数段转为True、负数段转为False，简洁指定哪些动作维需要相对化等处理。
# 返回类型：tuple[bool, ...]；类型/shape约定需与调用方配套。
def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


# 【_assert_quantile_stats】本函数位于“数据主线”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 flatten_dict(norm_stats).items → flatten_dict → ValueError 追踪具体实现。
# 输入接口：norm_stats:at.PyTree[NormStats]。
# 返回类型：None；类型/shape约定需与调用方配套。
# 内部调用线索：flatten_dict(norm_stats).items → flatten_dict → ValueError（含分支中的调用，实际路径由条件决定）。
def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
