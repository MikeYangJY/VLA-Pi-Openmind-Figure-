# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：输入与接口｜定义Observation、Actions以及所有模型共享的创建、加载、损失和采样接口。
# 阅读顺序：先看Observation.from_dict/to_dict与数据示例，再看preprocess_observation；之后看BaseModelConfig.load。
# 重点边界：训练和推理必须使用一致的图像范围、相机键、状态维数及mask；接口声明不等于实现了某种能力。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import abc
from collections.abc import Sequence
import dataclasses
import enum
import logging
import pathlib
from typing import Generic, TypeVar

import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch

from openpi.models_pytorch import pi0_pytorch
from openpi.shared import image_tools
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")

# Type variable for array types (JAX arrays, PyTorch tensors, or numpy arrays)
ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)


# 【ModelType】定义Observation、Actions以及所有模型共享的创建、加载、损失和采样接口。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
class ModelType(enum.Enum):
    """Supported model types."""

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"


# The model always expects these images
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


# This may need change if we release a small model.
IMAGE_RESOLUTION = (224, 224)


# Data format
#
# Data transforms produce the model input as a nested dictionary which is later converted
# into `Obesrvation` and `Actions` objects. See below.
#
# In the dictory form, this data should look like:
# {
#     # Observation data.
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # RGB image in [-1, 1] or [0, 255]
#         ...  # Additional camera views
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b],  # True if image is valid
#         ...  # Masks for additional views
#     },
#     "state": float32[*b, s],  # Low-dimensional robot state
#     "tokenized_prompt": int32[*b, l],  # Optional, tokenized language prompt
#     "tokenized_prompt_mask": bool[*b, l],  # Optional, mask for tokenized prompt
#     "token_ar_mask": int32[*b, l],  # Optional, autoregressive mask for FAST model
#     "token_loss_mask": bool[*b, l],  # Optional, loss mask for FAST model
#
#      # Actions data.
#      "actions": float32[*b ah ad]
# }
# where:
#   *b = batch dimensions
#   h,w = image height/width
#   s = state dimension
#   l = sequence length
#
# 【Observation】一次观测的结构化容器，含相机、mask、状态和文本token。
@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model.

    See `Observation.from_dict` to see the expected dictionary form. This is the format
    that should be produced by the data transforms.
    """

    # Images, in [-1, 1] float32.
    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    # Image masks, with same keys as images.
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    # Low-dimensional robot state.
    state: at.Float[ArrayT, "*b s"]

    # Tokenized prompt.
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    # Tokenized prompt mask.
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # pi0-fast model specific fields.

    # Token auto-regressive mask (for FAST autoregressive model).
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    # Token loss mask (for FAST autoregressive model).
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    # 【Observation.from_dict】把统一数据字典变成Observation对象，并处理图像dtype/数值范围。调用前平台字段应已完成映射。
    # 输入接口：data:at.PyTree[ArrayT]。
    # 返回类型：'Observation[ArrayT]'；类型/shape约定需与调用方配套。
    # 内部调用线索：ValueError → data['image'][key].astype → hasattr → data['image'][key].to(torch.float32).permute → data['image'][key].to（含分支中的调用，实际路径由条件决定）。
    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""
        # Ensure that tokenized_prompt and tokenized_prompt_mask are provided together.
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        # If images are uint8, convert them to [-1, 1] float32.
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
            elif hasattr(data["image"][key], "dtype") and data["image"][key].dtype == torch.uint8:
                data["image"][key] = data["image"][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )

    # 【Observation.to_dict】将Observation还原为统一字典，以便数据记录、PyTree处理或训练日志读取。
    # 返回类型：at.PyTree[ArrayT]；类型/shape约定需与调用方配套。
    # 内部调用线索：dataclasses.asdict → result.pop（含分支中的调用，实际路径由条件决定）。
    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result


# Defines the format of the actions. This field is included as "actions" inside the dictionary
# produced by the data transforms.
Actions = at.Float[ArrayT, "*b ah ad"]


# 【preprocess_observation】JAX端校验相机键，缩放图像，训练时做随机增强，并补图像mask；保留state和语言字段。
# 输入接口：rng:at.KeyArrayLike | None；observation:Observation；train:bool；image_keys:Sequence[str]；image_resolution:tuple[int, int]。
# 返回类型：Observation；类型/shape约定需与调用方配套。
# 内部调用线索：set(image_keys).issubset → set → ValueError → logger.info → image_tools.resize_with_pad（含分支中的调用，实际路径由条件决定）。
def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> Observation:
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in a default image mask (if necessary).
    """

    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)

        if train:
            # Convert from [-1, 1] to [0, 1] for augmax.
            image = image / 2.0 + 0.5

            transforms = []
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            sub_rngs = jax.random.split(rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # Back to [-1, 1].
            image = image * 2.0 - 1.0

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )


# 【BaseModelConfig】定义Observation、Actions以及所有模型共享的创建、加载、损失和采样接口。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models. Specific models should inherit from this class, and implement the `create`
    method to create the corresponding model.
    """

    # Action space dimension.
    # 字段含义：模型统一动作坐标宽度；平台实际动作维数可更小。
    action_dim: int
    # Action sequence length.
    # 字段含义：一次预测未来多少个控制时刻，与一次任务持续多久不同。
    action_horizon: int
    # Tokenized prompt maximum length.
    # 字段含义：文本/离散状态序列的最大token长度，超长会截断。
    max_token_len: int

    # 【BaseModelConfig.model_type】暴露模型类型，让数据变换和上层工厂选择匹配的处理路径。
    # 返回类型：ModelType；类型/shape约定需与调用方配套。
    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """The model type."""

    # 【BaseModelConfig.create】按当前配置构造所需对象或数据管线；创建结果交给上层脚本继续使用。
    # 输入接口：rng:at.KeyArrayLike。
    # 返回类型：'BaseModel'；类型/shape约定需与调用方配套。
    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """Create a new model, initializing parameters."""

    # 【BaseModelConfig.load】先按配置构造参数结构，再检查checkpoint参数路径/形状并装入模型；不是重新学习权重。
    # 输入接口：params:at.Params；remove_extra_params:bool。
    # 返回类型：'BaseModel'；类型/shape约定需与调用方配套。
    # 内部调用线索：nnx.eval_shape → jax.random.key → nnx.split → ocp.transform_utils.intersect_trees → state.to_pure_dict（含分支中的调用，实际路径由条件决定）。
    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """Create a model with the given parameters."""
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    # 【BaseModelConfig.load_pytorch】创建PI0Pytorch并加载safetensors权重；配置与权重结构必须对应。
    # 输入接口：train_config；weight_path:str。
    # 返回值可从这里追踪：model。
    # 内部调用线索：logger.info → pi0_pytorch.PI0Pytorch → safetensors.torch.load_model（含分支中的调用，实际路径由条件决定）。
    def load_pytorch(self, train_config, weight_path: str):
        logger.info(f"train_config: {train_config}")
        model = pi0_pytorch.PI0Pytorch(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model

    # 【BaseModelConfig.inputs_spec】本函数位于“输入与接口”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。
    # 输入接口：batch_size:int。
    # 返回类型：tuple[Observation, Actions]；类型/shape约定需与调用方配套。
    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """Returns the input specification for the model. Values are jax.ShapeDtypeStruct."""

    # 【BaseModelConfig.fake_obs】根据输入规格生成虚拟观测，用于初始化和结构测试，不代表真实机器人数据分布。
    # 输入接口：batch_size:int。
    # 返回类型：Observation；类型/shape约定需与调用方配套。
    # 内部调用线索：self.inputs_spec → jax.tree.map → jnp.ones（含分支中的调用，实际路径由条件决定）。
    def fake_obs(self, batch_size: int = 1) -> Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    # 【BaseModelConfig.fake_act】根据输入规格生成虚拟动作，用于检查损失和张量接口。
    # 输入接口：batch_size:int。
    # 返回类型：Actions；类型/shape约定需与调用方配套。
    # 内部调用线索：self.inputs_spec → jax.tree.map → jnp.ones（含分支中的调用，实际路径由条件决定）。
    def fake_act(self, batch_size: int = 1) -> Actions:
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


# 【BaseModel】定义Observation、Actions以及所有模型共享的创建、加载、损失和采样接口。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """Base class for all model implementations. Specific models should inherit from this class. They should call
    super().__init__() to initialize the shared attributes (action_dim, action_horizon, and max_token_len).
    """

    # 字段含义：模型统一动作坐标宽度；平台实际动作维数可更小。
    action_dim: int
    # 字段含义：一次预测未来多少个控制时刻，与一次任务持续多久不同。
    action_horizon: int
    # 字段含义：文本/离散状态序列的最大token长度，超长会截断。
    max_token_len: int

    # 【BaseModel.compute_loss】本函数位于“输入与接口”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。
    # 输入接口：rng:at.KeyArrayLike；observation:Observation；actions:Actions；train:bool。
    # 返回类型：at.Float[at.Array, '*b ah']；类型/shape约定需与调用方配套。
    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]: ...

    # 【BaseModel.sample_actions】本函数位于“输入与接口”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。
    # 输入接口：rng:at.KeyArrayLike；observation:Observation。
    # 返回类型：Actions；类型/shape约定需与调用方配套。
    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation, **kwargs) -> Actions: ...


# 【restore_params】从Orbax存档恢复模型参数，可指定dtype与分片；恢复权重本身不运行机器人。
# 输入接口：params_path:pathlib.Path | str；restore_type:type[np.ndarray] | type[jax.Array]；dtype:jnp.dtype | None；sharding:jax.sharding.Sharding | None。
# 返回类型：at.Params；类型/shape约定需与调用方配套。
# 内部调用线索：str(params_path).startswith → pathlib.Path(params_path).resolve → pathlib.Path → jax.sharding.Mesh → jax.devices（含分支中的调用，实际路径由条件决定）。
def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint.

    This works with checkpoints saved with `save_state` during openpi training (see `training/checkpoints.py`) as
    well as pre-trained checkpoints released for openpi.

    Args:
        params_path: The local path to the checkpoint directory.
        restore_type: The type to restore the params as. Can be set to `np.ndarray` to load the params as a numpy array.
        dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
        sharding: The sharding to use for the params. If not provided, the params will be replicated across all devices.

    Returns:
        The restored params.
    """
    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]

    # If the params were saved with `save_state` during openpi training, every key path will end with "value", which is
    # added by `nnx.State`. We remove the "value" suffix here and always return what NNX calls a "pure dict".
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)
