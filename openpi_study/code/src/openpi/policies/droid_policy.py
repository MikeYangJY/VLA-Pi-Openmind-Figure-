# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：平台数据适配｜映射DROID相机、关节状态与夹爪信息，连接模型统一输入输出格式。
# 阅读顺序：从示例字典到DroidInputs，检查图像布局、state拼接和输出动作截取。
# 重点边界：不同平台字段名和夹爪约定不同，不能直接把LIBERO字典传给DROID配置。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


# 【make_droid_example】本函数位于“平台数据适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.random.randint → np.random.rand 追踪具体实现。
# 返回类型：dict；类型/shape约定需与调用方配套。
# 内部调用线索：np.random.randint → np.random.rand（含分支中的调用，实际路径由条件决定）。
def make_droid_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


# 【_parse_image】将浮点图像转换到uint8，并在需要时从CHW转为HWC；先确认源图像数值范围。
# 输入接口：image。
# 返回类型：np.ndarray；类型/shape约定需与调用方配套。
# 内部调用线索：np.asarray → np.issubdtype → (255 * image).astype → einops.rearrange（含分支中的调用，实际路径由条件决定）。
def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


# 【DroidInputs】映射DROID相机、关节状态与夹爪信息，连接模型统一输入输出格式。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclasses.dataclass(frozen=True)
class DroidInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType

    # 【DroidInputs.__call__】按DROID字段取外部/腕部图像、关节和夹爪状态，组成模型统一输入。
    # 输入接口：data:dict。
    # 返回类型：dict；类型/shape约定需与调用方配套。
    # 内部调用线索：np.asarray → np.concatenate → _parse_image → np.zeros_like → ValueError（含分支中的调用，实际路径由条件决定）。
    def __call__(self, data: dict) -> dict:
        gripper_pos = np.asarray(data["observation/gripper_position"])
        if gripper_pos.ndim == 0:
            # Ensure gripper position is a 1D array, not a scalar, so we can concatenate with joint positions
            gripper_pos = gripper_pos[np.newaxis]
        state = np.concatenate([data["observation/joint_position"], gripper_pos])

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        base_image = _parse_image(data["observation/exterior_image_1_left"])
        wrist_image = _parse_image(data["observation/wrist_image_left"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


# 【DroidOutputs】映射DROID相机、关节状态与夹爪信息，连接模型统一输入输出格式。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclasses.dataclass(frozen=True)
class DroidOutputs(transforms.DataTransformFn):
    # 【DroidOutputs.__call__】执行映射DROID相机、关节状态与夹爪信息，连接模型统一输入输出格式。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。
    # 输入接口：data:dict。
    # 返回类型：dict；类型/shape约定需与调用方配套。
    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        return {"actions": np.asarray(data["actions"][..., :8])}
