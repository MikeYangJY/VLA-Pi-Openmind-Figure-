# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：入门数据适配｜把LIBERO图像、状态、指令改成模型统一格式，再把输出裁回7维环境动作。
# 阅读顺序：先看make_libero_example，再看LiberoInputs.__call__，最后看LiberoOutputs。
# 重点边界：缺相机时用占位图和mask；模型补齐到32维，返回环境时只取实际动作维数。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


# 【make_libero_example】本函数位于“入门数据适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.random.rand → np.random.randint 追踪具体实现。
# 返回类型：dict；类型/shape约定需与调用方配套。
# 内部调用线索：np.random.rand → np.random.randint（含分支中的调用，实际路径由条件决定）。
def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
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


# 【LiberoInputs】把LIBERO图像、状态、指令改成模型统一格式，再把输出裁回7维环境动作。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    # 【LiberoInputs.__call__】统一LIBERO的图像布局和键名，补缺失相机并设置mask，保留状态、训练动作和指令。
    # 输入接口：data:dict。
    # 返回类型：dict；类型/shape约定需与调用方配套。
    # 内部调用线索：_parse_image → np.zeros_like（含分支中的调用，实际路径由条件决定）。
    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


# 【LiberoOutputs】把LIBERO图像、状态、指令改成模型统一格式，再把输出裁回7维环境动作。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    # 【LiberoOutputs.__call__】去掉统一动作空间的padding，只返回LIBERO需要的前7维动作。
    # 输入接口：data:dict。
    # 返回类型：dict；类型/shape约定需与调用方配套。
    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][..., :7])}
