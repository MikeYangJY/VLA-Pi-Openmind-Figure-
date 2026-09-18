# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：平台数据适配｜处理ALOHA双臂的关节方向、夹爪标定与图像，把公开硬件坐标连接到训练约定。
# 阅读顺序：先看AlohaInputs/Outputs，再按调用读_decode_state、_encode_actions及夹爪转换。
# 重点边界：正负号与夹爪范围是物理含义，不是随便归一化；输入变换与输出逆变换必须一致。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


# 【make_aloha_example】本函数位于“平台数据适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.ones → np.random.randint 追踪具体实现。
# 返回类型：dict；类型/shape约定需与调用方配套。
# 内部调用线索：np.ones → np.random.randint（含分支中的调用，实际路径由条件决定）。
def make_aloha_example() -> dict:
    """Creates a random input example for the Aloha policy."""
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


# 【AlohaInputs】处理ALOHA双臂的关节方向、夹爪标定与图像，把公开硬件坐标连接到训练约定。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclasses.dataclass(frozen=True)
class AlohaInputs(transforms.DataTransformFn):
    """Inputs for the Aloha policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [14]
    - actions: [action_horizon, 14]
    """

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    # 【AlohaInputs.__call__】解码双臂ALOHA输入，并按配置转换关节方向与夹爪标定，再构造统一模型观测。
    # 输入接口：data:dict。
    # 返回类型：dict；类型/shape约定需与调用方配套。
    # 内部调用线索：_decode_aloha → set → ValueError → extra_image_names.items → np.zeros_like（含分支中的调用，实际路径由条件决定）。
    def __call__(self, data: dict) -> dict:
        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that base image always exists.
        base_image = in_images["cam_high"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Add the extra images.
        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


# 【AlohaOutputs】处理ALOHA双臂的关节方向、夹爪标定与图像，把公开硬件坐标连接到训练约定。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclasses.dataclass(frozen=True)
class AlohaOutputs(transforms.DataTransformFn):
    """Outputs for the Aloha policy."""

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    # 【AlohaOutputs.__call__】把模型动作解码回ALOHA真实平台的坐标/夹爪约定；应与输入适配成对理解。
    # 输入接口：data:dict。
    # 返回类型：dict；类型/shape约定需与调用方配套。
    # 内部调用线索：np.asarray → _encode_actions（含分支中的调用，实际路径由条件决定）。
    def __call__(self, data: dict) -> dict:
        # Only return the first 14 dims.
        actions = np.asarray(data["actions"][:, :14])
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}


# 【_joint_flip_mask】本函数位于“平台数据适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.array 追踪具体实现。
# 返回类型：np.ndarray；类型/shape约定需与调用方配套。
def _joint_flip_mask() -> np.ndarray:
    """Used to convert between aloha and pi joint angles."""
    return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])


# 【_normalize】本函数位于“平台数据适配”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。
# 输入接口：x；min_val；max_val。
# 返回值可从这里追踪：(x - min_val) / (max_val - min_val)。
def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


# 【_unnormalize】本函数位于“平台数据适配”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。
# 输入接口：x；min_val；max_val。
# 返回值可从这里追踪：x * (max_val - min_val) + min_val。
def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


# 【_gripper_to_angular】本函数位于“平台数据适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 _unnormalize → linear_to_radian → _normalize 追踪具体实现。
# 输入接口：value。
# 返回值可从这里追踪：_normalize(value, min_val=0.5476, max_val=1.6296)。
# 内部调用线索：_unnormalize → linear_to_radian → _normalize（含分支中的调用，实际路径由条件决定）。
def _gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    # 【_gripper_to_angular.linear_to_radian】本函数位于“平台数据适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.arcsin → np.clip 追踪具体实现。
    # 输入接口：linear_position；arm_length；horn_radius。
    # 返回值可从这里追踪：np.arcsin(np.clip(value, -1.0, 1.0))。
    # 内部调用线索：np.arcsin → np.clip（含分支中的调用，实际路径由条件决定）。
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(value, -1.0, 1.0))

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # pi0 gripper data is normalized (0, 1) between encoder counts (2405, 3110).
    # There are 4096 total encoder counts and aloha uses a zero of 2048.
    # Converting this to radians means that the normalized inputs are between (0.5476, 1.6296)
    return _normalize(value, min_val=0.5476, max_val=1.6296)


# 【_gripper_from_angular】本函数位于“平台数据适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 _normalize 追踪具体实现。
# 输入接口：value。
# 返回值可从这里追踪：_normalize(value, min_val=-0.6213, max_val=1.491)。
def _gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # We do not scale the output since the trossen model predictions are already in radians.
    # See the comment in _gripper_to_angular for a derivation of the constant
    value = value + 0.5476

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


# 【_gripper_from_angular_inv】本函数位于“平台数据适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 _unnormalize 追踪具体实现。
# 输入接口：value。
# 返回值可从这里追踪：value - 0.5476。
def _gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return value - 0.5476


# 【_decode_aloha】本函数位于“平台数据适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.asarray → _decode_state → convert_image 追踪具体实现。
# 输入接口：data:dict；adapt_to_pi:bool。
# 返回类型：dict；类型/shape约定需与调用方配套。
# 内部调用线索：np.asarray → _decode_state → convert_image → images.items（含分支中的调用，实际路径由条件决定）。
def _decode_aloha(data: dict, *, adapt_to_pi: bool = False) -> dict:
    # state is [left_arm_joint_angles, left_arm_gripper, right_arm_joint_angles, right_arm_gripper]
    # dim sizes: [6, 1, 6, 1]
    state = np.asarray(data["state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi)

    # 【_decode_aloha.convert_image】本函数位于“平台数据适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.asarray → np.issubdtype → (255 * img).astype 追踪具体实现。
    # 输入接口：img。
    # 返回值可从这里追踪：einops.rearrange(img, 'c h w -> h w c')。
    # 内部调用线索：np.asarray → np.issubdtype → (255 * img).astype → einops.rearrange（含分支中的调用，实际路径由条件决定）。
    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data


# 【_decode_state】本函数位于“平台数据适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 _joint_flip_mask → _gripper_to_angular 追踪具体实现。
# 输入接口：state:np.ndarray；adapt_to_pi:bool。
# 返回类型：np.ndarray；类型/shape约定需与调用方配套。
# 内部调用线索：_joint_flip_mask → _gripper_to_angular（含分支中的调用，实际路径由条件决定）。
def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        state = _joint_flip_mask() * state
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        state[[6, 13]] = _gripper_to_angular(state[[6, 13]])
    return state


# 【_encode_actions】本函数位于“平台数据适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 _joint_flip_mask → _gripper_from_angular 追踪具体实现。
# 输入接口：actions:np.ndarray；adapt_to_pi:bool。
# 返回类型：np.ndarray；类型/shape约定需与调用方配套。
# 内部调用线索：_joint_flip_mask → _gripper_from_angular（含分支中的调用，实际路径由条件决定）。
def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular(actions[:, [6, 13]])
    return actions


# 【_encode_actions_inv】本函数位于“平台数据适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 _joint_flip_mask → _gripper_from_angular_inv 追踪具体实现。
# 输入接口：actions:np.ndarray；adapt_to_pi:bool。
# 返回类型：np.ndarray；类型/shape约定需与调用方配套。
# 内部调用线索：_joint_flip_mask → _gripper_from_angular_inv（含分支中的调用，实际路径由条件决定）。
def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular_inv(actions[:, [6, 13]])
    return actions
