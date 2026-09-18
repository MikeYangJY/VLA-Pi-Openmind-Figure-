# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：环境适配｜将真实ALOHA环境包装成客户端runtime的统一Environment接口。
# 阅读顺序：reset准备episode；get_observation生成策略输入；apply_action转交底层环境。
# 重点边界：这是环境接口，不负责神经网络推理。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
from typing import List, Optional  # noqa: UP035

import einops
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

from examples.aloha_real import real_env as _real_env


# 【AlohaRealEnvironment】将真实ALOHA环境包装成客户端runtime的统一Environment接口。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
class AlohaRealEnvironment(_environment.Environment):
    """An environment for an Aloha robot on real hardware."""

    # 【AlohaRealEnvironment.__init__】初始化AlohaRealEnvironment的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._env、self._render_height、self._render_width、self._ts。
    # 输入接口：reset_position:Optional[List[float]]；render_height:int；render_width:int。
    # 返回类型：None；类型/shape约定需与调用方配套。
    def __init__(
        self,
        reset_position: Optional[List[float]] = None,  # noqa: UP006,UP007
        render_height: int = 224,
        render_width: int = 224,
    ) -> None:
        self._env = _real_env.make_real_env(init_node=True, reset_position=reset_position)
        self._render_height = render_height
        self._render_width = render_width

        self._ts = None

    # 【AlohaRealEnvironment.reset】清理上一episode/调用周期留下的状态，或调用底层环境进行重置。
    # 返回类型：None；类型/shape约定需与调用方配套。
    @override
    def reset(self) -> None:
        self._ts = self._env.reset()

    # 【AlohaRealEnvironment.is_episode_complete】检查本环境定义的episode结束条件；结束不一定与成功同义，要看具体条件。
    # 返回类型：bool；类型/shape约定需与调用方配套。
    @override
    def is_episode_complete(self) -> bool:
        return False

    # 【AlohaRealEnvironment.get_observation】读取并组织当前环境的传感器/状态，形成策略下一次决策的输入。
    # 返回类型：dict；类型/shape约定需与调用方配套。
    # 内部调用线索：RuntimeError → obs['images'].keys → image_tools.convert_to_uint8 → image_tools.resize_with_pad → einops.rearrange（含分支中的调用，实际路径由条件决定）。
    @override
    def get_observation(self) -> dict:
        if self._ts is None:
            raise RuntimeError("Timestep is not set. Call reset() first.")

        obs = self._ts.observation
        for k in list(obs["images"].keys()):
            if "_depth" in k:
                del obs["images"][k]

        for cam_name in obs["images"]:
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["images"][cam_name], self._render_height, self._render_width)
            )
            obs["images"][cam_name] = einops.rearrange(img, "h w c -> c h w")

        return {
            "state": obs["qpos"],
            "images": obs["images"],
        }

    # 【AlohaRealEnvironment.apply_action】把策略选择的动作传给底层环境；仿真推进状态，真机可能产生实际运动。
    # 输入接口：action:dict。
    # 返回类型：None；类型/shape约定需与调用方配套。
    @override
    def apply_action(self, action: dict) -> None:
        self._ts = self._env.step(action["actions"])
