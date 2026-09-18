# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：仿真环境适配｜将ALOHA仿真观测、动作和任务结束条件接到统一runtime。
# 阅读顺序：reset→_convert_observation→get_observation→apply_action。
# 重点边界：模拟环境的成功标志与真实机器人验收指标不同。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import gym_aloha  # noqa: F401
import gymnasium
import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override


# 【AlohaSimEnvironment】将ALOHA仿真观测、动作和任务结束条件接到统一runtime。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
class AlohaSimEnvironment(_environment.Environment):
    """An environment for an Aloha robot in simulation."""

    # 【AlohaSimEnvironment.__init__】初始化AlohaSimEnvironment的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._rng、self._gym、self._last_obs、self._done、self._episode_reward。
    # 输入接口：task:str；obs_type:str；seed:int。
    # 返回类型：None；类型/shape约定需与调用方配套。
    # 内部调用线索：np.random.seed → np.random.default_rng → gymnasium.make（含分支中的调用，实际路径由条件决定）。
    def __init__(self, task: str, obs_type: str = "pixels_agent_pos", seed: int = 0) -> None:
        np.random.seed(seed)
        self._rng = np.random.default_rng(seed)

        self._gym = gymnasium.make(task, obs_type=obs_type)

        self._last_obs = None
        self._done = True
        self._episode_reward = 0.0

    # 【AlohaSimEnvironment.reset】清理上一episode/调用周期留下的状态，或调用底层环境进行重置。
    # 返回类型：None；类型/shape约定需与调用方配套。
    # 内部调用线索：self._gym.reset → self._rng.integers → self._convert_observation（含分支中的调用，实际路径由条件决定）。
    @override
    def reset(self) -> None:
        gym_obs, _ = self._gym.reset(seed=int(self._rng.integers(2**32 - 1)))
        self._last_obs = self._convert_observation(gym_obs)  # type: ignore
        self._done = False
        self._episode_reward = 0.0

    # 【AlohaSimEnvironment.is_episode_complete】检查本环境定义的episode结束条件；结束不一定与成功同义，要看具体条件。
    # 返回类型：bool；类型/shape约定需与调用方配套。
    @override
    def is_episode_complete(self) -> bool:
        return self._done

    # 【AlohaSimEnvironment.get_observation】读取并组织当前环境的传感器/状态，形成策略下一次决策的输入。
    # 返回类型：dict；类型/shape约定需与调用方配套。
    @override
    def get_observation(self) -> dict:
        if self._last_obs is None:
            raise RuntimeError("Observation is not set. Call reset() first.")

        return self._last_obs  # type: ignore

    # 【AlohaSimEnvironment.apply_action】把策略选择的动作传给底层环境；仿真推进状态，真机可能产生实际运动。
    # 输入接口：action:dict。
    # 返回类型：None；类型/shape约定需与调用方配套。
    # 内部调用线索：self._gym.step → self._convert_observation → max（含分支中的调用，实际路径由条件决定）。
    @override
    def apply_action(self, action: dict) -> None:
        gym_obs, reward, terminated, truncated, info = self._gym.step(action["actions"])
        self._last_obs = self._convert_observation(gym_obs)  # type: ignore
        self._done = terminated or truncated
        self._episode_reward = max(self._episode_reward, reward)

    # 【AlohaSimEnvironment._convert_observation】本函数位于“仿真环境适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 image_tools.convert_to_uint8 → image_tools.resize_with_pad → np.transpose 追踪具体实现。
    # 输入接口：gym_obs:dict。
    # 返回类型：dict；类型/shape约定需与调用方配套。
    # 内部调用线索：image_tools.convert_to_uint8 → image_tools.resize_with_pad → np.transpose（含分支中的调用，实际路径由条件决定）。
    def _convert_observation(self, gym_obs: dict) -> dict:
        img = gym_obs["pixels"]["top"]
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
        # Convert axis order from [H, W, C] --> [C, H, W]
        img = np.transpose(img, (2, 0, 1))

        return {
            "state": gym_obs["agent_pos"],
            "images": {"cam_high": img},
        }
