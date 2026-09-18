# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：环境接口｜规定环境必须支持重置、观测、执行与episode结束检查。
# 阅读顺序：把接口与aloha_sim/env.py的实现逐一对应。
# 重点边界：抽象接口本身不含机器人驱动。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import abc


# 【Environment】规定环境必须支持重置、观测、执行与episode结束检查。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
class Environment(abc.ABC):
    """An Environment represents the robot and the environment it inhabits.

    The primary contract of environments is that they can be queried for observations
    about their state, and have actions applied to them to change that state.
    """

    # 【Environment.reset】清理上一episode/调用周期留下的状态，或调用底层环境进行重置。
    # 返回类型：None；类型/shape约定需与调用方配套。
    @abc.abstractmethod
    def reset(self) -> None:
        """Reset the environment to its initial state.

        This will be called once before starting each episode.
        """

    # 【Environment.is_episode_complete】检查本环境定义的episode结束条件；结束不一定与成功同义，要看具体条件。
    # 返回类型：bool；类型/shape约定需与调用方配套。
    @abc.abstractmethod
    def is_episode_complete(self) -> bool:
        """Allow the environment to signal that the episode is complete.

        This will be called after each step. It should return `True` if the episode is
        complete (either successfully or unsuccessfully), and `False` otherwise.
        """

    # 【Environment.get_observation】读取并组织当前环境的传感器/状态，形成策略下一次决策的输入。
    # 返回类型：dict；类型/shape约定需与调用方配套。
    @abc.abstractmethod
    def get_observation(self) -> dict:
        """Query the environment for the current state."""

    # 【Environment.apply_action】把策略选择的动作传给底层环境；仿真推进状态，真机可能产生实际运动。
    # 输入接口：action:dict。
    # 返回类型：None；类型/shape约定需与调用方配套。
    @abc.abstractmethod
    def apply_action(self, action: dict) -> None:
        """Take an action in the environment."""
