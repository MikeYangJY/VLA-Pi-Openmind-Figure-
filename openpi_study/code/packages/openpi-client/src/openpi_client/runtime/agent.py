# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：运行接口｜定义根据观测取得动作及重置的Agent抽象。
# 阅读顺序：Runtime向Agent要动作；具体PolicyAgent把调用转给policy。
# 重点边界：Agent不是语言agent推理模型，而是机器人runtime里的接口名称。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import abc


# 【Agent】定义根据观测取得动作及重置的Agent抽象。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
class Agent(abc.ABC):
    """An Agent is the thing with agency, i.e. the entity that makes decisions.

    Agents receive observations about the state of the world, and return actions
    to take in response.
    """

    # 【Agent.get_action】本函数位于“运行接口”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。
    # 输入接口：observation:dict。
    # 返回类型：dict；类型/shape约定需与调用方配套。
    @abc.abstractmethod
    def get_action(self, observation: dict) -> dict:
        """Query the agent for the next action."""

    # 【Agent.reset】清理上一episode/调用周期留下的状态，或调用底层环境进行重置。
    # 返回类型：None；类型/shape约定需与调用方配套。
    @abc.abstractmethod
    def reset(self) -> None:
        """Reset the agent to its initial state."""
