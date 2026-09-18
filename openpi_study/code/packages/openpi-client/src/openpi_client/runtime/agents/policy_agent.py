# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：策略适配器｜把BasePolicy包装成runtime所需Agent。
# 阅读顺序：get_action调用policy.infer；reset转发到策略。
# 重点边界：这层不新增训练，也不自动改变动作空间。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
from typing_extensions import override

from openpi_client import base_policy as _base_policy
from openpi_client.runtime import agent as _agent


# 【PolicyAgent】把BasePolicy包装成runtime所需Agent。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
class PolicyAgent(_agent.Agent):
    """An agent that uses a policy to determine actions."""

    # 【PolicyAgent.__init__】初始化PolicyAgent的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._policy。
    # 输入接口：policy:_base_policy.BasePolicy。
    # 返回类型：None；类型/shape约定需与调用方配套。
    def __init__(self, policy: _base_policy.BasePolicy) -> None:
        self._policy = policy

    # 【PolicyAgent.get_action】本函数位于“策略适配器”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self._policy.infer 追踪具体实现。
    # 输入接口：observation:dict。
    # 返回类型：dict；类型/shape约定需与调用方配套。
    @override
    def get_action(self, observation: dict) -> dict:
        return self._policy.infer(observation)

    # 【PolicyAgent.reset】清理上一episode/调用周期留下的状态，或调用底层环境进行重置。
    # 返回类型：None；类型/shape约定需与调用方配套。
    def reset(self) -> None:
        self._policy.reset()
