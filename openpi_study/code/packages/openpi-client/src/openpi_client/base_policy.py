# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：客户端接口｜定义infer和reset，统一远程策略与本地包装器的使用方式。
# 阅读顺序：先看接口，再看WebsocketClientPolicy与ActionChunkBroker如何实现。
# 重点边界：接口返回动作，不承担环境执行。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import abc
from typing import Dict


# 【BasePolicy】定义infer和reset，统一远程策略与本地包装器的使用方式。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
class BasePolicy(abc.ABC):
    # 【BasePolicy.infer】本函数位于“客户端接口”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。
    # 输入接口：obs:Dict。
    # 返回类型：Dict；类型/shape约定需与调用方配套。
    @abc.abstractmethod
    def infer(self, obs: Dict) -> Dict:
        """Infer actions from observations."""

    # 【BasePolicy.reset】清理上一episode/调用周期留下的状态，或调用底层环境进行重置。
    # 返回类型：None；类型/shape约定需与调用方配套。
    def reset(self) -> None:
        """Reset the policy to its initial state."""
        pass
