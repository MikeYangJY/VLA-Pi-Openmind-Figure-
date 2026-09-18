# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：运行事件接口｜定义episode开始、每步和结束的回调，供视频与日志工具订阅。
# 阅读顺序：先看三个回调，再看VideoSaver/VideoDisplay的实现。
# 重点边界：旁路记录器可以观察控制过程，但不是策略本体。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import abc


# 【Subscriber】定义episode开始、每步和结束的回调，供视频与日志工具订阅。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
class Subscriber(abc.ABC):
    """Subscribes to events in the runtime.

    Subscribers can be used to save data, visualize, etc.
    """

    # 【Subscriber.on_episode_start】处理runtime广播的事件，更新本订阅者的显示/记录状态。
    # 返回类型：None；类型/shape约定需与调用方配套。
    @abc.abstractmethod
    def on_episode_start(self) -> None:
        """Called when an episode starts."""

    # 【Subscriber.on_step】处理runtime广播的事件，更新本订阅者的显示/记录状态。
    # 输入接口：observation:dict；action:dict。
    # 返回类型：None；类型/shape约定需与调用方配套。
    @abc.abstractmethod
    def on_step(self, observation: dict, action: dict) -> None:
        """Append a step to the episode."""

    # 【Subscriber.on_episode_end】处理runtime广播的事件，更新本订阅者的显示/记录状态。
    # 返回类型：None；类型/shape约定需与调用方配套。
    @abc.abstractmethod
    def on_episode_end(self) -> None:
        """Called when an episode ends."""
