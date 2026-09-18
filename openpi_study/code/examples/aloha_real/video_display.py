# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：观测可视化｜订阅episode与step事件，实时展示相机画面。
# 阅读顺序：episode开始初始化窗口；每步更新画面；结束清理。
# 重点边界：显示画面不修改策略权重，也不能代替成功率评价。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import matplotlib.pyplot as plt
import numpy as np
from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override


# 【VideoDisplay】订阅episode与step事件，实时展示相机画面。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
class VideoDisplay(_subscriber.Subscriber):
    """Displays video frames."""

    # 【VideoDisplay.__init__】初始化VideoDisplay的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._ax、self._plt_img。
    # 返回类型：None；类型/shape约定需与调用方配套。
    def __init__(self) -> None:
        self._ax: plt.Axes | None = None
        self._plt_img: plt.Image | None = None

    # 【VideoDisplay.on_episode_start】处理runtime广播的事件，更新本订阅者的显示/记录状态。
    # 返回类型：None；类型/shape约定需与调用方配套。
    # 内部调用线索：plt.ion → plt.subplot（含分支中的调用，实际路径由条件决定）。
    @override
    def on_episode_start(self) -> None:
        plt.ion()
        self._ax = plt.subplot()
        self._plt_img = None

    # 【VideoDisplay.on_step】处理runtime广播的事件，更新本订阅者的显示/记录状态。
    # 输入接口：observation:dict；action:dict。
    # 返回类型：None；类型/shape约定需与调用方配套。
    # 内部调用线索：np.transpose → self._ax.imshow → self._plt_img.set_data → plt.pause（含分支中的调用，实际路径由条件决定）。
    @override
    def on_step(self, observation: dict, action: dict) -> None:
        assert self._ax is not None

        im = observation["image"][0]  # [C, H, W]
        im = np.transpose(im, (1, 2, 0))  # [H, W, C]

        if self._plt_img is None:
            self._plt_img = self._ax.imshow(im)
        else:
            self._plt_img.set_data(im)
        plt.pause(0.001)

    # 【VideoDisplay.on_episode_end】处理runtime广播的事件，更新本订阅者的显示/记录状态。
    # 返回类型：None；类型/shape约定需与调用方配套。
    # 内部调用线索：plt.ioff → plt.close（含分支中的调用，实际路径由条件决定）。
    @override
    def on_episode_end(self) -> None:
        plt.ioff()
        plt.close()
