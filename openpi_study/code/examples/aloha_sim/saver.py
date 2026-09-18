# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：评测记录｜订阅运行事件，把每步图像累积成episode视频。
# 阅读顺序：开始清空帧列表；每步收集帧；结束写视频。
# 重点边界：视频是观察行为的辅助材料，不能替代逐次试验结果。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import logging
import pathlib

import imageio
import numpy as np
from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override


# 【VideoSaver】订阅运行事件，把每步图像累积成episode视频。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
class VideoSaver(_subscriber.Subscriber):
    """Saves episode data."""

    # 【VideoSaver.__init__】初始化VideoSaver的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._out_dir、self._images、self._subsample。
    # 输入接口：out_dir:pathlib.Path；subsample:int。
    # 返回类型：None；类型/shape约定需与调用方配套。
    def __init__(self, out_dir: pathlib.Path, subsample: int = 1) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir
        self._images: list[np.ndarray] = []
        self._subsample = subsample

    # 【VideoSaver.on_episode_start】处理runtime广播的事件，更新本订阅者的显示/记录状态。
    # 返回类型：None；类型/shape约定需与调用方配套。
    @override
    def on_episode_start(self) -> None:
        self._images = []

    # 【VideoSaver.on_step】处理runtime广播的事件，更新本订阅者的显示/记录状态。
    # 输入接口：observation:dict；action:dict。
    # 返回类型：None；类型/shape约定需与调用方配套。
    # 内部调用线索：np.transpose → self._images.append（含分支中的调用，实际路径由条件决定）。
    @override
    def on_step(self, observation: dict, action: dict) -> None:
        im = observation["images"]["cam_high"]  # [C, H, W]
        im = np.transpose(im, (1, 2, 0))  # [H, W, C]
        self._images.append(im)

    # 【VideoSaver.on_episode_end】处理runtime广播的事件，更新本订阅者的显示/记录状态。
    # 返回类型：None；类型/shape约定需与调用方配套。
    # 内部调用线索：self._out_dir.glob → max → p.stem.split → logging.info → imageio.mimwrite（含分支中的调用，实际路径由条件决定）。
    @override
    def on_episode_end(self) -> None:
        existing = list(self._out_dir.glob("out_[0-9]*.mp4"))
        next_idx = max([int(p.stem.split("_")[1]) for p in existing], default=-1) + 1
        out_path = self._out_dir / f"out_{next_idx}.mp4"

        logging.info(f"Saving video to {out_path}")
        imageio.mimwrite(
            out_path,
            [np.asarray(x) for x in self._images[:: self._subsample]],
            fps=50 // max(1, self._subsample),
        )
