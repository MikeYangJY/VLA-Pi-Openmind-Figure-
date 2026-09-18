# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：动作执行缓存｜缓存策略一次返回的动作块，每次infer只取当前时间步。
# 阅读顺序：首次请求模型→按_cur_step切片→计数→达到执行长度后清空缓存→重新请求。
# 重点边界：这不是RTC异步前缀约束；执行缓存期间不会自动用每帧新观测重新预测。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
from typing import Dict

import numpy as np
import tree
from typing_extensions import override

from openpi_client import base_policy as _base_policy


# 【ActionChunkBroker】在多次调用间缓存动作块，按执行步数输出单步动作。
class ActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """

    # 【ActionChunkBroker.__init__】初始化ActionChunkBroker的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._policy、self._action_horizon、self._cur_step、self._last_results。
    # 输入接口：policy:_base_policy.BasePolicy；action_horizon:int。
    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int):
        self._policy = policy
        self._action_horizon = action_horizon
        self._cur_step: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None

    # 【ActionChunkBroker.infer】只有缓存为空时才调用内部策略；随后每次取当前动作索引。达到执行长度便清空，下一次才利用新观测预测。
    # 输入接口：obs:Dict。
    # 返回类型：Dict；类型/shape约定需与调用方配套。
    # 内部调用线索：self._policy.infer → tree.map_structure（含分支中的调用，实际路径由条件决定）。
    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._last_results is None:
            self._last_results = self._policy.infer(obs)
            self._cur_step = 0

        # 【ActionChunkBroker.infer.slicer】本函数位于“动作执行缓存”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 isinstance 追踪具体实现。
        # 输入接口：x。
        # 返回值可从这里追踪：x[self._cur_step, ...] / x。
        def slicer(x):
            if isinstance(x, np.ndarray):
                return x[self._cur_step, ...]
            else:
                return x

        results = tree.map_structure(slicer, self._last_results)
        # 学习提示：这里增加的是已消费动作的索引，不是大模型训练步数。
        self._cur_step += 1

        # 学习提示：达到客户端选择的执行长度后丢弃缓存，下一次使用新观测重新预测。
        if self._cur_step >= self._action_horizon:
            self._last_results = None

        return results

    # 【ActionChunkBroker.reset】重置内部策略并清除旧episode动作缓存，防止下一任务执行剩余动作。
    # 返回类型：None；类型/shape约定需与调用方配套。
    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0
