# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：测试环境｜为pytest选择可用的JAX设备并设置测试相关行为。
# 阅读顺序：读自动fixture和pytest_configure，理解为什么无GPU也可跑部分轻量测试。
# 重点边界：测试能启动不代表大模型可在该设备高效推理。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import os

import pynvml
import pytest


# 【set_jax_cpu_backend_if_no_gpu】本函数位于“测试环境”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 pynvml.nvmlInit → pynvml.nvmlShutdown 追踪具体实现。
# 返回类型：None；类型/shape约定需与调用方配套。
# 内部调用线索：pynvml.nvmlInit → pynvml.nvmlShutdown（含分支中的调用，实际路径由条件决定）。
def set_jax_cpu_backend_if_no_gpu() -> None:
    try:
        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        # No GPU found.
        os.environ["JAX_PLATFORMS"] = "cpu"


# 【pytest_configure】本函数位于“测试环境”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 set_jax_cpu_backend_if_no_gpu 追踪具体实现。
# 输入接口：config:pytest.Config。
# 返回类型：None；类型/shape约定需与调用方配套。
def pytest_configure(config: pytest.Config) -> None:
    set_jax_cpu_backend_if_no_gpu()
