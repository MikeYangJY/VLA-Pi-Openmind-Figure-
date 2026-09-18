# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：验证与示例｜用小样本和断言检查msgpack_numpy相关行为，是理解预期输入输出的例子。
# 阅读顺序：先看test函数里的构造输入，再看被测调用和assert/数值比较。
# 重点边界：测试覆盖的是列出的行为，不等于完整机器人部署或任务成功率验证。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import numpy as np
import pytest
import tree

from openpi_client import msgpack_numpy


# 【_check】本函数位于“验证与示例”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 isinstance → np.array_equal 追踪具体实现。
# 输入接口：expected；actual。
# 内部调用线索：isinstance → np.array_equal（含分支中的调用，实际路径由条件决定）。
def _check(expected, actual):
    if isinstance(expected, np.ndarray):
        assert expected.shape == actual.shape
        assert expected.dtype == actual.dtype
        assert np.array_equal(expected, actual, equal_nan=expected.dtype.kind == "f")
    else:
        assert expected == actual


# 【test_pack_unpack】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。数值比较或预期异常给出通过条件。
# 输入接口：data。
# 内部调用线索：msgpack_numpy.packb → msgpack_numpy.unpackb → tree.map_structure（含分支中的调用，实际路径由条件决定）。
@pytest.mark.parametrize(
    "data",
    [
        1,  # int
        1.0,  # float
        "hello",  # string
        np.bool_(True),  # boolean scalar
        np.array([1, 2, 3])[0],  # int scalar
        np.str_("asdf"),  # string scalar
        [1, 2, 3],  # list
        {"key": "value"},  # dict
        {"key": [1, 2, 3]},  # nested dict
        np.array(1.0),  # 0D array
        np.array([1, 2, 3], dtype=np.int32),  # 1D integer array
        np.array(["asdf", "qwer"]),  # string array
        np.array([True, False]),  # boolean array
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),  # 2D float array
        np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int16),  # 3D integer array
        np.array([np.nan, np.inf, -np.inf]),  # special float values
        {"arr": np.array([1, 2, 3]), "nested": {"arr": np.array([4, 5, 6])}},  # nested dict with arrays
        [np.array([1, 2]), np.array([3, 4])],  # list of arrays
        np.zeros((3, 4, 5), dtype=np.float32),  # 3D zeros
        np.ones((2, 3), dtype=np.float64),  # 2D ones with double precision
    ],
)
def test_pack_unpack(data):
    packed = msgpack_numpy.packb(data)
    unpacked = msgpack_numpy.unpackb(packed)
    tree.map_structure(_check, data, unpacked)
