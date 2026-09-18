# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：网络序列化｜让MessagePack能够携带NumPy数组的内容、形状和dtype。
# 阅读顺序：pack_array附带数组元数据；unpack_array据此恢复数组；普通对象按默认流程处理。
# 重点边界：序列化只改变传输形式，不应改变数组数值或维度。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
"""Adds NumPy array support to msgpack.

msgpack is good for (de)serializing data over a network for multiple reasons:
- msgpack is secure (as opposed to pickle/dill/etc which allow for arbitrary code execution)
- msgpack is widely used and has good cross-language support
- msgpack does not require a schema (as opposed to protobuf/flatbuffers/etc) which is convenient in dynamically typed
    languages like Python and JavaScript
- msgpack is fast and efficient (as opposed to readable formats like JSON/YAML/etc); I found that msgpack was ~4x faster
    than pickle for serializing large arrays using the below strategy

The code below is adapted from https://github.com/lebedov/msgpack-numpy. The reason not to use that library directly is
that it falls back to pickle for object arrays.
"""

import functools

import msgpack
import numpy as np


# 【pack_array】把NumPy数组转换成MessagePack可传输结构，保留shape、dtype和字节。
# 输入接口：obj。
# 返回值可从这里追踪：{b'__ndarray__': True, b'data': obj.tobytes(), b'dtype': obj.dtype.str, b'shape': obj.shape} / {b'__npgeneric__': True, b'data': obj.item(), b'dtype': obj.dtype.str}。
# 内部调用线索：isinstance → ValueError → obj.tobytes → obj.item（含分支中的调用，实际路径由条件决定）。
def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


# 【unpack_array】根据记录的shape/dtype从传输字节恢复NumPy数组。
# 输入接口：obj。
# 返回值可从这里追踪：np.ndarray(buffer=obj[b'data'], dtype=np.dtype(obj[b'dtype']), shape=obj[b'shape']) / np.dtype(obj[b'dtype']).type(obj[b'data'])。
# 内部调用线索：np.ndarray → np.dtype → np.dtype(obj[b'dtype']).type（含分支中的调用，实际路径由条件决定）。
def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)
