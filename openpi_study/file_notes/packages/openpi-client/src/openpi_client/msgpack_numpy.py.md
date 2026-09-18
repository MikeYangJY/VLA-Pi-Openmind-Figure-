# `packages/openpi-client/src/openpi_client/msgpack_numpy.py` 中文阅读说明

**定位：** 网络序列化。

让MessagePack能够携带NumPy数组的内容、形状和dtype。

**建议读法：** pack_array附带数组元数据；unpack_array据此恢复数组；普通对象按默认流程处理。

**易错点：** 序列化只改变传输形式，不应改变数组数值或维度。

[注释源码](../../../../../code/packages/openpi-client/src/openpi_client/msgpack_numpy.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/packages/openpi-client/src/openpi_client/msgpack_numpy.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [pack_array](../../../../../code/packages/openpi-client/src/openpi_client/msgpack_numpy.py#L31) | 把NumPy数组转换成MessagePack可传输结构，保留shape、dtype和字节。 |
| [unpack_array](../../../../../code/packages/openpi-client/src/openpi_client/msgpack_numpy.py#L57) | 根据记录的shape/dtype从传输字节恢复NumPy数组。 |
