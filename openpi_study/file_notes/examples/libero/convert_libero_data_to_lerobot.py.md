# `examples/libero/convert_libero_data_to_lerobot.py` 中文阅读说明

**定位：** 数据转换示例。

将LIBERO轨迹转成LeRobot帧与episode格式，保留图像、状态、动作和任务标签。

**建议读法：** 先定义features，再遍历episode/step添加帧，最后保存episode与数据集。

**易错点：** 数据格式转换不训练模型；时间顺序、动作单位和任务标签必须保留。

[注释源码](../../../code/examples/libero/convert_libero_data_to_lerobot.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/convert_libero_data_to_lerobot.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [main](../../../code/examples/libero/convert_libero_data_to_lerobot.py#L46) | 本脚本入口：将LIBERO轨迹转成LeRobot帧与episode格式，保留图像、状态、动作和任务标签。 先定义features，再遍历episode/step添加帧，最后保存episode与数据集。 |
