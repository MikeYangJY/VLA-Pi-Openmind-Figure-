# `examples/droid/compute_droid_nonidle_ranges.py` 中文阅读说明

**定位：** 数据筛选。

根据DROID动作变化计算非静止区间，为大规模数据训练过滤空闲片段。

**建议读法：** 读取轨迹→比较运动变化→标出保留区间→保存过滤索引。

**易错点：** 静止不总等于无价值：接触等待等任务需要结合目标理解筛选阈值。

[注释源码](../../../code/examples/droid/compute_droid_nonidle_ranges.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/droid/compute_droid_nonidle_ranges.py)
