# `src/openpi/shared/normalize.py` 中文阅读说明

**定位：** 数据统计底层。

维护流式均值方差和直方图分位数，并保存/读取NormStats。

**建议读法：** RunningStats.update累计样本，get_statistics输出统计，serialize/save负责存盘。

**易错点：** 归一化统计来自数据，不是神经网络可训练权重。

[注释源码](../../../../code/src/openpi/shared/normalize.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/shared/normalize.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [NormStats](../../../../code/src/openpi/shared/normalize.py#L17) | 每个物理维度的mean/std/q01/q99等统计。 |
| [RunningStats](../../../../code/src/openpi/shared/normalize.py#L25) | 可以逐批更新的统计器，用于大数据集而非模型梯度学习。 |
| [RunningStats.__init__](../../../../code/src/openpi/shared/normalize.py#L29) | 初始化RunningStats的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._count、self._mean、self._mean_of_squares、self._min、self._max。 |
| [RunningStats.update](../../../../code/src/openpi/shared/normalize.py#L43) | 流式累积每一维的样本统计与直方图，无需一次把全部数据放入内存。 |
| [RunningStats.get_statistics](../../../../code/src/openpi/shared/normalize.py#L89) | 从累计值输出mean/std和近似分位数，供Normalize/Unnormalize使用。 |
| [RunningStats._adjust_histograms](../../../../code/src/openpi/shared/normalize.py#L106) | 本函数位于“数据统计底层”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.linspace → np.histogram 追踪具体实现。 |
| [RunningStats._update_histograms](../../../../code/src/openpi/shared/normalize.py#L121) | 本函数位于“数据统计底层”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.histogram 追踪具体实现。 |
| [RunningStats._compute_quantiles](../../../../code/src/openpi/shared/normalize.py#L131) | 本函数位于“数据统计底层”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.cumsum → np.searchsorted → q_values.append 追踪具体实现。 |
| [_NormStatsDict](../../../../code/src/openpi/shared/normalize.py#L146) | 维护流式均值方差和直方图分位数，并保存/读取NormStats。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [serialize_json](../../../../code/src/openpi/shared/normalize.py#L155) | 本函数位于“数据统计底层”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 _NormStatsDict(norm_stats=norm_stats).model_dump_json → _NormStatsDict 追踪具体实现。 |
| [deserialize_json](../../../../code/src/openpi/shared/normalize.py#L164) | 本函数位于“数据统计底层”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 _NormStatsDict → json.loads 追踪具体实现。 |
| [save](../../../../code/src/openpi/shared/normalize.py#L173) | 把当前数据/状态写入指定存储；文件路径与覆盖行为由这里的实现决定。 |
| [load](../../../../code/src/openpi/shared/normalize.py#L184) | 读取或合并外部资源/参数；检查路径、结构与调用方期待是否一致。 |
