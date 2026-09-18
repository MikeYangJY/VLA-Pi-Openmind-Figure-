# `src/openpi/training/droid_rlds_dataset.py` 中文阅读说明

**定位：** DROID数据支线。

流式读取RLDS轨迹，筛选、重组相机与动作，并沿时间构造未来动作块。

**建议读法：** 先看DroidRldsDataset构造中的restructure/chunk_actions/filter/decode_images，再看迭代器。

**易错点：** 这条路径有额外TensorFlow/RLDS依赖；先学LIBERO主线时可以后读。

[注释源码](../../../../code/src/openpi/training/droid_rlds_dataset.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/droid_rlds_dataset.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [DroidActionSpace](../../../../code/src/openpi/training/droid_rlds_dataset.py#L28) | 流式读取RLDS轨迹，筛选、重组相机与动作，并沿时间构造未来动作块。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [RLDSDataset](../../../../code/src/openpi/training/droid_rlds_dataset.py#L37) | 流式读取RLDS轨迹，筛选、重组相机与动作，并沿时间构造未来动作块。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [DroidRldsDataset](../../../../code/src/openpi/training/droid_rlds_dataset.py#L45) | 流式读取RLDS轨迹，筛选、重组相机与动作，并沿时间构造未来动作块。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [DroidRldsDataset.__init__](../../../../code/src/openpi/training/droid_rlds_dataset.py#L49) | 初始化DroidRldsDataset的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.dataset、self.batch_size、self.shuffle。 |
| [DroidRldsDataset.__init__.prepare_single_dataset](../../../../code/src/openpi/training/droid_rlds_dataset.py#L80) | 本函数位于“DROID数据支线”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 tfds.builder → dl.DLataset.from_rlds → dataset.filter 追踪具体实现。 |
| [DroidRldsDataset.__init__.prepare_single_dataset.restructure](../../../../code/src/openpi/training/droid_rlds_dataset.py#L135) | 本函数位于“DROID数据支线”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 tf.concat → tf.cond → tf.random.uniform 追踪具体实现。 |
| [DroidRldsDataset.__init__.prepare_single_dataset.chunk_actions](../../../../code/src/openpi/training/droid_rlds_dataset.py#L197) | 本函数位于“DROID数据支线”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 tf.shape → tf.broadcast_to → tf.range 追踪具体实现。 |
| [DroidRldsDataset.__init__.prepare_single_dataset.filter_from_dict](../../../../code/src/openpi/training/droid_rlds_dataset.py#L227) | 本函数位于“DROID数据支线”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。 |
| [DroidRldsDataset.__init__.prepare_single_dataset.remove_passes_filter](../../../../code/src/openpi/training/droid_rlds_dataset.py#L236) | 本函数位于“DROID数据支线”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 frame.pop 追踪具体实现。 |
| [DroidRldsDataset.__init__.prepare_single_dataset.decode_images](../../../../code/src/openpi/training/droid_rlds_dataset.py#L246) | 本函数位于“DROID数据支线”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 tf.io.decode_image 追踪具体实现。 |
| [DroidRldsDataset.__iter__](../../../../code/src/openpi/training/droid_rlds_dataset.py#L276) | 定义迭代读取方式；训练代码每次next从这里获得下一条样本或batch。 |
| [DroidRldsDataset.__len__](../../../../code/src/openpi/training/droid_rlds_dataset.py#L281) | 报告容器/数据集长度，供采样器或调用方安排迭代。 |
