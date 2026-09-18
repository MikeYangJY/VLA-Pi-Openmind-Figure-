# `src/openpi/training/weight_loaders.py` 中文阅读说明

**定位：** 模型初始化。

把预训练checkpoint或PaliGemma权重映射进当前模型参数树。

**建议读法：** 先看WeightLoader接口，再看CheckpointWeightLoader与PaliGemmaWeightLoader的加载/合并规则。

**易错点：** 只匹配部分权重时其余参数需要初始化；参数树路径和形状必须正确。

[注释源码](../../../../code/src/openpi/training/weight_loaders.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/weight_loaders.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [WeightLoader](../../../../code/src/openpi/training/weight_loaders.py#L24) | 把预训练checkpoint或PaliGemma权重映射进当前模型参数树。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [WeightLoader.load](../../../../code/src/openpi/training/weight_loaders.py#L28) | 读取或合并外部资源/参数；检查路径、结构与调用方期待是否一致。 |
| [NoOpWeightLoader](../../../../code/src/openpi/training/weight_loaders.py#L43) | 把预训练checkpoint或PaliGemma权重映射进当前模型参数树。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [NoOpWeightLoader.load](../../../../code/src/openpi/training/weight_loaders.py#L47) | 读取或合并外部资源/参数；检查路径、结构与调用方期待是否一致。 |
| [CheckpointWeightLoader](../../../../code/src/openpi/training/weight_loaders.py#L53) | 把预训练checkpoint或PaliGemma权重映射进当前模型参数树。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [CheckpointWeightLoader.load](../../../../code/src/openpi/training/weight_loaders.py#L69) | 读取或合并外部资源/参数；检查路径、结构与调用方期待是否一致。 |
| [PaliGemmaWeightLoader](../../../../code/src/openpi/training/weight_loaders.py#L78) | 把预训练checkpoint或PaliGemma权重映射进当前模型参数树。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [PaliGemmaWeightLoader.load](../../../../code/src/openpi/training/weight_loaders.py#L89) | 读取或合并外部资源/参数；检查路径、结构与调用方期待是否一致。 |
| [_merge_params](../../../../code/src/openpi/training/weight_loaders.py#L104) | 本函数位于“模型初始化”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 flax.traverse_util.flatten_dict → flat_loaded.items → v.astype 追踪具体实现。 |
