# `src/openpi/training/utils.py` 中文阅读说明

**定位：** 训练状态容器。

用TrainState集中保存步数、参数、优化器状态、模型结构及EMA。

**建议读法：** 对照scripts/train.py中状态如何创建、更新、保存；打印工具用于检查数组树形状。

**易错点：** model_def是结构而非训练好的数值权重；EMA是一份平滑参数。

[注释源码](../../../../code/src/openpi/training/utils.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/utils.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [TrainState](../../../../code/src/openpi/training/utils.py#L22) | 训练时需保存的全部状态，不只是神经网络参数。 |
| [tree_to_info](../../../../code/src/openpi/training/utils.py#L39) | 本函数位于“训练状态容器”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 jax.tree_util.tree_flatten_with_path → '\n'.join → jax.tree_util.keystr 追踪具体实现。 |
| [array_tree_to_info](../../../../code/src/openpi/training/utils.py#L51) | 本函数位于“训练状态容器”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 tree_to_info 追踪具体实现。 |
