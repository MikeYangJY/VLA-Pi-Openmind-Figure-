# `src/openpi/training/sharding.py` 中文阅读说明

**定位：** 多GPU工程。

构造设备mesh与张量分片规则，为JAX训练降低单卡内存压力。

**建议读法：** make_mesh定义设备轴；fsdp_sharding按参数形状分配分片；activation约束作用于中间张量。

**易错点：** 分片改变计算放置，不改变任务监督；单卡读懂模型后再看本文件。

[注释源码](../../../../code/src/openpi/training/sharding.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/sharding.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [_MeshState](../../../../code/src/openpi/training/sharding.py#L20) | 构造设备mesh与张量分片规则，为JAX训练降低单卡内存压力。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [make_mesh](../../../../code/src/openpi/training/sharding.py#L28) | 本函数位于“多GPU工程”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 jax.device_count → ValueError → jax.make_mesh 追踪具体实现。 |
| [set_mesh](../../../../code/src/openpi/training/sharding.py#L40) | 本函数位于“多GPU工程”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 ValueError 追踪具体实现。 |
| [activation_sharding_constraint](../../../../code/src/openpi/training/sharding.py#L57) | 本函数位于“多GPU工程”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 jax.lax.with_sharding_constraint → jax.sharding.NamedSharding → jax.sharding.PartitionSpec 追踪具体实现。 |
| [fsdp_sharding](../../../../code/src/openpi/training/sharding.py#L68) | 本函数位于“多GPU工程”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 jax.tree_util.tree_map_with_path 追踪具体实现。 |
| [fsdp_sharding._shard_arr](../../../../code/src/openpi/training/sharding.py#L94) | 本函数位于“多GPU工程”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 jax.sharding.NamedSharding → jax.sharding.PartitionSpec → hasattr 追踪具体实现。 |
