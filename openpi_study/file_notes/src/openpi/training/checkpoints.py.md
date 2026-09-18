# `src/openpi/training/checkpoints.py` 中文阅读说明

**定位：** 训练持久化。

管理实验目录、保存/恢复训练状态，并把归一化等资产一起保存。

**建议读法：** initialize_checkpoint_dir处理新建/续训；save_state分离参数与训练状态；restore_state恢复。

**易错点：** 断点续训需要优化器/步数等状态；推理只加载权重不是完整resume。

[注释源码](../../../../code/src/openpi/training/checkpoints.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/checkpoints.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [initialize_checkpoint_dir](../../../../code/src/openpi/training/checkpoints.py#L30) | 决定新建、恢复或覆盖实验目录，并创建checkpoint管理器；overwrite须留意已有文件。 |
| [save_state](../../../../code/src/openpi/training/checkpoints.py#L78) | 将训练状态与推理资产写入checkpoint，区分可恢复训练的信息和推理参数。 |
| [save_state.save_assets](../../../../code/src/openpi/training/checkpoints.py#L87) | 本函数位于“训练持久化”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 data_loader.data_config → _normalize.save 追踪具体实现。 |
| [restore_state](../../../../code/src/openpi/training/checkpoints.py#L109) | 把保存的参数、优化器及训练步等恢复到既定结构，延续原训练进度。 |
| [load_norm_stats](../../../../code/src/openpi/training/checkpoints.py#L134) | 本函数位于“训练持久化”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 epath.Path → _normalize.load → logging.info 追踪具体实现。 |
| [Callback](../../../../code/src/openpi/training/checkpoints.py#L142) | 管理实验目录、保存/恢复训练状态，并把归一化等资产一起保存。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [Callback.__call__](../../../../code/src/openpi/training/checkpoints.py#L146) | 执行管理实验目录、保存/恢复训练状态，并把归一化等资产一起保存。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [CallbackHandler](../../../../code/src/openpi/training/checkpoints.py#L150) | 管理实验目录、保存/恢复训练状态，并把归一化等资产一起保存。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [CallbackHandler.save](../../../../code/src/openpi/training/checkpoints.py#L156) | 把当前数据/状态写入指定存储；文件路径与覆盖行为由这里的实现决定。 |
| [CallbackHandler.async_save](../../../../code/src/openpi/training/checkpoints.py#L164) | 本函数位于“训练持久化”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 future.CommitFutureAwaitingContractedSignals → asyncio.to_thread 追踪具体实现。 |
| [CallbackHandler.restore](../../../../code/src/openpi/training/checkpoints.py#L168) | 本函数位于“训练持久化”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 NotImplementedError 追踪具体实现。 |
| [CallbackSave](../../../../code/src/openpi/training/checkpoints.py#L175) | 管理实验目录、保存/恢复训练状态，并把归一化等资产一起保存。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [CallbackRestore](../../../../code/src/openpi/training/checkpoints.py#L181) | 管理实验目录、保存/恢复训练状态，并把归一化等资产一起保存。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [_split_params](../../../../code/src/openpi/training/checkpoints.py#L187) | 本函数位于“训练持久化”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 dataclasses.replace 追踪具体实现。 |
| [_merge_params](../../../../code/src/openpi/training/checkpoints.py#L200) | 本函数位于“训练持久化”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 dataclasses.replace 追踪具体实现。 |
