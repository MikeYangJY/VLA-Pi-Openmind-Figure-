# `scripts/train.py` 中文阅读说明

**定位：** 训练主入口。

JAX训练完整流程：加载配置/数据/权重，编译训练步，更新并保存参数。

**建议读法：** 先train_step理解一次更新，再main理解循环，最后init_train_state理解初始化与分片。

**易错点：** 模型compute_loss只算误差，value_and_grad和optimizer负责学习；checkpoint overwrite选项会影响已有实验目录。

[注释源码](../../code/scripts/train.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/scripts/train.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [init_logging](../../code/scripts/train.py#L39) | 本函数位于“训练主入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 CustomFormatter → logging.getLogger → logger.setLevel 追踪具体实现。 |
| [init_logging.CustomFormatter](../../code/scripts/train.py#L44) | JAX训练完整流程：加载配置/数据/权重，编译训练步，更新并保存参数。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [init_logging.CustomFormatter.format](../../code/scripts/train.py#L49) | 本函数位于“训练主入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 level_mapping.get → super().format 追踪具体实现。 |
| [init_wandb](../../code/scripts/train.py#L66) | 本函数位于“训练主入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 wandb.init → ckpt_dir.exists → FileNotFoundError 追踪具体实现。 |
| [_load_weights_and_validate](../../code/scripts/train.py#L93) | 读取初始化权重并校验PyTree结构、形状与dtype，滤除只表示形状而非真实参数的占位。 |
| [init_train_state](../../code/scripts/train.py#L109) | 创建优化器和模型状态，按需加载预训练参数并设置分片；resume路径先准备结构供恢复。 |
| [init_train_state.init](../../code/scripts/train.py#L118) | 本函数位于“训练主入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 jax.random.split → config.model.create → nnx.split 追踪具体实现。 |
| [train_step](../../code/scripts/train.py#L169) | JAX的一次参数更新：组回模型→compute_loss→对可训练参数求梯度→优化器更新→可选EMA→返回状态与指标。 |
| [train_step.loss_fn](../../code/scripts/train.py#L183) | 把模型按样本/时间返回的误差取均值为标量，便于自动微分。 |
| [main](../../code/scripts/train.py#L236) | 本脚本入口：JAX训练完整流程：加载配置/数据/权重，编译训练步，更新并保存参数。 先train_step理解一次更新，再main理解循环，最后init_train_state理解初始化与分片。 |
