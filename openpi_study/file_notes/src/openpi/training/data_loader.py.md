# `src/openpi/training/data_loader.py` 中文阅读说明

**定位：** 训练数据主线。

从LeRobot或RLDS读样本，应用变换、组batch，再转成模型需要的Observation与Actions。

**建议读法：** create_data_loader→平台dataset→transform_dataset→TorchDataLoader/RLDSDataLoader→DataLoaderImpl。

**易错点：** JAX训练也可使用PyTorch DataLoader读数据；读数据后端不等于模型计算后端。

[注释源码](../../../../code/src/openpi/training/data_loader.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/data_loader.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [Dataset](../../../../code/src/openpi/training/data_loader.py#L29) | 从LeRobot或RLDS读样本，应用变换、组batch，再转成模型需要的Observation与Actions。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [Dataset.__getitem__](../../../../code/src/openpi/training/data_loader.py#L35) | 按索引读取一个样本，并执行此包装器定义的变换。 |
| [Dataset.__len__](../../../../code/src/openpi/training/data_loader.py#L40) | 报告容器/数据集长度，供采样器或调用方安排迭代。 |
| [IterableDataset](../../../../code/src/openpi/training/data_loader.py#L45) | 从LeRobot或RLDS读样本，应用变换、组batch，再转成模型需要的Observation与Actions。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [IterableDataset.__iter__](../../../../code/src/openpi/training/data_loader.py#L50) | 定义迭代读取方式；训练代码每次next从这里获得下一条样本或batch。 |
| [IterableDataset.__len__](../../../../code/src/openpi/training/data_loader.py#L55) | 报告容器/数据集长度，供采样器或调用方安排迭代。 |
| [DataLoader](../../../../code/src/openpi/training/data_loader.py#L60) | 从LeRobot或RLDS读样本，应用变换、组batch，再转成模型需要的Observation与Actions。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [DataLoader.data_config](../../../../code/src/openpi/training/data_loader.py#L65) | 本函数位于“训练数据主线”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 NotImplementedError 追踪具体实现。 |
| [DataLoader.__iter__](../../../../code/src/openpi/training/data_loader.py#L71) | 定义迭代读取方式；训练代码每次next从这里获得下一条样本或batch。 |
| [TransformedDataset](../../../../code/src/openpi/training/data_loader.py#L76) | 对按索引读取的样本应用统一变换。 |
| [TransformedDataset.__init__](../../../../code/src/openpi/training/data_loader.py#L79) | 初始化TransformedDataset的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._dataset、self._transform。 |
| [TransformedDataset.__getitem__](../../../../code/src/openpi/training/data_loader.py#L86) | 按索引读取一个样本，并执行此包装器定义的变换。 |
| [TransformedDataset.__len__](../../../../code/src/openpi/training/data_loader.py#L91) | 报告容器/数据集长度，供采样器或调用方安排迭代。 |
| [IterableTransformedDataset](../../../../code/src/openpi/training/data_loader.py#L96) | 对流式样本或batch应用统一变换。 |
| [IterableTransformedDataset.__init__](../../../../code/src/openpi/training/data_loader.py#L99) | 初始化IterableTransformedDataset的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._dataset、self._transform、self._is_batched。 |
| [IterableTransformedDataset.__iter__](../../../../code/src/openpi/training/data_loader.py#L112) | 定义迭代读取方式；训练代码每次next从这里获得下一条样本或batch。 |
| [IterableTransformedDataset.__len__](../../../../code/src/openpi/training/data_loader.py#L132) | 报告容器/数据集长度，供采样器或调用方安排迭代。 |
| [FakeDataset](../../../../code/src/openpi/training/data_loader.py#L137) | 按模型输入规格生成虚拟训练样本，用于管线测试。 |
| [FakeDataset.__init__](../../../../code/src/openpi/training/data_loader.py#L140) | 初始化FakeDataset的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._num_samples。 |
| [FakeDataset.__getitem__](../../../../code/src/openpi/training/data_loader.py#L148) | 按索引读取一个样本，并执行此包装器定义的变换。 |
| [FakeDataset.__getitem__.make_from_spec](../../../../code/src/openpi/training/data_loader.py#L155) | 本函数位于“训练数据主线”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 jax.random.split → jax.random.uniform → jax.random.randint 追踪具体实现。 |
| [FakeDataset.__len__](../../../../code/src/openpi/training/data_loader.py#L176) | 报告容器/数据集长度，供采样器或调用方安排迭代。 |
| [create_torch_dataset](../../../../code/src/openpi/training/data_loader.py#L184) | 创建LeRobot/Fake数据集，并按action_horizon要求读取未来动作序列；可能从task索引补prompt。 |
| [create_rlds_dataset](../../../../code/src/openpi/training/data_loader.py#L211) | 本函数位于“训练数据主线”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 DroidRldsDataset 追踪具体实现。 |
| [transform_dataset](../../../../code/src/openpi/training/data_loader.py#L233) | 严格按repack→平台适配→归一化→模型变换的顺序包装数据集；未有统计时提示先计算。 |
| [transform_iterable_dataset](../../../../code/src/openpi/training/data_loader.py#L259) | 对流式数据集使用相同变换顺序，并兼容输入已是batch的情况。 |
| [create_data_loader](../../../../code/src/openpi/training/data_loader.py#L292) | 根据TrainConfig创建数据配置，按是否指定RLDS目录选择数据后端，返回统一batch迭代器。 |
| [create_torch_data_loader](../../../../code/src/openpi/training/data_loader.py#L344) | 组织数据集、采样器、worker与batch；分布式训练时拆分全局batch，避免每张卡重复读取全部样本。 |
| [create_rlds_data_loader](../../../../code/src/openpi/training/data_loader.py#L417) | 本函数位于“训练数据主线”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 NotImplementedError → create_rlds_dataset → transform_iterable_dataset 追踪具体实现。 |
| [TorchDataLoader](../../../../code/src/openpi/training/data_loader.py#L459) | 用PyTorch数据加载机制迭代batch，可服务JAX或PyTorch模型。 |
| [TorchDataLoader.__init__](../../../../code/src/openpi/training/data_loader.py#L465) | 初始化TorchDataLoader的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._sharding、self._num_batches、self._data_loader。 |
| [TorchDataLoader.torch_loader](../../../../code/src/openpi/training/data_loader.py#L532) | 本函数位于“训练数据主线”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。 |
| [TorchDataLoader.__iter__](../../../../code/src/openpi/training/data_loader.py#L537) | 定义迭代读取方式；训练代码每次next从这里获得下一条样本或batch。 |
| [_collate_fn](../../../../code/src/openpi/training/data_loader.py#L560) | 将多条样本沿batch轴堆叠，保持嵌套字典/数组结构。 |
| [_worker_init_fn](../../../../code/src/openpi/training/data_loader.py#L570) | 为数据加载子进程关闭JAX显存预分配并使用platform分配器；这里不设置随机种子，也不能在JAX已导入后靠它切换后端。 |
| [RLDSDataLoader](../../../../code/src/openpi/training/data_loader.py#L579) | 从LeRobot或RLDS读样本，应用变换、组batch，再转成模型需要的Observation与Actions。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [RLDSDataLoader.__init__](../../../../code/src/openpi/training/data_loader.py#L588) | 初始化RLDSDataLoader的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._dataset、self._num_batches、self._sharding。 |
| [RLDSDataLoader.__iter__](../../../../code/src/openpi/training/data_loader.py#L613) | 定义迭代读取方式；训练代码每次next从这里获得下一条样本或batch。 |
| [DataLoaderImpl](../../../../code/src/openpi/training/data_loader.py#L629) | 把底层字典batch转成统一模型输入/目标。 |
| [DataLoaderImpl.__init__](../../../../code/src/openpi/training/data_loader.py#L632) | 初始化DataLoaderImpl的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._data_config、self._data_loader。 |
| [DataLoaderImpl.data_config](../../../../code/src/openpi/training/data_loader.py#L638) | 本函数位于“训练数据主线”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。 |
| [DataLoaderImpl.__iter__](../../../../code/src/openpi/training/data_loader.py#L642) | 将底层loader给出的字典拆成Observation和Actions，交给训练步。 |
