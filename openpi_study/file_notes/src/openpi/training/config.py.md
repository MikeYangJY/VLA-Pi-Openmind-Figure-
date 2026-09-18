# `src/openpi/training/config.py` 中文阅读说明

**定位：** 训练总配置。

把模型、数据、权重加载、优化器和输出位置组合成具名实验。

**建议读法：** 先get_config定位pi0_libero/pi05_libero，再看TrainConfig、LeRobotLiberoDataConfig与ModelTransformFactory。

**易错点：** 数据配置同样用于推理；pi05为True不意味着公开了论文中的web共训、FAST联合目标和完整KI流程。

[注释源码](../../../../code/src/openpi/training/config.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [AssetsConfig](../../../../code/src/openpi/training/config.py#L45) | 数据统计等附属资产的位置与平台标识。 |
| [DataConfig](../../../../code/src/openpi/training/config.py#L74) | 数据来源及三层变换的集合，同时关联归一化资产。 |
| [GroupFactory](../../../../code/src/openpi/training/config.py#L117) | 把模型、数据、权重加载、优化器和输出位置组合成具名实验。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [GroupFactory.__call__](../../../../code/src/openpi/training/config.py#L121) | 执行把模型、数据、权重加载、优化器和输出位置组合成具名实验。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [ModelTransformFactory](../../../../code/src/openpi/training/config.py#L127) | 把模型、数据、权重加载、优化器和输出位置组合成具名实验。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [ModelTransformFactory.__call__](../../../../code/src/openpi/training/config.py#L137) | 选择模型对应的变换；π0与π0.5共用PaligemmaTokenizer，但后者可传入离散state。FAST分支不在本轮学习主线。 |
| [DataConfigFactory](../../../../code/src/openpi/training/config.py#L192) | 把模型、数据、权重加载、优化器和输出位置组合成具名实验。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [DataConfigFactory.create](../../../../code/src/openpi/training/config.py#L205) | 按当前配置构造所需对象或数据管线；创建结果交给上层脚本继续使用。 |
| [DataConfigFactory.create_base_config](../../../../code/src/openpi/training/config.py#L212) | 确定数据/资产ID、加载统计，并按模型类型设归一化方式；具体任务工厂继续补平台变换。 |
| [DataConfigFactory._load_norm_stats](../../../../code/src/openpi/training/config.py#L227) | 本函数位于“训练总配置”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 _normalize.load → _download.maybe_download → logging.info 追踪具体实现。 |
| [FakeDataConfig](../../../../code/src/openpi/training/config.py#L242) | 把模型、数据、权重加载、优化器和输出位置组合成具名实验。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [FakeDataConfig.create](../../../../code/src/openpi/training/config.py#L250) | 按当前配置构造所需对象或数据管线；创建结果交给上层脚本继续使用。 |
| [SimpleDataConfig](../../../../code/src/openpi/training/config.py#L256) | 把模型、数据、权重加载、优化器和输出位置组合成具名实验。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [SimpleDataConfig.create](../../../../code/src/openpi/training/config.py#L267) | 按当前配置构造所需对象或数据管线；创建结果交给上层脚本继续使用。 |
| [LeRobotAlohaDataConfig](../../../../code/src/openpi/training/config.py#L277) | 把模型、数据、权重加载、优化器和输出位置组合成具名实验。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [LeRobotAlohaDataConfig.create](../../../../code/src/openpi/training/config.py#L311) | 按当前配置构造所需对象或数据管线；创建结果交给上层脚本继续使用。 |
| [LeRobotLiberoDataConfig](../../../../code/src/openpi/training/config.py#L336) | 把模型、数据、权重加载、优化器和输出位置组合成具名实验。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [LeRobotLiberoDataConfig.create](../../../../code/src/openpi/training/config.py#L350) | 把数据集键映射成LIBERO推理键，添加平台输入/输出变换，再加模型变换；extra_delta是特定旧checkpoint兼容选项。 |
| [RLDSDroidDataConfig](../../../../code/src/openpi/training/config.py#L419) | 把模型、数据、权重加载、优化器和输出位置组合成具名实验。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [RLDSDroidDataConfig.create](../../../../code/src/openpi/training/config.py#L446) | 按当前配置构造所需对象或数据管线；创建结果交给上层脚本继续使用。 |
| [LeRobotDROIDDataConfig](../../../../code/src/openpi/training/config.py#L492) | 把模型、数据、权重加载、优化器和输出位置组合成具名实验。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [LeRobotDROIDDataConfig.create](../../../../code/src/openpi/training/config.py#L503) | 按当前配置构造所需对象或数据管线；创建结果交给上层脚本继续使用。 |
| [TrainConfig](../../../../code/src/openpi/training/config.py#L536) | 一项实验的完整配方：模型、数据、权重来源、优化器、训练步数与存档位置。 |
| [TrainConfig.assets_dirs](../../../../code/src/openpi/training/config.py#L625) | 本函数位于“训练总配置”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 (pathlib.Path(self.assets_base_dir) / self.name).resolve → pathlib.Path 追踪具体实现。 |
| [TrainConfig.checkpoint_dir](../../../../code/src/openpi/training/config.py#L633) | 本函数位于“训练总配置”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 ValueError → (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).r… → pathlib.Path 追踪具体实现。 |
| [TrainConfig.trainable_filter](../../../../code/src/openpi/training/config.py#L643) | 用参数类型与freeze_filter选出参与梯度更新的参数；配置存在不等于所有参数都训练。 |
| [TrainConfig.__post_init__](../../../../code/src/openpi/training/config.py#L649) | 本函数位于“训练总配置”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 ValueError 追踪具体实现。 |
| [cli](../../../../code/src/openpi/training/config.py#L1076) | 把命令行参数解析成配置对象，再交给脚本入口。 |
| [get_config](../../../../code/src/openpi/training/config.py#L1084) | 按名称返回配置；名称不匹配时进入错误处理，配置对象随后控制模型或实验构建。 |
