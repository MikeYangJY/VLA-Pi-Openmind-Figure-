# `src/openpi/models/model.py` 中文阅读说明

**定位：** 输入与接口。

定义Observation、Actions以及所有模型共享的创建、加载、损失和采样接口。

**建议读法：** 先看Observation.from_dict/to_dict与数据示例，再看preprocess_observation；之后看BaseModelConfig.load。

**易错点：** 训练和推理必须使用一致的图像范围、相机键、状态维数及mask；接口声明不等于实现了某种能力。

[注释源码](../../../../code/src/openpi/models/model.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/model.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [ModelType](../../../../code/src/openpi/models/model.py#L37) | 定义Observation、Actions以及所有模型共享的创建、加载、损失和采样接口。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [Observation](../../../../code/src/openpi/models/model.py#L91) | 一次观测的结构化容器，含相机、mask、状态和文本token。 |
| [Observation.from_dict](../../../../code/src/openpi/models/model.py#L122) | 把统一数据字典变成Observation对象，并处理图像dtype/数值范围。调用前平台字段应已完成映射。 |
| [Observation.to_dict](../../../../code/src/openpi/models/model.py#L146) | 将Observation还原为统一字典，以便数据记录、PyTree处理或训练日志读取。 |
| [preprocess_observation](../../../../code/src/openpi/models/model.py#L163) | JAX端校验相机键，缩放图像，训练时做随机增强，并补图像mask；保留state和语言字段。 |
| [BaseModelConfig](../../../../code/src/openpi/models/model.py#L232) | 定义Observation、Actions以及所有模型共享的创建、加载、损失和采样接口。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [BaseModelConfig.model_type](../../../../code/src/openpi/models/model.py#L251) | 暴露模型类型，让数据变换和上层工厂选择匹配的处理路径。 |
| [BaseModelConfig.create](../../../../code/src/openpi/models/model.py#L258) | 按当前配置构造所需对象或数据管线；创建结果交给上层脚本继续使用。 |
| [BaseModelConfig.load](../../../../code/src/openpi/models/model.py#L265) | 先按配置构造参数结构，再检查checkpoint参数路径/形状并装入模型；不是重新学习权重。 |
| [BaseModelConfig.load_pytorch](../../../../code/src/openpi/models/model.py#L279) | 创建PI0Pytorch并加载safetensors权重；配置与权重结构必须对应。 |
| [BaseModelConfig.inputs_spec](../../../../code/src/openpi/models/model.py#L289) | 本函数位于“输入与接口”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。 |
| [BaseModelConfig.fake_obs](../../../../code/src/openpi/models/model.py#L296) | 根据输入规格生成虚拟观测，用于初始化和结构测试，不代表真实机器人数据分布。 |
| [BaseModelConfig.fake_act](../../../../code/src/openpi/models/model.py#L304) | 根据输入规格生成虚拟动作，用于检查损失和张量接口。 |
| [BaseModel](../../../../code/src/openpi/models/model.py#L311) | 定义Observation、Actions以及所有模型共享的创建、加载、损失和采样接口。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [BaseModel.compute_loss](../../../../code/src/openpi/models/model.py#L327) | 本函数位于“输入与接口”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。 |
| [BaseModel.sample_actions](../../../../code/src/openpi/models/model.py#L340) | 本函数位于“输入与接口”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。 |
| [restore_params](../../../../code/src/openpi/models/model.py#L347) | 从Orbax存档恢复模型参数，可指定dtype与分片；恢复权重本身不运行机器人。 |
