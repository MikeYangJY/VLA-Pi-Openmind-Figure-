# `src/openpi/models_pytorch/preprocessing_pytorch.py` 中文阅读说明

**定位：** 图像预处理对照。

把Observation图像转为PyTorch所需布局、大小和训练增强，并保留其他字段。

**建议读法：** 检查相机键与张量形状，进行resize/augmentation，构造处理后的Observation。

**易错点：** 训练可以随机增强，推理应确定性处理；图像通道布局和[-1,1]数值范围要一起检查。

[注释源码](../../../../code/src/openpi/models_pytorch/preprocessing_pytorch.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models_pytorch/preprocessing_pytorch.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [preprocess_observation_pytorch](../../../../code/src/openpi/models_pytorch/preprocessing_pytorch.py#L30) | 本函数位于“图像预处理对照”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 set(image_keys).issubset → set → ValueError 追踪具体实现。 |
| [preprocess_observation_pytorch.SimpleProcessedObservation](../../../../code/src/openpi/models_pytorch/preprocessing_pytorch.py#L171) | 把Observation图像转为PyTorch所需布局、大小和训练增强，并保留其他字段。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [preprocess_observation_pytorch.SimpleProcessedObservation.__init__](../../../../code/src/openpi/models_pytorch/preprocessing_pytorch.py#L174) | 初始化SimpleProcessedObservation的依赖和内部状态；构造对象本身与后续执行/训练要区分。 |
