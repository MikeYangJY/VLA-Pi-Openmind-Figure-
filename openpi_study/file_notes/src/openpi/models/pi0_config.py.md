# `src/openpi/models/pi0_config.py` 中文阅读说明

**定位：** 核心配置。

决定模型大小、动作维数、预测长度和pi05开关，并构造模型。

**建议读法：** 从Pi0Config字段到__post_init__，再到inputs_spec和create；LoRA相关freeze_filter最后读。

**易错点：** 默认action_dim=32是统一模型宽度，不代表每台机器人都有32个关节；action_horizon=50是预测步数。

[注释源码](../../../../code/src/openpi/models/pi0_config.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0_config.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [Pi0Config](../../../../code/src/openpi/models/pi0_config.py#L26) | 决定模型大小、动作维数、预测长度和pi05开关，并构造模型。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [Pi0Config.__post_init__](../../../../code/src/openpi/models/pi0_config.py#L54) | 根据pi05开关补默认值：π0通常48个文本token，π0.5通常200；离散state默认跟随pi05。还校验PyTorch编译模式。 |
| [Pi0Config.model_type](../../../../code/src/openpi/models/pi0_config.py#L71) | 暴露模型类型，让数据变换和上层工厂选择匹配的处理路径。 |
| [Pi0Config.create](../../../../code/src/openpi/models/pi0_config.py#L82) | 按当前配置构造所需对象或数据管线；创建结果交给上层脚本继续使用。 |
| [Pi0Config.inputs_spec](../../../../code/src/openpi/models/pi0_config.py#L92) | 构造只含shape/dtype的信息，规定相机、state、token和动作的维度；这是接口模板，不会采集真机数据。 |
| [Pi0Config.get_freeze_filter](../../../../code/src/openpi/models/pi0_config.py#L119) | 根据LoRA分支选出不更新的参数；没有LoRA时返回不冻结。训练脚本再从中推导可训练参数。 |
