# `src/openpi/training/optimizer.py` 中文阅读说明

**定位：** 参数更新。

定义学习率日程与优化器配置，把梯度变成参数更新。

**建议读法：** 从create_optimizer追到AdamW.create及CosineDecaySchedule；区分学习率、梯度裁剪和权重衰减。

**易错点：** 优化器不决定训练目标；flow loss是在模型中定义的。

[注释源码](../../../../code/src/openpi/training/optimizer.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/optimizer.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [LRScheduleConfig](../../../../code/src/openpi/training/optimizer.py#L18) | 定义学习率日程与优化器配置，把梯度变成参数更新。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [LRScheduleConfig.create](../../../../code/src/openpi/training/optimizer.py#L21) | 按当前配置构造所需对象或数据管线；创建结果交给上层脚本继续使用。 |
| [CosineDecaySchedule](../../../../code/src/openpi/training/optimizer.py#L26) | 定义学习率日程与优化器配置，把梯度变成参数更新。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [CosineDecaySchedule.create](../../../../code/src/openpi/training/optimizer.py#L36) | 按当前配置构造所需对象或数据管线；创建结果交给上层脚本继续使用。 |
| [RsqrtDecaySchedule](../../../../code/src/openpi/training/optimizer.py#L48) | 定义学习率日程与优化器配置，把梯度变成参数更新。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [RsqrtDecaySchedule.create](../../../../code/src/openpi/training/optimizer.py#L58) | 按当前配置构造所需对象或数据管线；创建结果交给上层脚本继续使用。 |
| [OptimizerConfig](../../../../code/src/openpi/training/optimizer.py#L74) | 定义学习率日程与优化器配置，把梯度变成参数更新。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [OptimizerConfig.create](../../../../code/src/openpi/training/optimizer.py#L78) | 按当前配置构造所需对象或数据管线；创建结果交给上层脚本继续使用。 |
| [AdamW](../../../../code/src/openpi/training/optimizer.py#L87) | 定义学习率日程与优化器配置，把梯度变成参数更新。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [AdamW.create](../../../../code/src/openpi/training/optimizer.py#L101) | 按当前配置构造所需对象或数据管线；创建结果交给上层脚本继续使用。 |
| [SGD](../../../../code/src/openpi/training/optimizer.py#L115) | 定义学习率日程与优化器配置，把梯度变成参数更新。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [SGD.create](../../../../code/src/openpi/training/optimizer.py#L125) | 按当前配置构造所需对象或数据管线；创建结果交给上层脚本继续使用。 |
| [create_optimizer](../../../../code/src/openpi/training/optimizer.py#L138) | 把学习率日程和优化器配置组装成更新规则，按需加权重衰减mask。 |
