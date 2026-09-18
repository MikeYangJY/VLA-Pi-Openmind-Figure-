# `src/openpi/policies/policy_config.py` 中文阅读说明

**定位：** 推理组装。

根据训练配置、checkpoint和归一化统计，构造可以infer的Policy。

**建议读法：** 识别权重格式→加载模型→创建数据配置→加载同checkpoint统计→按顺序组装输入/输出变换。

**易错点：** 参数与norm_stats必须配套；这一步不会启动训练。

[注释源码](../../../../code/src/openpi/policies/policy_config.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/policy_config.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [create_trained_policy](../../../../code/src/openpi/policies/policy_config.py#L26) | 组装推理系统：权重、数据适配、归一化、token化与输出逆变换；读取checkpoint里的统计确保与训练一致。 |
