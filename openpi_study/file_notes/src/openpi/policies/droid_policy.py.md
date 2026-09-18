# `src/openpi/policies/droid_policy.py` 中文阅读说明

**定位：** 平台数据适配。

映射DROID相机、关节状态与夹爪信息，连接模型统一输入输出格式。

**建议读法：** 从示例字典到DroidInputs，检查图像布局、state拼接和输出动作截取。

**易错点：** 不同平台字段名和夹爪约定不同，不能直接把LIBERO字典传给DROID配置。

[注释源码](../../../../code/src/openpi/policies/droid_policy.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/droid_policy.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [make_droid_example](../../../../code/src/openpi/policies/droid_policy.py#L19) | 本函数位于“平台数据适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.random.randint → np.random.rand 追踪具体实现。 |
| [_parse_image](../../../../code/src/openpi/policies/droid_policy.py#L34) | 将浮点图像转换到uint8，并在需要时从CHW转为HWC；先确认源图像数值范围。 |
| [DroidInputs](../../../../code/src/openpi/policies/droid_policy.py#L45) | 映射DROID相机、关节状态与夹爪信息，连接模型统一输入输出格式。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [DroidInputs.__call__](../../../../code/src/openpi/policies/droid_policy.py#L53) | 按DROID字段取外部/腕部图像、关节和夹爪状态，组成模型统一输入。 |
| [DroidOutputs](../../../../code/src/openpi/policies/droid_policy.py#L97) | 映射DROID相机、关节状态与夹爪信息，连接模型统一输入输出格式。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [DroidOutputs.__call__](../../../../code/src/openpi/policies/droid_policy.py#L101) | 执行映射DROID相机、关节状态与夹爪信息，连接模型统一输入输出格式。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
