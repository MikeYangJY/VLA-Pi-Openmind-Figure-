# `src/openpi/policies/libero_policy.py` 中文阅读说明

**定位：** 入门数据适配。

把LIBERO图像、状态、指令改成模型统一格式，再把输出裁回7维环境动作。

**建议读法：** 先看make_libero_example，再看LiberoInputs.__call__，最后看LiberoOutputs。

**易错点：** 缺相机时用占位图和mask；模型补齐到32维，返回环境时只取实际动作维数。

[注释源码](../../../../code/src/openpi/policies/libero_policy.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/libero_policy.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [make_libero_example](../../../../code/src/openpi/policies/libero_policy.py#L19) | 本函数位于“入门数据适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.random.rand → np.random.randint 追踪具体实现。 |
| [_parse_image](../../../../code/src/openpi/policies/libero_policy.py#L33) | 将浮点图像转换到uint8，并在需要时从CHW转为HWC；先确认源图像数值范围。 |
| [LiberoInputs](../../../../code/src/openpi/policies/libero_policy.py#L44) | 把LIBERO图像、状态、指令改成模型统一格式，再把输出裁回7维环境动作。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [LiberoInputs.__call__](../../../../code/src/openpi/policies/libero_policy.py#L60) | 统一LIBERO的图像布局和键名，补缺失相机并设置mask，保留状态、训练动作和指令。 |
| [LiberoOutputs](../../../../code/src/openpi/policies/libero_policy.py#L106) | 把LIBERO图像、状态、指令改成模型统一格式，再把输出裁回7维环境动作。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [LiberoOutputs.__call__](../../../../code/src/openpi/policies/libero_policy.py#L117) | 去掉统一动作空间的padding，只返回LIBERO需要的前7维动作。 |
