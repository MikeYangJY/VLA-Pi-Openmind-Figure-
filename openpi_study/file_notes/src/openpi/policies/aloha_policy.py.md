# `src/openpi/policies/aloha_policy.py` 中文阅读说明

**定位：** 平台数据适配。

处理ALOHA双臂的关节方向、夹爪标定与图像，把公开硬件坐标连接到训练约定。

**建议读法：** 先看AlohaInputs/Outputs，再按调用读_decode_state、_encode_actions及夹爪转换。

**易错点：** 正负号与夹爪范围是物理含义，不是随便归一化；输入变换与输出逆变换必须一致。

[注释源码](../../../../code/src/openpi/policies/aloha_policy.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/aloha_policy.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [make_aloha_example](../../../../code/src/openpi/policies/aloha_policy.py#L19) | 本函数位于“平台数据适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.ones → np.random.randint 追踪具体实现。 |
| [AlohaInputs](../../../../code/src/openpi/policies/aloha_policy.py#L35) | 处理ALOHA双臂的关节方向、夹爪标定与图像，把公开硬件坐标连接到训练约定。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [AlohaInputs.__call__](../../../../code/src/openpi/policies/aloha_policy.py#L56) | 解码双臂ALOHA输入，并按配置转换关节方向与夹爪标定，再构造统一模型观测。 |
| [AlohaOutputs](../../../../code/src/openpi/policies/aloha_policy.py#L106) | 处理ALOHA双臂的关节方向、夹爪标定与图像，把公开硬件坐标连接到训练约定。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [AlohaOutputs.__call__](../../../../code/src/openpi/policies/aloha_policy.py#L117) | 把模型动作解码回ALOHA真实平台的坐标/夹爪约定；应与输入适配成对理解。 |
| [_joint_flip_mask](../../../../code/src/openpi/policies/aloha_policy.py#L125) | 本函数位于“平台数据适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.array 追踪具体实现。 |
| [_normalize](../../../../code/src/openpi/policies/aloha_policy.py#L133) | 本函数位于“平台数据适配”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。 |
| [_unnormalize](../../../../code/src/openpi/policies/aloha_policy.py#L140) | 本函数位于“平台数据适配”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。 |
| [_gripper_to_angular](../../../../code/src/openpi/policies/aloha_policy.py#L148) | 本函数位于“平台数据适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 _unnormalize → linear_to_radian → _normalize 追踪具体实现。 |
| [_gripper_to_angular.linear_to_radian](../../../../code/src/openpi/policies/aloha_policy.py#L162) | 本函数位于“平台数据适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.arcsin → np.clip 追踪具体实现。 |
| [_gripper_from_angular](../../../../code/src/openpi/policies/aloha_policy.py#L178) | 本函数位于“平台数据适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 _normalize 追踪具体实现。 |
| [_gripper_from_angular_inv](../../../../code/src/openpi/policies/aloha_policy.py#L194) | 本函数位于“平台数据适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 _unnormalize 追踪具体实现。 |
| [_decode_aloha](../../../../code/src/openpi/policies/aloha_policy.py#L204) | 本函数位于“平台数据适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.asarray → _decode_state → convert_image 追踪具体实现。 |
| [_decode_aloha.convert_image](../../../../code/src/openpi/policies/aloha_policy.py#L214) | 本函数位于“平台数据适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.asarray → np.issubdtype → (255 * img).astype 追踪具体实现。 |
| [_decode_state](../../../../code/src/openpi/policies/aloha_policy.py#L234) | 本函数位于“平台数据适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 _joint_flip_mask → _gripper_to_angular 追踪具体实现。 |
| [_encode_actions](../../../../code/src/openpi/policies/aloha_policy.py#L247) | 本函数位于“平台数据适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 _joint_flip_mask → _gripper_from_angular 追踪具体实现。 |
| [_encode_actions_inv](../../../../code/src/openpi/policies/aloha_policy.py#L259) | 本函数位于“平台数据适配”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 _joint_flip_mask → _gripper_from_angular_inv 追踪具体实现。 |
