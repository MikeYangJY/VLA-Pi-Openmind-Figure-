# `src/openpi/transforms.py` 中文阅读说明

**定位：** 数据主线。

一组可组合的小变换，处理字段映射、图像、归一化、动作相对化、token与补维。

**建议读法：** 沿Group→Repack→Normalize→TokenizePrompt→PadStatesAndActions读；输出按反向语义还原。FAST专属类可跳过。

**易错点：** 变换顺序影响物理含义；动作时间维与动作坐标维不能混淆。

[注释源码](../../../code/src/openpi/transforms.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/transforms.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [DataTransformFn](../../../code/src/openpi/transforms.py#L31) | 一组可组合的小变换，处理字段映射、图像、归一化、动作相对化、token与补维。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [DataTransformFn.__call__](../../../code/src/openpi/transforms.py#L35) | 执行一组可组合的小变换，处理字段映射、图像、归一化、动作相对化、token与补维。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [Group](../../../code/src/openpi/transforms.py#L51) | 输入/输出变换的有序集合。 |
| [Group.push](../../../code/src/openpi/transforms.py#L63) | 输入变换追加到末尾，输出逆变换放到开头；这样嵌套的数据表示可以按相反顺序还原。 |
| [CompositeTransform](../../../code/src/openpi/transforms.py#L78) | 一组可组合的小变换，处理字段映射、图像、归一化、动作相对化、token与补维。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [CompositeTransform.__call__](../../../code/src/openpi/transforms.py#L86) | 依次调用每个transform，并把上一步字典交给下一步；顺序就是数据管线的语义。 |
| [compose](../../../code/src/openpi/transforms.py#L95) | 本函数位于“数据主线”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 CompositeTransform 追踪具体实现。 |
| [RepackTransform](../../../code/src/openpi/transforms.py#L102) | 按给定键路径映射重组样本。 |
| [RepackTransform.__call__](../../../code/src/openpi/transforms.py#L125) | 先展平原字典，再按structure映射取值并组回目标结构；主要重命名/重排，不做数值学习。 |
| [InjectDefaultPrompt](../../../code/src/openpi/transforms.py#L132) | 一组可组合的小变换，处理字段映射、图像、归一化、动作相对化、token与补维。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [InjectDefaultPrompt.__call__](../../../code/src/openpi/transforms.py#L138) | 只有输入没有prompt且配置给了默认指令时才补入，不覆盖已有用户指令。 |
| [Normalize](../../../code/src/openpi/transforms.py#L146) | 把状态和动作映射到统一数值尺度。 |
| [Normalize.__post_init__](../../../code/src/openpi/transforms.py#L155) | 本函数位于“数据主线”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 _assert_quantile_stats 追踪具体实现。 |
| [Normalize.__call__](../../../code/src/openpi/transforms.py#L162) | 执行把状态和动作映射到统一数值尺度。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [Normalize._normalize](../../../code/src/openpi/transforms.py#L176) | 用训练数据的mean/std做z-score；1e-6避免除零，不是可训练参数。 |
| [Normalize._normalize_quantile](../../../code/src/openpi/transforms.py#L184) | 用q01/q99把数据线性映射到约[-1,1]；超出分位数的数据仍可能超出这个区间。 |
| [Unnormalize](../../../code/src/openpi/transforms.py#L194) | 把输出恢复到真实动作尺度。 |
| [Unnormalize.__post_init__](../../../code/src/openpi/transforms.py#L201) | 本函数位于“数据主线”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 _assert_quantile_stats 追踪具体实现。 |
| [Unnormalize.__call__](../../../code/src/openpi/transforms.py#L208) | 执行把输出恢复到真实动作尺度。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [Unnormalize._unnormalize](../../../code/src/openpi/transforms.py#L223) | 逆转z-score，把模型输出恢复到训练动作的数值单位；补齐维数采用中性统计。 |
| [Unnormalize._unnormalize_quantile](../../../code/src/openpi/transforms.py#L231) | 逆转分位数映射；真实维度按q01/q99恢复，额外padding维度单独保留。 |
| [ResizeImages](../../../code/src/openpi/transforms.py#L242) | 一组可组合的小变换，处理字段映射、图像、归一化、动作相对化、token与补维。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [ResizeImages.__call__](../../../code/src/openpi/transforms.py#L250) | 逐相机等比例缩放并补边到统一输入分辨率。 |
| [SubsampleActions](../../../code/src/openpi/transforms.py#L257) | 一组可组合的小变换，处理字段映射、图像、归一化、动作相对化、token与补维。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [SubsampleActions.__call__](../../../code/src/openpi/transforms.py#L263) | 沿时间维按stride取未来动作，不是减少动作坐标数量。 |
| [DeltaActions](../../../code/src/openpi/transforms.py#L270) | 把指定坐标的绝对动作变成相对当前状态的增量。 |
| [DeltaActions.__call__](../../../code/src/openpi/transforms.py#L282) | 只在mask选中的动作维上减去当前state，得到相对目标；夹爪等未选维度不变。 |
| [AbsoluteActions](../../../code/src/openpi/transforms.py#L297) | 把增量动作还原成对应绝对目标。 |
| [AbsoluteActions.__call__](../../../code/src/openpi/transforms.py#L309) | 将选中维度的相对动作加回当前state，恢复环境期望的绝对目标。 |
| [TokenizePrompt](../../../code/src/openpi/transforms.py#L324) | 一组可组合的小变换，处理字段映射、图像、归一化、动作相对化、token与补维。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [TokenizePrompt.__call__](../../../code/src/openpi/transforms.py#L333) | 取prompt并按配置决定是否把state也交给PaligemmaTokenizer，添加tokenized_prompt与mask。 |
| [TokenizeFASTInputs](../../../code/src/openpi/transforms.py#L352) | 一组可组合的小变换，处理字段映射、图像、归一化、动作相对化、token与补维。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [TokenizeFASTInputs.__call__](../../../code/src/openpi/transforms.py#L359) | 执行一组可组合的小变换，处理字段映射、图像、归一化、动作相对化、token与补维。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [ExtractFASTActions](../../../code/src/openpi/transforms.py#L379) | 一组可组合的小变换，处理字段映射、图像、归一化、动作相对化、token与补维。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [ExtractFASTActions.__call__](../../../code/src/openpi/transforms.py#L390) | 执行一组可组合的小变换，处理字段映射、图像、归一化、动作相对化、token与补维。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [PromptFromLeRobotTask](../../../code/src/openpi/transforms.py#L404) | 一组可组合的小变换，处理字段映射、图像、归一化、动作相对化、token与补维。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [PromptFromLeRobotTask.__call__](../../../code/src/openpi/transforms.py#L414) | 把数据集任务索引映射成语言指令，让每个训练样本带prompt。 |
| [PadStatesAndActions](../../../code/src/openpi/transforms.py#L427) | 一组可组合的小变换，处理字段映射、图像、归一化、动作相对化、token与补维。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [PadStatesAndActions.__call__](../../../code/src/openpi/transforms.py#L435) | 把最后一维补到模型action_dim；训练含actions时一起补，推理只有state也可以处理。 |
| [flatten_dict](../../../code/src/openpi/transforms.py#L445) | 本函数位于“数据主线”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 traverse_util.flatten_dict 追踪具体实现。 |
| [unflatten_dict](../../../code/src/openpi/transforms.py#L453) | 本函数位于“数据主线”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 traverse_util.unflatten_dict 追踪具体实现。 |
| [transform_dict](../../../code/src/openpi/transforms.py#L462) | 本函数位于“数据主线”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 flatten_dict → re.compile → patterns.items 追踪具体实现。 |
| [apply_tree](../../../code/src/openpi/transforms.py#L520) | 按参考统计/结构逐叶应用函数，可严格检查缺失字段；用于批量归一化嵌套数据。 |
| [apply_tree.transform](../../../code/src/openpi/transforms.py#L529) | 本函数位于“数据主线”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 fn 追踪具体实现。 |
| [pad_to_dim](../../../code/src/openpi/transforms.py#L545) | 沿指定维度补常数，使输入适配统一长度；shape相容不代表不同机器人的动作语义相同。 |
| [make_bool_mask](../../../code/src/openpi/transforms.py#L557) | 把正数段转为True、负数段转为False，简洁指定哪些动作维需要相对化等处理。 |
| [_assert_quantile_stats](../../../code/src/openpi/transforms.py#L583) | 本函数位于“数据主线”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 flatten_dict(norm_stats).items → flatten_dict → ValueError 追踪具体实现。 |
