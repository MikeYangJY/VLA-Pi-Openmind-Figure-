# `src/openpi/transforms_test.py` 中文阅读说明

**定位：** 验证与示例。

用小样本和断言检查transforms相关行为，是理解预期输入输出的例子。

**建议读法：** 先看test函数里的构造输入，再看被测调用和assert/数值比较。

**易错点：** 测试覆盖的是列出的行为，不等于完整机器人部署或任务成功率验证。

[注释源码](../../../code/src/openpi/transforms_test.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/transforms_test.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [test_repack_transform](../../../code/src/openpi/transforms_test.py#L16) | 验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：transform(item) == {'a': {'b': 1}, 'd': 2}。 |
| [test_delta_actions](../../../code/src/openpi/transforms_test.py#L29) | 验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：np.all(transformed['state'] == np.array([1, 2, 3]))。 |
| [test_delta_actions_noop](../../../code/src/openpi/transforms_test.py#L41) | 验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：transform(item) is item。 |
| [test_absolute_actions](../../../code/src/openpi/transforms_test.py#L56) | 验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：np.all(transformed['state'] == np.array([1, 2, 3]))。 |
| [test_absolute_actions_noop](../../../code/src/openpi/transforms_test.py#L68) | 验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：transform(item) is item。 |
| [test_make_bool_mask](../../../code/src/openpi/transforms_test.py#L82) | 验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：_transforms.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)。 |
| [test_tokenize_prompt](../../../code/src/openpi/transforms_test.py#L89) | 验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：np.allclose(tok_prompt, data['tokenized_prompt'])。 |
| [test_tokenize_no_prompt](../../../code/src/openpi/transforms_test.py#L102) | 验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。数值比较或预期异常给出通过条件。 |
| [test_transform_dict](../../../code/src/openpi/transforms_test.py#L111) | 验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：output == {'a': {'c': 1}}。 |
| [test_extract_prompt_from_task](../../../code/src/openpi/transforms_test.py#L139) | 验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：data['prompt'] == 'Hello, world!'。 |
