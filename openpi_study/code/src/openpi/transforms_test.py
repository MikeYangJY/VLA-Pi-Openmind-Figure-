# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：验证与示例｜用小样本和断言检查transforms相关行为，是理解预期输入输出的例子。
# 阅读顺序：先看test函数里的构造输入，再看被测调用和assert/数值比较。
# 重点边界：测试覆盖的是列出的行为，不等于完整机器人部署或任务成功率验证。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import numpy as np
import pytest

import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms


# 【test_repack_transform】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：transform(item) == {'a': {'b': 1}, 'd': 2}。
# 内部调用线索：_transforms.RepackTransform → transform（含分支中的调用，实际路径由条件决定）。
def test_repack_transform():
    transform = _transforms.RepackTransform(
        structure={
            "a": {"b": "b/c"},
            "d": "e/f",
        }
    )
    item = {"b": {"c": 1}, "e": {"f": 2}}
    assert transform(item) == {"a": {"b": 1}, "d": 2}


# 【test_delta_actions】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：np.all(transformed['state'] == np.array([1, 2, 3]))。
# 内部调用线索：np.array → _transforms.DeltaActions → transform → np.all（含分支中的调用，实际路径由条件决定）。
def test_delta_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.DeltaActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 2, 5], [5, 4, 7]]))


# 【test_delta_actions_noop】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：transform(item) is item。
# 内部调用线索：np.array → _transforms.DeltaActions → transform（含分支中的调用，实际路径由条件决定）。
def test_delta_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.DeltaActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.DeltaActions(mask=[True, False])
    assert transform(item) is item


# 【test_absolute_actions】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：np.all(transformed['state'] == np.array([1, 2, 3]))。
# 内部调用线索：np.array → _transforms.AbsoluteActions → transform → np.all（含分支中的调用，实际路径由条件决定）。
def test_absolute_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.AbsoluteActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 6, 5], [5, 8, 7]]))


# 【test_absolute_actions_noop】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：transform(item) is item。
# 内部调用线索：np.array → _transforms.AbsoluteActions → transform（含分支中的调用，实际路径由条件决定）。
def test_absolute_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.AbsoluteActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.AbsoluteActions(mask=[True, False])
    assert transform(item) is item


# 【test_make_bool_mask】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：_transforms.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)。
def test_make_bool_mask():
    assert _transforms.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert _transforms.make_bool_mask(2, 0, 2) == (True, True, True, True)


# 【test_tokenize_prompt】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：np.allclose(tok_prompt, data['tokenized_prompt'])。
# 内部调用线索：_tokenizer.PaligemmaTokenizer → _transforms.TokenizePrompt → transform → tokenizer.tokenize → np.allclose（含分支中的调用，实际路径由条件决定）。
def test_tokenize_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=12)
    transform = _transforms.TokenizePrompt(tokenizer)

    data = transform({"prompt": "Hello, world!"})

    tok_prompt, tok_mask = tokenizer.tokenize("Hello, world!")
    assert np.allclose(tok_prompt, data["tokenized_prompt"])
    assert np.allclose(tok_mask, data["tokenized_prompt_mask"])


# 【test_tokenize_no_prompt】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。数值比较或预期异常给出通过条件。
# 内部调用线索：_transforms.TokenizePrompt → _tokenizer.PaligemmaTokenizer → pytest.raises → transform（含分支中的调用，实际路径由条件决定）。
def test_tokenize_no_prompt():
    transform = _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer())

    with pytest.raises(ValueError, match="Prompt is required"):
        transform({})


# 【test_transform_dict】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：output == {'a': {'c': 1}}。
# 内部调用线索：_transforms.transform_dict → pytest.raises（含分支中的调用，实际路径由条件决定）。
def test_transform_dict():
    # Rename and remove keys.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a/b": "a/c", "a/c": None}, input)
    assert output == {"a": {"c": 1}}

    # Raises and error since the renamed key conflicts with an existing key.
    with pytest.raises(ValueError, match="Key 'a/c' already exists in output"):
        _transforms.transform_dict({"a/b": "a/c"}, input)

    # Full match is required and so nothing will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a": None}, input)
    assert output == input

    # The regex matches the entire key and so the entire input will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a.+": None}, input)
    assert output == {}

    # Replace keys using backreferences. All leaves named 'c' are replaced with 'd'.
    input = {"a": {"b": 1, "c": 1}, "b": {"c": 2}}
    output = _transforms.transform_dict({"(.+)/c": r"\1/d"}, input)
    assert output == {"a": {"b": 1, "d": 1}, "b": {"d": 2}}


# 【test_extract_prompt_from_task】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：data['prompt'] == 'Hello, world!'。
# 内部调用线索：_transforms.PromptFromLeRobotTask → transform → pytest.raises（含分支中的调用，实际路径由条件决定）。
def test_extract_prompt_from_task():
    transform = _transforms.PromptFromLeRobotTask({1: "Hello, world!"})

    data = transform({"task_index": 1})
    assert data["prompt"] == "Hello, world!"

    with pytest.raises(ValueError, match="task_index=2 not found in task mapping"):
        transform({"task_index": 2})
