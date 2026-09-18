# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：验证与示例｜用小样本和断言检查tokenizer相关行为，是理解预期输入输出的例子。
# 阅读顺序：先看test函数里的构造输入，再看被测调用和assert/数值比较。
# 重点边界：测试覆盖的是列出的行为，不等于完整机器人部署或任务成功率验证。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import numpy as np

from openpi.models import tokenizer as _tokenizer


# 【test_tokenize】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：tokens.shape == (10,)。
# 内部调用线索：_tokenizer.PaligemmaTokenizer → tokenizer.tokenize（含分支中的调用，实际路径由条件决定）。
def test_tokenize():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=10)
    tokens, masks = tokenizer.tokenize("Hello, world!")

    assert tokens.shape == (10,)
    assert masks.shape == (10,)


# 【test_fast_tokenizer】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：tokens.shape == (256,)。
# 内部调用线索：np.random.rand(5).astype → np.random.rand → np.random.rand(3, 2).astype → _tokenizer.FASTTokenizer → tokenizer.tokenize（含分支中的调用，实际路径由条件决定）。
def test_fast_tokenizer():
    prompt = "Hello, world!"
    state = np.random.rand(5).astype(np.float32)
    action = np.random.rand(3, 2).astype(np.float32)
    tokenizer = _tokenizer.FASTTokenizer(max_len=256)
    tokens, token_masks, ar_masks, loss_masks = tokenizer.tokenize(prompt, state, action)

    assert tokens.shape == (256,)
    assert token_masks.shape == (256,)
    assert ar_masks.shape == (256,)
    assert loss_masks.shape == (256,)

    act = tokenizer.extract_actions(tokens, 3, 2)
    assert act.shape == (3, 2)
