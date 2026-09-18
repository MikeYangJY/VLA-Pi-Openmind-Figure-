# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：验证与示例｜用小样本和断言检查normalize相关行为，是理解预期输入输出的例子。
# 阅读顺序：先看test函数里的构造输入，再看被测调用和assert/数值比较。
# 重点边界：测试覆盖的是列出的行为，不等于完整机器人部署或任务成功率验证。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import numpy as np

import openpi.shared.normalize as normalize


# 【test_normalize_update】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：np.allclose(results.mean, np.mean(arr, axis=0))。
# 内部调用线索：np.arange(12).reshape → np.arange → normalize.RunningStats → stats.update → stats.get_statistics（含分支中的调用，实际路径由条件决定）。
def test_normalize_update():
    arr = np.arange(12).reshape(4, 3)  # 4 vectors of length 3

    stats = normalize.RunningStats()
    for i in range(len(arr)):
        stats.update(arr[i : i + 1])  # Update with one vector at a time
    results = stats.get_statistics()

    assert np.allclose(results.mean, np.mean(arr, axis=0))
    assert np.allclose(results.std, np.std(arr, axis=0))


# 【test_serialize_deserialize】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：np.allclose(norm_stats['test'].mean, norm_stats2['test'].mean)。
# 内部调用线索：normalize.RunningStats → stats.update → np.arange(12).reshape → np.arange → stats.get_statistics（含分支中的调用，实际路径由条件决定）。
def test_serialize_deserialize():
    stats = normalize.RunningStats()
    stats.update(np.arange(12).reshape(4, 3))  # 4 vectors of length 3

    norm_stats = {"test": stats.get_statistics()}
    norm_stats2 = normalize.deserialize_json(normalize.serialize_json(norm_stats))
    assert np.allclose(norm_stats["test"].mean, norm_stats2["test"].mean)
    assert np.allclose(norm_stats["test"].std, norm_stats2["test"].std)


# 【test_multiple_batch_dimensions】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：np.allclose(results.mean, expected_mean)。
# 内部调用线索：np.random.rand → normalize.RunningStats → stats.update → stats.get_statistics → arr.reshape（含分支中的调用，实际路径由条件决定）。
def test_multiple_batch_dimensions():
    # Test with multiple batch dimensions: (2, 3, 4) where 4 is vector dimension
    batch_shape = (2, 3, 4)
    arr = np.random.rand(*batch_shape)

    stats = normalize.RunningStats()
    stats.update(arr)  # Should handle (2, 3, 4) -> reshape to (6, 4)
    results = stats.get_statistics()

    # Flatten batch dimensions and compute expected stats
    flattened = arr.reshape(-1, arr.shape[-1])  # (6, 4)
    expected_mean = np.mean(flattened, axis=0)
    expected_std = np.std(flattened, axis=0)

    assert np.allclose(results.mean, expected_mean)
    assert np.allclose(results.std, expected_std)
