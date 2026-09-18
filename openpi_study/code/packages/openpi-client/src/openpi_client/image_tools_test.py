# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：验证与示例｜用小样本和断言检查image_tools相关行为，是理解预期输入输出的例子。
# 阅读顺序：先看test函数里的构造输入，再看被测调用和assert/数值比较。
# 重点边界：测试覆盖的是列出的行为，不等于完整机器人部署或任务成功率验证。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import numpy as np

import openpi_client.image_tools as image_tools


# 【test_resize_with_pad_shapes】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：resized_images.shape == (2, height, width, 3)。
# 内部调用线索：np.zeros → image_tools.resize_with_pad → np.all（含分支中的调用，实际路径由条件决定）。
def test_resize_with_pad_shapes():
    # Test case 1: Resize image with larger dimensions
    images = np.zeros((2, 10, 10, 3), dtype=np.uint8)  # Input images of shape (batch_size, height, width, channels)
    height = 20
    width = 20
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (2, height, width, 3)
    assert np.all(resized_images == 0)

    # Test case 2: Resize image with smaller dimensions
    images = np.zeros((3, 30, 30, 3), dtype=np.uint8)
    height = 15
    width = 15
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (3, height, width, 3)
    assert np.all(resized_images == 0)

    # Test case 3: Resize image with the same dimensions
    images = np.zeros((1, 50, 50, 3), dtype=np.uint8)
    height = 50
    width = 50
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (1, height, width, 3)
    assert np.all(resized_images == 0)

    # Test case 3: Resize image with odd-numbered padding
    images = np.zeros((1, 256, 320, 3), dtype=np.uint8)
    height = 60
    width = 80
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (1, height, width, 3)
    assert np.all(resized_images == 0)
