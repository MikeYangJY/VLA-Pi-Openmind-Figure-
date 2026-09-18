# `src/openpi/shared/image_tools.py` 中文阅读说明

**定位：** 图像尺寸工具。

在JAX/PyTorch中等比例缩放图像，再补边到目标尺寸。

**建议读法：** 按输入尺寸计算缩放比与padding，resize后补边。

**易错点：** 保持长宽比不等于直接拉伸；这会影响物体几何外观。

[注释源码](../../../../code/src/openpi/shared/image_tools.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/shared/image_tools.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [resize_with_pad](../../../../code/src/openpi/shared/image_tools.py#L23) | 保持图像长宽比缩放，再用padding补到目标尺寸，而非直接把物体拉伸变形。 |
| [resize_with_pad_torch](../../../../code/src/openpi/shared/image_tools.py#L69) | 本函数位于“图像尺寸工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 images.dim → images.unsqueeze → images.permute 追踪具体实现。 |
