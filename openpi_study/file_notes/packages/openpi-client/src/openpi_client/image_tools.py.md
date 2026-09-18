# `packages/openpi-client/src/openpi_client/image_tools.py` 中文阅读说明

**定位：** 客户端图像工具。

在CPU上把图像转换到uint8并等比例缩放补边，减少客户端依赖。

**建议读法：** convert_to_uint8规范数值；resize_with_pad处理尺寸；PIL执行实际缩放。

**易错点：** 模型端与客户端两处图像处理都要核对范围及通道顺序。

[注释源码](../../../../../code/packages/openpi-client/src/openpi_client/image_tools.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/packages/openpi-client/src/openpi_client/image_tools.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [convert_to_uint8](../../../../../code/packages/openpi-client/src/openpi_client/image_tools.py#L15) | 本函数位于“客户端图像工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.issubdtype → (255 * img).astype 追踪具体实现。 |
| [resize_with_pad](../../../../../code/packages/openpi-client/src/openpi_client/image_tools.py#L29) | 保持图像长宽比缩放，再用padding补到目标尺寸，而非直接把物体拉伸变形。 |
| [_resize_with_pad_pil](../../../../../code/packages/openpi-client/src/openpi_client/image_tools.py#L56) | 本函数位于“客户端图像工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 max → image.resize → Image.new 追踪具体实现。 |
