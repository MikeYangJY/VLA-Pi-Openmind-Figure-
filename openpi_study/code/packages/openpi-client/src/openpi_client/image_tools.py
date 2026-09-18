# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：客户端图像工具｜在CPU上把图像转换到uint8并等比例缩放补边，减少客户端依赖。
# 阅读顺序：convert_to_uint8规范数值；resize_with_pad处理尺寸；PIL执行实际缩放。
# 重点边界：模型端与客户端两处图像处理都要核对范围及通道顺序。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import numpy as np
from PIL import Image


# 【convert_to_uint8】本函数位于“客户端图像工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.issubdtype → (255 * img).astype 追踪具体实现。
# 输入接口：img:np.ndarray。
# 返回类型：np.ndarray；类型/shape约定需与调用方配套。
# 内部调用线索：np.issubdtype → (255 * img).astype（含分支中的调用，实际路径由条件决定）。
def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Converts an image to uint8 if it is a float image.

    This is important for reducing the size of the image when sending it over the network.
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


# 【resize_with_pad】保持图像长宽比缩放，再用padding补到目标尺寸，而非直接把物体拉伸变形。
# 输入接口：images:np.ndarray；height:int；width:int；method。
# 返回类型：np.ndarray；类型/shape约定需与调用方配套。
# 内部调用线索：images.reshape → np.stack → _resize_with_pad_pil → Image.fromarray → resized.reshape（含分支中的调用，实际路径由条件决定）。
def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


# 【_resize_with_pad_pil】本函数位于“客户端图像工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 max → image.resize → Image.new 追踪具体实现。
# 输入接口：image:Image.Image；height:int；width:int；method:int。
# 返回类型：Image.Image；类型/shape约定需与调用方配套。
# 内部调用线索：max → image.resize → Image.new → zero_image.paste（含分支中的调用，实际路径由条件决定）。
def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image
