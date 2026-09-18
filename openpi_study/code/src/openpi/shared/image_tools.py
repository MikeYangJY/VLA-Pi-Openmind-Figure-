# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：图像尺寸工具｜在JAX/PyTorch中等比例缩放图像，再补边到目标尺寸。
# 阅读顺序：按输入尺寸计算缩放比与padding，resize后补边。
# 重点边界：保持长宽比不等于直接拉伸；这会影响物体几何外观。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import functools

import jax
import jax.numpy as jnp
import torch
import torch.nn.functional as F  # noqa: N812

import openpi.shared.array_typing as at


# 【resize_with_pad】保持图像长宽比缩放，再用padding补到目标尺寸，而非直接把物体拉伸变形。
# 输入接口：images:at.UInt8[at.Array, '*b h w c'] | at.Float[at.Array…；height:int；width:int；method:jax.image.ResizeMethod。
# 返回类型：at.UInt8[at.Array, '*b {height} {width} c'] | at.Float[at.Array, '*b {height} {width} c']；类型/shape约定需与调用方配套。
# 内部调用线索：max → jax.image.resize → jnp.round(resized_images).clip(0, 255).astype → jnp.round(resized_images).clip → jnp.round（含分支中的调用，实际路径由条件决定）。
@functools.partial(jax.jit, static_argnums=(1, 2, 3))
@at.typecheck
def resize_with_pad(
    images: at.UInt8[at.Array, "*b h w c"] | at.Float[at.Array, "*b h w c"],
    height: int,
    width: int,
    method: jax.image.ResizeMethod = jax.image.ResizeMethod.LINEAR,
) -> at.UInt8[at.Array, "*b {height} {width} c"] | at.Float[at.Array, "*b {height} {width} c"]:
    """Replicates tf.image.resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].
    """
    has_batch_dim = images.ndim == 4
    if not has_batch_dim:
        images = images[None]  # type: ignore
    cur_height, cur_width = images.shape[1:3]
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_images = jax.image.resize(
        images, (images.shape[0], resized_height, resized_width, images.shape[3]), method=method
    )
    if images.dtype == jnp.uint8:
        # round from float back to uint8
        resized_images = jnp.round(resized_images).clip(0, 255).astype(jnp.uint8)
    elif images.dtype == jnp.float32:
        resized_images = resized_images.clip(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w
    padded_images = jnp.pad(
        resized_images,
        ((0, 0), (pad_h0, pad_h1), (pad_w0, pad_w1), (0, 0)),
        constant_values=0 if images.dtype == jnp.uint8 else -1.0,
    )

    if not has_batch_dim:
        padded_images = padded_images[0]
    return padded_images


# 【resize_with_pad_torch】本函数位于“图像尺寸工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 images.dim → images.unsqueeze → images.permute 追踪具体实现。
# 输入接口：images:torch.Tensor；height:int；width:int；mode:str。
# 返回类型：torch.Tensor；类型/shape约定需与调用方配套。
# 内部调用线索：images.dim → images.unsqueeze → images.permute → max → F.interpolate（含分支中的调用，实际路径由条件决定）。
def resize_with_pad_torch(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input
    """
    # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        # Convert to channels-first for torch operations
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

    batch_size, channels, cur_height, cur_width = images.shape

    # Calculate resize ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize
    resized_images = F.interpolate(
        images, size=(resized_height, resized_width), mode=mode, align_corners=False if mode == "bilinear" else None
    )

    # Handle dtype-specific clipping
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad
    constant_value = 0 if images.dtype == torch.uint8 else -1.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # left, right, top, bottom
        mode="constant",
        value=constant_value,
    )

    # Convert back to original format if needed
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]
        if batch_size == 1 and images.shape[0] == 1:
            padded_images = padded_images.squeeze(0)  # Remove batch dimension if it was added

    return padded_images
