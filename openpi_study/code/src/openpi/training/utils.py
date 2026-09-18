# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：训练状态容器｜用TrainState集中保存步数、参数、优化器状态、模型结构及EMA。
# 阅读顺序：对照scripts/train.py中状态如何创建、更新、保存；打印工具用于检查数组树形状。
# 重点边界：model_def是结构而非训练好的数值权重；EMA是一份平滑参数。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
from collections.abc import Callable
from typing import Any

from flax import nnx
from flax import struct
import jax
import optax

from openpi.models import model as _model
from openpi.shared import array_typing as at


# 【TrainState】训练时需保存的全部状态，不只是神经网络参数。
@at.typecheck
@struct.dataclass
class TrainState:
    step: at.Int[at.ArrayLike, ""]
    params: nnx.State
    model_def: nnx.GraphDef[_model.BaseModel]
    opt_state: optax.OptState
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    # 字段含义：指数滑动平均参数的平滑系数，None表示不使用EMA。
    ema_decay: float | None = struct.field(pytree_node=False)
    ema_params: nnx.State | None = None


# 【tree_to_info】本函数位于“训练状态容器”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 jax.tree_util.tree_flatten_with_path → '\n'.join → jax.tree_util.keystr 追踪具体实现。
# 输入接口：tree:at.PyTree；interp_func:Callable[[Any], str]。
# 返回类型：str；类型/shape约定需与调用方配套。
# 内部调用线索：jax.tree_util.tree_flatten_with_path → '\n'.join → jax.tree_util.keystr → interp_func（含分支中的调用，实际路径由条件决定）。
@at.typecheck
def tree_to_info(tree: at.PyTree, interp_func: Callable[[Any], str] = str) -> str:
    """Converts a PyTree into a human-readable string for logging. Optionally, `interp_func` can be provided to convert
    the leaf values to more meaningful strings.
    """
    tree, _ = jax.tree_util.tree_flatten_with_path(tree)
    return "\n".join(f"{jax.tree_util.keystr(path)}: {interp_func(value)}" for path, value in tree)


# 【array_tree_to_info】本函数位于“训练状态容器”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 tree_to_info 追踪具体实现。
# 输入接口：tree:at.PyTree。
# 返回类型：str；类型/shape约定需与调用方配套。
@at.typecheck
def array_tree_to_info(tree: at.PyTree) -> str:
    """Converts a PyTree of arrays into a human-readable string for logging."""
    return tree_to_info(tree, lambda x: f"{x.shape}@{x.dtype}")
