# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：类型与形状｜用类型别名、运行时检查和PyTree比较表达张量结构约束。
# 阅读顺序：先看Array/Float等别名，再看typecheck与check_pytree_equality。
# 重点边界：标注中的b、ah、ad是维度名字，不是额外的数据变量；关闭检查不会改变模型公式。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import contextlib
import functools as ft
import inspect
from typing import TypeAlias, TypeVar, cast

import beartype
import jax
import jax._src.tree_util as private_tree_util
import jax.core
from jaxtyping import ArrayLike
from jaxtyping import Bool  # noqa: F401
from jaxtyping import DTypeLike  # noqa: F401
from jaxtyping import Float
from jaxtyping import Int  # noqa: F401
from jaxtyping import Key  # noqa: F401
from jaxtyping import Num  # noqa: F401
from jaxtyping import PyTree
from jaxtyping import Real  # noqa: F401
from jaxtyping import UInt8  # noqa: F401
from jaxtyping import config
from jaxtyping import jaxtyped
import jaxtyping._decorator
import torch

# patch jaxtyping to handle https://github.com/patrick-kidger/jaxtyping/issues/277.
# the problem is that custom PyTree nodes are sometimes initialized with arbitrary types (e.g., `jax.ShapeDtypeStruct`,
# `jax.Sharding`, or even <object>) due to JAX tracing operations. this patch skips typechecking when the stack trace
# contains `jax._src.tree_util`, which should only be the case during tree unflattening.
_original_check_dataclass_annotations = jaxtyping._decorator._check_dataclass_annotations  # noqa: SLF001
# Redefine Array to include both JAX arrays and PyTorch tensors
Array = jax.Array | torch.Tensor


# 【_check_dataclass_annotations】本函数位于“类型与形状”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 any → frame.frame.f_globals.get → inspect.stack 追踪具体实现。
# 输入接口：typechecker。
# 返回值可从这里追踪：_original_check_dataclass_annotations(self, typechecker) / None。
# 内部调用线索：any → frame.frame.f_globals.get → inspect.stack → _original_check_dataclass_annotations（含分支中的调用，实际路径由条件决定）。
def _check_dataclass_annotations(self, typechecker):
    if not any(
        frame.frame.f_globals.get("__name__") in {"jax._src.tree_util", "flax.nnx.transforms.compilation"}
        for frame in inspect.stack()
    ):
        return _original_check_dataclass_annotations(self, typechecker)
    return None


jaxtyping._decorator._check_dataclass_annotations = _check_dataclass_annotations  # noqa: SLF001

KeyArrayLike: TypeAlias = jax.typing.ArrayLike
Params: TypeAlias = PyTree[Float[ArrayLike, "..."]]

T = TypeVar("T")


# runtime type-checking decorator
# 【typecheck】本函数位于“类型与形状”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 cast → ft.partial(jaxtyped, typechecker=beartype.beartype) → ft.partial 追踪具体实现。
# 输入接口：t:T。
# 返回类型：T；类型/shape约定需与调用方配套。
# 内部调用线索：cast → ft.partial(jaxtyped, typechecker=beartype.beartype) → ft.partial（含分支中的调用，实际路径由条件决定）。
def typecheck(t: T) -> T:
    return cast(T, ft.partial(jaxtyped, typechecker=beartype.beartype)(t))


# 【disable_typechecking】本函数位于“类型与形状”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 config.update 追踪具体实现。
@contextlib.contextmanager
def disable_typechecking():
    initial = config.jaxtyping_disable
    config.update("jaxtyping_disable", True)  # noqa: FBT003
    yield
    config.update("jaxtyping_disable", initial)


# 【check_pytree_equality】比较嵌套参数树的路径，并按选项检查形状和dtype，防止错误checkpoint静默加载。
# 输入接口：expected:PyTree；got:PyTree；check_shapes:bool；check_dtypes:bool。
# 内部调用线索：private_tree_util.equality_errors → ValueError → '\n'.join → jax.tree_util.keystr → jax.tree_util.tree_map_with_path（含分支中的调用，实际路径由条件决定）。
def check_pytree_equality(*, expected: PyTree, got: PyTree, check_shapes: bool = False, check_dtypes: bool = False):
    """Checks that two PyTrees have the same structure and optionally checks shapes and dtypes. Creates a much nicer
    error message than if `jax.tree.map` is naively used on PyTrees with different structures.
    """

    if errors := list(private_tree_util.equality_errors(expected, got)):
        raise ValueError(
            "PyTrees have different structure:\n"
            + (
                "\n".join(
                    f"   - at keypath '{jax.tree_util.keystr(path)}': expected {thing1}, got {thing2}, so {explanation}.\n"
                    for path, thing1, thing2, explanation in errors
                )
            )
        )

    if check_shapes or check_dtypes:

        # 【check_pytree_equality.check】本函数位于“类型与形状”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 ValueError → jax.tree_util.keystr 追踪具体实现。
        # 输入接口：kp；x；y。
        # 内部调用线索：ValueError → jax.tree_util.keystr（含分支中的调用，实际路径由条件决定）。
        def check(kp, x, y):
            if check_shapes and x.shape != y.shape:
                raise ValueError(f"Shape mismatch at {jax.tree_util.keystr(kp)}: expected {x.shape}, got {y.shape}")

            if check_dtypes and x.dtype != y.dtype:
                raise ValueError(f"Dtype mismatch at {jax.tree_util.keystr(kp)}: expected {x.dtype}, got {y.dtype}")

        jax.tree_util.tree_map_with_path(check, expected, got)
