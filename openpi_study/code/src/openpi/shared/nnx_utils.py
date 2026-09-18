# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：JAX桥接工具｜把带状态的NNX模块安全包装成JIT函数，并提供参数路径筛选与映射。
# 阅读顺序：module_jit拆结构与状态再编译/合并；PathRegex选参数；state_map处理选中的叶子。
# 重点边界：Python对象有状态，JIT偏好显式数据流；这就是这些拆分/合并步骤存在的原因。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
from collections.abc import Callable
import dataclasses
import functools
import inspect
import re
from typing import Any, ParamSpec, TypeVar

import flax.nnx as nnx
import jax

P = ParamSpec("P")
R = TypeVar("R")


# 【module_jit】把NNX模块结构和状态显式分开，再用JAX编译包装方法；返回值保持模块方法使用方式。
# 输入接口：meth:Callable[P, R]。
# 返回类型：Callable[P, R]；类型/shape约定需与调用方配套。
# 内部调用线索：inspect.ismethod → isinstance → ValueError → nnx.split → jax.jit（含分支中的调用，实际路径由条件决定）。
def module_jit(meth: Callable[P, R], *jit_args, **jit_kwargs) -> Callable[P, R]:
    """A higher-order function to JIT-compile `nnx.Module` methods, freezing the module's state in the process.

    Why not `nnx.jit`? For some reason, naively applying `nnx.jit` to `nnx.Module` methods, bound or unbound, uses much
    more memory than necessary. I'm guessing it has something to do with the fact that it must keep track of module
    mutations. Also, `nnx.jit` has some inherent overhead compared to a standard `jax.jit`, since every call must
    traverse the NNX module graph. See https://github.com/google/flax/discussions/4224 for details.

    `module_jit` is an alternative that avoids these issues by freezing the module's state. The function returned by
    `module_jit` acts exactly like the original method, except that the state of the module is frozen to whatever it was
    when `module_jit` was called. Mutations to the module within `meth` are still allowed, but they will be discarded
    after the method call completes.
    """
    if not (inspect.ismethod(meth) and isinstance(meth.__self__, nnx.Module)):
        raise ValueError("module_jit must only be used on bound methods of nnx.Modules.")

    graphdef, state = nnx.split(meth.__self__)

    # 【module_jit.fun】本函数位于“JAX桥接工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 nnx.merge → meth.__func__ 追踪具体实现。
    # 输入接口：state:nnx.State。
    # 返回类型：R；类型/shape约定需与调用方配套。
    # 内部调用线索：nnx.merge → meth.__func__（含分支中的调用，实际路径由条件决定）。
    def fun(state: nnx.State, *args: P.args, **kwargs: P.kwargs) -> R:
        module = nnx.merge(graphdef, state)
        return meth.__func__(module, *args, **kwargs)

    jitted_fn = jax.jit(fun, *jit_args, **jit_kwargs)

    # 【module_jit.wrapper】本函数位于“JAX桥接工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 jitted_fn 追踪具体实现。
    # 返回类型：R；类型/shape约定需与调用方配套。
    @functools.wraps(meth)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return jitted_fn(state, *args, **kwargs)

    return wrapper


# 【PathRegex】把带状态的NNX模块安全包装成JIT函数，并提供参数路径筛选与映射。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclasses.dataclass(frozen=True)
class PathRegex:
    """NNX Filter that matches paths using a regex.

    By default, paths are joined with a `/` separator. This can be overridden by setting the `sep` argument.
    """

    pattern: str | re.Pattern
    sep: str = "/"

    # 【PathRegex.__post_init__】本函数位于“JAX桥接工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 isinstance → object.__setattr__ → re.compile 追踪具体实现。
    # 内部调用线索：isinstance → object.__setattr__ → re.compile（含分支中的调用，实际路径由条件决定）。
    def __post_init__(self):
        if not isinstance(self.pattern, re.Pattern):
            object.__setattr__(self, "pattern", re.compile(self.pattern))

    # 【PathRegex.__call__】执行把带状态的NNX模块安全包装成JIT函数，并提供参数路径筛选与映射。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。
    # 输入接口：path:nnx.filterlib.PathParts；x:Any。
    # 返回类型：bool；类型/shape约定需与调用方配套。
    # 内部调用线索：self.sep.join → isinstance → self.pattern.fullmatch（含分支中的调用，实际路径由条件决定）。
    def __call__(self, path: nnx.filterlib.PathParts, x: Any) -> bool:
        joined_path = self.sep.join(str(x) for x in path)
        assert isinstance(self.pattern, re.Pattern)
        return self.pattern.fullmatch(joined_path) is not None


# 【state_map】本函数位于“JAX桥接工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 set → state.filter(filter).flat_state → state.filter 追踪具体实现。
# 输入接口：state:nnx.State；filter:nnx.filterlib.Filter；fn:Callable[[Any], Any]。
# 返回类型：nnx.State；类型/shape约定需与调用方配套。
# 内部调用线索：set → state.filter(filter).flat_state → state.filter → state.map → fn（含分支中的调用，实际路径由条件决定）。
def state_map(state: nnx.State, filter: nnx.filterlib.Filter, fn: Callable[[Any], Any]) -> nnx.State:
    """Apply a function to the leaves of the state that match the filter."""
    filtered_keys = set(state.filter(filter).flat_state())
    return state.map(lambda k, v: fn(v) if k in filtered_keys else v)
