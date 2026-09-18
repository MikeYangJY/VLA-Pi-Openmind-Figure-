# `src/openpi/shared/nnx_utils.py` 中文阅读说明

**定位：** JAX桥接工具。

把带状态的NNX模块安全包装成JIT函数，并提供参数路径筛选与映射。

**建议读法：** module_jit拆结构与状态再编译/合并；PathRegex选参数；state_map处理选中的叶子。

**易错点：** Python对象有状态，JIT偏好显式数据流；这就是这些拆分/合并步骤存在的原因。

[注释源码](../../../../code/src/openpi/shared/nnx_utils.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/shared/nnx_utils.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [module_jit](../../../../code/src/openpi/shared/nnx_utils.py#L25) | 把NNX模块结构和状态显式分开，再用JAX编译包装方法；返回值保持模块方法使用方式。 |
| [module_jit.fun](../../../../code/src/openpi/shared/nnx_utils.py#L47) | 本函数位于“JAX桥接工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 nnx.merge → meth.__func__ 追踪具体实现。 |
| [module_jit.wrapper](../../../../code/src/openpi/shared/nnx_utils.py#L56) | 本函数位于“JAX桥接工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 jitted_fn 追踪具体实现。 |
| [PathRegex](../../../../code/src/openpi/shared/nnx_utils.py#L64) | 把带状态的NNX模块安全包装成JIT函数，并提供参数路径筛选与映射。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [PathRegex.__post_init__](../../../../code/src/openpi/shared/nnx_utils.py#L75) | 本函数位于“JAX桥接工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 isinstance → object.__setattr__ → re.compile 追踪具体实现。 |
| [PathRegex.__call__](../../../../code/src/openpi/shared/nnx_utils.py#L83) | 执行把带状态的NNX模块安全包装成JIT函数，并提供参数路径筛选与映射。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [state_map](../../../../code/src/openpi/shared/nnx_utils.py#L93) | 本函数位于“JAX桥接工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 set → state.filter(filter).flat_state → state.filter 追踪具体实现。 |
