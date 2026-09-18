# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：多GPU工程｜构造设备mesh与张量分片规则，为JAX训练降低单卡内存压力。
# 阅读顺序：make_mesh定义设备轴；fsdp_sharding按参数形状分配分片；activation约束作用于中间张量。
# 重点边界：分片改变计算放置，不改变任务监督；单卡读懂模型后再看本文件。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import contextlib
import logging

import jax
import numpy as np

BATCH_AXIS = "batch"
FSDP_AXIS = "fsdp"
# In FSDP, we shard the data across both the batch and FSDP axes.
DATA_AXIS = (BATCH_AXIS, FSDP_AXIS)


# 【_MeshState】构造设备mesh与张量分片规则，为JAX训练降低单卡内存压力。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
class _MeshState:
    active_mesh: jax.sharding.Mesh | None = None


# 【make_mesh】本函数位于“多GPU工程”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 jax.device_count → ValueError → jax.make_mesh 追踪具体实现。
# 输入接口：num_fsdp_devices:int。
# 返回类型：jax.sharding.Mesh；类型/shape约定需与调用方配套。
# 内部调用线索：jax.device_count → ValueError → jax.make_mesh（含分支中的调用，实际路径由条件决定）。
def make_mesh(num_fsdp_devices: int) -> jax.sharding.Mesh:
    if jax.device_count() % num_fsdp_devices != 0:
        raise ValueError(
            f"Number of devices {jax.device_count()} must be divisible by the number of FSDP devices {num_fsdp_devices}."
        )
    mesh_shape = (jax.device_count() // num_fsdp_devices, num_fsdp_devices)
    return jax.make_mesh(mesh_shape, (BATCH_AXIS, FSDP_AXIS))


# 【set_mesh】本函数位于“多GPU工程”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 ValueError 追踪具体实现。
# 输入接口：mesh:jax.sharding.Mesh。
@contextlib.contextmanager
def set_mesh(mesh: jax.sharding.Mesh):
    """Plumbing the mesh deep into the module tree is extremely cumbersome; until the JAX team lands a better API, a
    custom context manager like this one is the recommended way to maintain a reference to a global mesh. This is only used
    in `activation_sharding_constraint` below."""
    if _MeshState.active_mesh is not None:
        raise ValueError("Cannot nest set_mesh context managers.")
    _MeshState.active_mesh = mesh
    try:
        yield
    finally:
        _MeshState.active_mesh = None


# 【activation_sharding_constraint】本函数位于“多GPU工程”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 jax.lax.with_sharding_constraint → jax.sharding.NamedSharding → jax.sharding.PartitionSpec 追踪具体实现。
# 输入接口：pytree。
# 返回值可从这里追踪：pytree / jax.lax.with_sharding_constraint(pytree, jax.sharding.NamedSharding(_MeshState.active_mesh, jax.sharding.Parti…。
# 内部调用线索：jax.lax.with_sharding_constraint → jax.sharding.NamedSharding → jax.sharding.PartitionSpec（含分支中的调用，实际路径由条件决定）。
def activation_sharding_constraint(pytree):
    if _MeshState.active_mesh is None:
        return pytree
    return jax.lax.with_sharding_constraint(
        pytree, jax.sharding.NamedSharding(_MeshState.active_mesh, jax.sharding.PartitionSpec(DATA_AXIS))
    )


# 【fsdp_sharding】本函数位于“多GPU工程”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 jax.tree_util.tree_map_with_path 追踪具体实现。
# 输入接口：pytree；mesh:jax.sharding.Mesh；min_size_mbytes:int；log:bool。
# 返回值可从这里追踪：jax.tree_util.tree_map_with_path(_shard_arr, pytree)。
def fsdp_sharding(
    pytree,
    mesh: jax.sharding.Mesh,
    *,
    min_size_mbytes: int = 4,  # 4 MiB
    log: bool = False,
):
    """Apply FSDP sharding to a pytree of arrays based on the mesh shape.

    Args:
        pytree: A pytree to be apply sharding specified by the mesh, note that only array types (eg. contains .shape attr)
          will be considered for sharding.
        mesh: The mesh being used for applying sharding on to pytree.
        min_size_mbytes: The minimum size of the array in MiB to be considered for sharding, any array smaller than this
          will be replicated.
        log: If true, will log the sharding decisions for arrays that are being considered for sharding.

    Returns:
        The sharded pytree.
    """
    min_size_bytes = min_size_mbytes * 2**20

    # 【fsdp_sharding._shard_arr】本函数位于“多GPU工程”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 jax.sharding.NamedSharding → jax.sharding.PartitionSpec → hasattr 追踪具体实现。
    # 输入接口：kp；array:jax.ShapeDtypeStruct。
    # 返回值可从这里追踪：jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())。
    # 内部调用线索：jax.sharding.NamedSharding → jax.sharding.PartitionSpec → hasattr → np.prod → np.dtype（含分支中的调用，实际路径由条件决定）。
    def _shard_arr(kp, array: jax.ShapeDtypeStruct):
        # if fsdp is not actually going to be used, replicate everything to avoid extraneous logging
        if mesh.shape[FSDP_AXIS] == 1:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        # replicate scalar and vector arrays
        if not hasattr(array, "shape"):
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        if len(array.shape) < 2:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        # replicate small arrays
        if (arr_size := np.prod(array.shape) * np.dtype(array.dtype).itemsize) < min_size_bytes:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

        # shard matrices and larger tensors along the largest axis that is divisible by the fsdp dimension
        axes = np.argsort(array.shape)[::-1]
        spec = [None] * len(axes)
        for i in axes:
            if array.shape[i] % mesh.shape[FSDP_AXIS] == 0:
                if log:
                    logging.info(
                        f"Sharding {jax.tree_util.keystr(kp)} of shape {array.shape} ({arr_size / 2**20:.2f} MiB) along axis {i}"
                    )
                spec[i] = FSDP_AXIS
                return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*spec))

        # replicate if no valid sharding was found
        if log:
            logging.warning(
                f"Could not find a valid sharding for {jax.tree_util.keystr(kp)} of shape {array.shape} with mesh of shape {mesh.shape}"
            )
        return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    return jax.tree_util.tree_map_with_path(_shard_arr, pytree)
