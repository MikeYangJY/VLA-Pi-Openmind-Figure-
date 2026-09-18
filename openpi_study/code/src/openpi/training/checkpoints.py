# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：训练持久化｜管理实验目录、保存/恢复训练状态，并把归一化等资产一起保存。
# 阅读顺序：initialize_checkpoint_dir处理新建/续训；save_state分离参数与训练状态；restore_state恢复。
# 重点边界：断点续训需要优化器/步数等状态；推理只加载权重不是完整resume。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
from typing import Protocol

from etils import epath
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


# 【initialize_checkpoint_dir】决定新建、恢复或覆盖实验目录，并创建checkpoint管理器；overwrite须留意已有文件。
# 输入接口：checkpoint_dir:epath.Path | str；keep_period:int | None；overwrite:bool；resume:bool。
# 返回类型：tuple[ocp.CheckpointManager, bool]；类型/shape约定需与调用方配套。
# 内部调用线索：epath.Path(checkpoint_dir).resolve → epath.Path → checkpoint_dir.exists → checkpoint_dir.rmtree → checkpoint_dir.mkdir（含分支中的调用，实际路径由条件决定）。
def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


# 【save_state】将训练状态与推理资产写入checkpoint，区分可恢复训练的信息和推理参数。
# 输入接口：checkpoint_manager:ocp.CheckpointManager；state:training_utils.TrainState；data_loader:_data_loader.DataLoader；step:int。
# 内部调用线索：at.disable_typechecking → _split_params → checkpoint_manager.save（含分支中的调用，实际路径由条件决定）。
def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    # 【save_state.save_assets】本函数位于“训练持久化”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 data_loader.data_config → _normalize.save 追踪具体实现。
    # 输入接口：directory:epath.Path。
    # 内部调用线索：data_loader.data_config → _normalize.save（含分支中的调用，实际路径由条件决定）。
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


# 【restore_state】把保存的参数、优化器及训练步等恢复到既定结构，延续原训练进度。
# 输入接口：checkpoint_manager:ocp.CheckpointManager；state:training_utils.TrainState；data_loader:_data_loader.DataLoader；step:int | None。
# 返回类型：training_utils.TrainState；类型/shape约定需与调用方配套。
# 内部调用线索：at.disable_typechecking → _split_params → checkpoint_manager.restore → _merge_params（含分支中的调用，实际路径由条件决定）。
def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    return _merge_params(restored["train_state"], restored["params"])


# 【load_norm_stats】本函数位于“训练持久化”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 epath.Path → _normalize.load → logging.info 追踪具体实现。
# 输入接口：assets_dir:epath.Path | str；asset_id:str。
# 返回类型：dict[str, _normalize.NormStats] | None；类型/shape约定需与调用方配套。
# 内部调用线索：epath.Path → _normalize.load → logging.info（含分支中的调用，实际路径由条件决定）。
def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


# 【Callback】管理实验目录、保存/恢复训练状态，并把归一化等资产一起保存。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
class Callback(Protocol):
    # 【Callback.__call__】执行管理实验目录、保存/恢复训练状态，并把归一化等资产一起保存。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。
    # 输入接口：directory:epath.Path。
    # 返回类型：None；类型/shape约定需与调用方配套。
    def __call__(self, directory: epath.Path) -> None: ...


# 【CallbackHandler】管理实验目录、保存/恢复训练状态，并把归一化等资产一起保存。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    # 【CallbackHandler.save】把当前数据/状态写入指定存储；文件路径与覆盖行为由这里的实现决定。
    # 输入接口：directory:epath.Path；args:CallbackSave。
    # 内部调用线索：jax.process_index → args.callback（含分支中的调用，实际路径由条件决定）。
    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    # 【CallbackHandler.async_save】本函数位于“训练持久化”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 future.CommitFutureAwaitingContractedSignals → asyncio.to_thread 追踪具体实现。
    # 输入接口：directory:epath.Path；args:CallbackSave。
    # 返回类型：list[futures.Future]；类型/shape约定需与调用方配套。
    # 内部调用线索：future.CommitFutureAwaitingContractedSignals → asyncio.to_thread（含分支中的调用，实际路径由条件决定）。
    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    # 【CallbackHandler.restore】本函数位于“训练持久化”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 NotImplementedError 追踪具体实现。
    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


# 【CallbackSave】管理实验目录、保存/恢复训练状态，并把归一化等资产一起保存。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


# 【CallbackRestore】管理实验目录、保存/恢复训练状态，并把归一化等资产一起保存。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


# 【_split_params】本函数位于“训练持久化”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 dataclasses.replace 追踪具体实现。
# 输入接口：state:training_utils.TrainState。
# 返回类型：tuple[training_utils.TrainState, at.Params]；类型/shape约定需与调用方配套。
def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


# 【_merge_params】本函数位于“训练持久化”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 dataclasses.replace 追踪具体实现。
# 输入接口：train_state:training_utils.TrainState；params:dict[str, at.Params]。
# 返回类型：training_utils.TrainState；类型/shape约定需与调用方配套。
def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
