# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：训练主入口｜JAX训练完整流程：加载配置/数据/权重，编译训练步，更新并保存参数。
# 阅读顺序：先train_step理解一次更新，再main理解循环，最后init_train_state理解初始化与分片。
# 重点边界：模型compute_loss只算误差，value_and_grad和optimizer负责学习；checkpoint overwrite选项会影响已有实验目录。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


# 【init_logging】本函数位于“训练主入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 CustomFormatter → logging.getLogger → logger.setLevel 追踪具体实现。
# 内部调用线索：CustomFormatter → logging.getLogger → logger.setLevel → logger.handlers[0].setFormatter（含分支中的调用，实际路径由条件决定）。
def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    # 【init_logging.CustomFormatter】JAX训练完整流程：加载配置/数据/权重，编译训练步，更新并保存参数。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
    class CustomFormatter(logging.Formatter):
        # 【init_logging.CustomFormatter.format】本函数位于“训练主入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 level_mapping.get → super().format 追踪具体实现。
        # 输入接口：record。
        # 返回值可从这里追踪：super().format(record)。
        # 内部调用线索：level_mapping.get → super().format（含分支中的调用，实际路径由条件决定）。
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


# 【init_wandb】本函数位于“训练主入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 wandb.init → ckpt_dir.exists → FileNotFoundError 追踪具体实现。
# 输入接口：config:_config.TrainConfig；resuming:bool；log_code:bool；enabled:bool。
# 内部调用线索：wandb.init → ckpt_dir.exists → FileNotFoundError → (ckpt_dir / 'wandb_id.txt').read_text().strip → (ckpt_dir / 'wandb_id.txt').read_text（含分支中的调用，实际路径由条件决定）。
def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


# 【_load_weights_and_validate】读取初始化权重并校验PyTree结构、形状与dtype，滤除只表示形状而非真实参数的占位。
# 输入接口：loader:_weight_loaders.WeightLoader；params_shape:at.Params。
# 返回类型：at.Params；类型/shape约定需与调用方配套。
# 内部调用线索：loader.load → at.check_pytree_equality → traverse_util.unflatten_dict → traverse_util.flatten_dict(loaded_params).items → traverse_util.flatten_dict（含分支中的调用，实际路径由条件决定）。
def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


# 【init_train_state】创建优化器和模型状态，按需加载预训练参数并设置分片；resume路径先准备结构供恢复。
# 输入接口：config:_config.TrainConfig；init_rng:at.KeyArrayLike；mesh:jax.sharding.Mesh；resume:bool。
# 返回类型：tuple[training_utils.TrainState, Any]；类型/shape约定需与调用方配套。
# 内部调用线索：_optimizer.create_optimizer → jax.eval_shape → sharding.fsdp_sharding → _load_weights_and_validate → train_state_shape.params.to_pure_dict（含分支中的调用，实际路径由条件决定）。
@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    # 【init_train_state.init】本函数位于“训练主入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 jax.random.split → config.model.create → nnx.split 追踪具体实现。
    # 输入接口：rng:at.KeyArrayLike；partial_params:at.Params | None。
    # 返回类型：training_utils.TrainState；类型/shape约定需与调用方配套。
    # 内部调用线索：jax.random.split → config.model.create → nnx.split → state.replace_by_pure_dict → nnx.merge（含分支中的调用，实际路径由条件决定）。
    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


# 【train_step】JAX的一次参数更新：组回模型→compute_loss→对可训练参数求梯度→优化器更新→可选EMA→返回状态与指标。
# 输入接口：config:_config.TrainConfig；rng:at.KeyArrayLike；state:training_utils.TrainState；batch:tuple[_model.Observation, _model.Actions]。
# 返回类型：tuple[training_utils.TrainState, dict[str, at.Array]]；类型/shape约定需与调用方配套。
# 内部调用线索：nnx.merge → model.train → jax.random.fold_in → nnx.DiffState → nnx.value_and_grad(loss_fn, argnums=diff_state)（含分支中的调用，实际路径由条件决定）。
@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    # 【train_step.loss_fn】把模型按样本/时间返回的误差取均值为标量，便于自动微分。
    # 输入接口：model:_model.BaseModel；rng:at.KeyArrayLike；observation:_model.Observation；actions:_model.Actions。
    # 返回值可从这里追踪：jnp.mean(chunked_loss)。
    # 内部调用线索：model.compute_loss → jnp.mean（含分支中的调用，实际路径由条件决定）。
    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    # 学习提示：仅对筛选出的可训练参数求导；冻结参数仍可参与前向条件计算。
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    # 学习提示：优化器读取梯度和历史动量等状态，算出参数增量；这一步还没写回模型。
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    # 学习提示：把参数增量真正应用到可训练参数。
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


# 【main】本脚本入口：JAX训练完整流程：加载配置/数据/权重，编译训练步，更新并保存参数。 先train_step理解一次更新，再main理解循环，最后init_train_state理解初始化与分片。
# 输入接口：config:_config.TrainConfig。
# 内部调用线索：init_logging → logging.info → platform.node → jax.device_count → ValueError（含分支中的调用，实际路径由条件决定）。
def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
