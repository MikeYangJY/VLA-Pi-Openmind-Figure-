# `scripts/train_pytorch.py` 中文阅读说明

**定位：** 训练对照入口。

PyTorch训练流程，管理DDP、数据、优化器、学习率、日志与checkpoint。

**建议读法：** 从main到train_loop：读batch→model→loss.mean→backward→clip→optimizer.step；再看save/load。

**易错点：** DDP和JAX FSDP不是同一实现；已发布功能以此版本README为准，不把两后端支持项混用。

[注释源码](../../code/scripts/train_pytorch.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/scripts/train_pytorch.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [init_logging](../../code/scripts/train_pytorch.py#L58) | 本函数位于“训练对照入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 CustomFormatter → logging.getLogger → logger.setLevel 追踪具体实现。 |
| [init_logging.CustomFormatter](../../code/scripts/train_pytorch.py#L62) | PyTorch训练流程，管理DDP、数据、优化器、学习率、日志与checkpoint。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [init_logging.CustomFormatter.format](../../code/scripts/train_pytorch.py#L67) | 本函数位于“训练对照入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 level_mapping.get → super().format 追踪具体实现。 |
| [init_wandb](../../code/scripts/train_pytorch.py#L88) | 本函数位于“训练对照入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 wandb.init → ckpt_dir.exists → FileNotFoundError 追踪具体实现。 |
| [setup_ddp](../../code/scripts/train_pytorch.py#L113) | 本函数位于“训练对照入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 os.environ.get → torch.distributed.is_initialized → torch.cuda.is_available 追踪具体实现。 |
| [cleanup_ddp](../../code/scripts/train_pytorch.py#L133) | 本函数位于“训练对照入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 torch.distributed.is_initialized → torch.distributed.barrier → torch.distributed.destroy_process_group 追踪具体实现。 |
| [set_seed](../../code/scripts/train_pytorch.py#L142) | 本函数位于“训练对照入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 torch.manual_seed → np.random.seed → torch.cuda.is_available 追踪具体实现。 |
| [build_datasets](../../code/scripts/train_pytorch.py#L153) | 本函数位于“训练对照入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 _data.create_data_loader → data_loader.data_config 追踪具体实现。 |
| [get_model_state_dict](../../code/scripts/train_pytorch.py#L163) | 本函数位于“训练对照入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 isinstance → model.module.state_dict → model.state_dict 追踪具体实现。 |
| [get_model_parameters](../../code/scripts/train_pytorch.py#L176) | 本函数位于“训练对照入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 isinstance → model.module.parameters → model.parameters 追踪具体实现。 |
| [save_checkpoint](../../code/scripts/train_pytorch.py#L188) | 本函数位于“训练对照入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 tmp_ckpt_dir.exists → shutil.rmtree → tmp_ckpt_dir.mkdir 追踪具体实现。 |
| [load_checkpoint](../../code/scripts/train_pytorch.py#L240) | 本函数位于“训练对照入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 checkpoint_dir.iterdir → d.is_dir → d.name.isdigit 追踪具体实现。 |
| [get_latest_checkpoint_step](../../code/scripts/train_pytorch.py#L321) | 本函数位于“训练对照入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 checkpoint_dir.iterdir → d.is_dir → d.name.isdigit 追踪具体实现。 |
| [log_memory_usage](../../code/scripts/train_pytorch.py#L334) | 本函数位于“训练对照入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 torch.cuda.is_available → torch.cuda.memory_allocated → torch.cuda.memory_reserved 追踪具体实现。 |
| [train_loop](../../code/scripts/train_pytorch.py#L362) | 本函数位于“训练对照入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 setup_ddp → dist.get_rank → set_seed 追踪具体实现。 |
| [train_loop.lr_schedule](../../code/scripts/train_pytorch.py#L529) | 本函数位于“训练对照入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 min → max → np.cos 追踪具体实现。 |
| [main](../../code/scripts/train_pytorch.py#L684) | 本脚本入口：PyTorch训练流程，管理DDP、数据、优化器、学习率、日志与checkpoint。 从main到train_loop：读batch→model→loss.mean→backward→clip→optimizer.step；再看save/load。 |
