# `src/openpi/models_pytorch/pi0_pytorch.py` 中文阅读说明

**定位：** 核心模型对照。

π0/π0.5的PyTorch版本：forward计算flow loss，sample_actions进行Euler积分。

**建议读法：** 先对照JAX版的embed_prefix/embed_suffix，再读forward、sample_actions和denoise_step；显存优化辅助函数后读。

**易错点：** gradient checkpointing是重算激活省显存，不是保存模型权重；pi05分支的AdaRMS时间条件需追到Gemma层。

[注释源码](../../../../code/src/openpi/models_pytorch/pi0_pytorch.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models_pytorch/pi0_pytorch.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [get_safe_dtype](../../../../code/src/openpi/models_pytorch/pi0_pytorch.py#L23) | 本函数位于“核心模型对照”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。 |
| [create_sinusoidal_pos_embedding](../../../../code/src/openpi/models_pytorch/pi0_pytorch.py#L38) | PyTorch版flow时间sin/cos编码，输入[B]，输出[B,D]，供后续动作/时间融合。 |
| [sample_beta](../../../../code/src/openpi/models_pytorch/pi0_pytorch.py#L62) | 本函数位于“核心模型对照”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 torch.as_tensor → torch.distributions.Beta → dist.sample 追踪具体实现。 |
| [make_att_2d_masks](../../../../code/src/openpi/models_pytorch/pi0_pytorch.py#L73) | PyTorch版分块注意力mask。cumsum给各token编号，比较编号得到能否注意，再排除padding。 |
| [PI0Pytorch](../../../../code/src/openpi/models_pytorch/pi0_pytorch.py#L106) | 共享模型的PyTorch实现，forward是训练损失，sample_actions是推理动作。 |
| [PI0Pytorch.__init__](../../../../code/src/openpi/models_pytorch/pi0_pytorch.py#L110) | 初始化PI0Pytorch的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.config、self.pi05、self.paligemma_with_expert、self.action_in_proj、self.action_out_proj。 |
| [PI0Pytorch.gradient_checkpointing_enable](../../../../code/src/openpi/models_pytorch/pi0_pytorch.py#L154) | 切换训练激活重算选项，在显存占用和额外计算之间取舍，不涉及磁盘存档。 |
| [PI0Pytorch.gradient_checkpointing_disable](../../../../code/src/openpi/models_pytorch/pi0_pytorch.py#L164) | 切换训练激活重算选项，在显存占用和额外计算之间取舍，不涉及磁盘存档。 |
| [PI0Pytorch.is_gradient_checkpointing_enabled](../../../../code/src/openpi/models_pytorch/pi0_pytorch.py#L175) | 本函数位于“核心模型对照”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。 |
| [PI0Pytorch._apply_checkpoint](../../../../code/src/openpi/models_pytorch/pi0_pytorch.py#L183) | 在训练且启用时用激活重算换显存；否则直接调用传入函数。这里的checkpoint不是磁盘权重存档。 |
| [PI0Pytorch._prepare_attention_masks_4d](../../../../code/src/openpi/models_pytorch/pi0_pytorch.py#L194) | 将布尔可见性[B,N,N]转成带head广播轴的加性mask[B,1,N,N]：可见为0，不可见为很大的负数。 |
| [PI0Pytorch._preprocess_observation](../../../../code/src/openpi/models_pytorch/pi0_pytorch.py#L203) | 本函数位于“核心模型对照”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 _preprocessing.preprocess_observation_pytorch → observation.images.values → observation.image_masks.values 追踪具体实现。 |
| [PI0Pytorch.sample_noise](../../../../code/src/openpi/models_pytorch/pi0_pytorch.py#L217) | 生成与动作形状相同的标准高斯噪声；不涉及机器人传感器噪声标定。 |
| [PI0Pytorch.sample_time](../../../../code/src/openpi/models_pytorch/pi0_pytorch.py#L230) | 从Beta(1.5,1)采样flow时间并缩放到接近(0,1]范围，决定训练样本噪声强度。 |
| [PI0Pytorch.embed_prefix](../../../../code/src/openpi/models_pytorch/pi0_pytorch.py#L239) | 图像经视觉塔得到patch embedding，语言查embedding并缩放；沿token轴拼接，并建立前缀有效性与注意力分块mask。 |
| [PI0Pytorch.embed_prefix.image_embed_func](../../../../code/src/openpi/models_pytorch/pi0_pytorch.py#L255) | 本函数位于“核心模型对照”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.paligemma_with_expert.embed_image 追踪具体实现。 |
| [PI0Pytorch.embed_prefix.lang_embed_func](../../../../code/src/openpi/models_pytorch/pi0_pytorch.py#L273) | 本函数位于“核心模型对照”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.paligemma_with_expert.embed_language_tokens → math.sqrt 追踪具体实现。 |
| [PI0Pytorch.embed_suffix](../../../../code/src/openpi/models_pytorch/pi0_pytorch.py#L301) | 把动作、可选连续state与flow时间变成专家输入；pi05分支通过time MLP提供AdaRMS条件。 |
| [PI0Pytorch.embed_suffix.state_proj_func](../../../../code/src/openpi/models_pytorch/pi0_pytorch.py#L316) | 本函数位于“核心模型对照”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.state_proj 追踪具体实现。 |
| [PI0Pytorch.embed_suffix.action_proj_func](../../../../code/src/openpi/models_pytorch/pi0_pytorch.py#L341) | 本函数位于“核心模型对照”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.action_in_proj 追踪具体实现。 |
| [PI0Pytorch.embed_suffix.mlp_func](../../../../code/src/openpi/models_pytorch/pi0_pytorch.py#L356) | 本函数位于“核心模型对照”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.action_time_mlp_in → F.silu → self.action_time_mlp_out 追踪具体实现。 |
| [PI0Pytorch.embed_suffix.time_mlp_func](../../../../code/src/openpi/models_pytorch/pi0_pytorch.py#L370) | 本函数位于“核心模型对照”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.time_mlp_in → F.silu → self.time_mlp_out 追踪具体实现。 |
| [PI0Pytorch.forward](../../../../code/src/openpi/models_pytorch/pi0_pytorch.py#L402) | 训练前向：构造x_t与目标u_t，联合运行前缀/后缀，取动作token输出向量场，返回未归约MSE；训练循环再取平均并反传。 |
| [PI0Pytorch.forward.forward_func](../../../../code/src/openpi/models_pytorch/pi0_pytorch.py#L441) | 本函数位于“核心模型对照”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.paligemma_with_expert.forward 追踪具体实现。 |
| [PI0Pytorch.forward.action_out_proj_func](../../../../code/src/openpi/models_pytorch/pi0_pytorch.py#L463) | 本函数位于“核心模型对照”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.action_out_proj 追踪具体实现。 |
| [PI0Pytorch.sample_actions](../../../../code/src/openpi/models_pytorch/pi0_pytorch.py#L476) | 无梯度推理：缓存前缀K/V，从噪声开始循环denoise_step与Euler更新，输出整个动作块。 |
| [PI0Pytorch.denoise_step](../../../../code/src/openpi/models_pytorch/pi0_pytorch.py#L527) | 仅重算当前动作后缀，拼接对前缀缓存的注意力mask，得到此flow时间的v_t。 |
