# `examples/convert_jax_model_to_pytorch.py` 中文阅读说明

**定位：** 权重转换。

读取Orbax/JAX权重并重排维度与名称，生成PyTorch可加载的模型权重。

**建议读法：** 先convert_pi0_checkpoint看总体路径，再看PaliGemma/Gemma切片与转置规则。

**易错点：** 转换权重不是重新训练；同名层的矩阵轴约定可能不同，必须按形状验证。

[注释源码](../../code/examples/convert_jax_model_to_pytorch.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/convert_jax_model_to_pytorch.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [slice_paligemma_state_dict](../../code/examples/convert_jax_model_to_pytorch.py#L60) | 本函数位于“权重转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 state_dict.pop(jax_key).transpose → state_dict.pop → state_dict.pop(jax_key).reshape 追踪具体实现。 |
| [slice_gemma_state_dict](../../code/examples/convert_jax_model_to_pytorch.py#L285) | 本函数位于“权重转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 hasattr → state_dict.pop → llm_attention_q_einsum[i].transpose(0, 2, 1).reshape 追踪具体实现。 |
| [slice_initial_orbax_checkpoint](../../code/examples/convert_jax_model_to_pytorch.py#L414) | 本函数位于“权重转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 openpi.models.model.restore_params → traversals.flatten_mapping 追踪具体实现。 |
| [load_jax_model_and_print_keys](../../code/examples/convert_jax_model_to_pytorch.py#L429) | 本函数位于“权重转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 checkpoint_dir.startswith → os.path.abspath → ocp.PyTreeCheckpointer 追踪具体实现。 |
| [convert_pi0_checkpoint](../../code/examples/convert_jax_model_to_pytorch.py#L446) | 本函数位于“权重转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 print → slice_initial_orbax_checkpoint → isinstance 追踪具体实现。 |
| [convert_pi0_checkpoint.PaliGemmaConfig](../../code/examples/convert_jax_model_to_pytorch.py#L501) | 读取Orbax/JAX权重并重排维度与名称，生成PyTorch可加载的模型权重。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [convert_pi0_checkpoint.PaliGemmaConfig.__init__](../../code/examples/convert_jax_model_to_pytorch.py#L504) | 初始化PaliGemmaConfig的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.vision_config、self.text_config。 |
| [main](../../code/examples/convert_jax_model_to_pytorch.py#L588) | 本脚本入口：读取Orbax/JAX权重并重排维度与名称，生成PyTorch可加载的模型权重。 先convert_pi0_checkpoint看总体路径，再看PaliGemma/Gemma切片与转置规则。 |
