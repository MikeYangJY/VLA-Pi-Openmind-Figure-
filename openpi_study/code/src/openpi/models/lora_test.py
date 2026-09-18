# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：验证与示例｜用小样本和断言检查lora相关行为，是理解预期输入输出的例子。
# 阅读顺序：先看test函数里的构造输入，再看被测调用和assert/数值比较。
# 重点边界：测试覆盖的是列出的行为，不等于完整机器人部署或任务成功率验证。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.models.lora as lora


# 【test_lora_einsum_params_shape】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：'lora_a' not in params['params']。
# 内部调用线索：lora.Einsum → lora.LoRAConfig → jax.random.key → jax.random.normal → einsum.init（含分支中的调用，实际路径由条件决定）。
def test_lora_einsum_params_shape():
    shape = (3, 8, 32, 4)  # (3KDH)
    einsum = lora.Einsum(shape)
    lora0 = lora.Einsum(shape, lora_config=lora.LoRAConfig(rank=2))
    lora1 = lora.Einsum(shape, lora_config=lora.LoRAConfig(rank=2, axes=(1, 2)))

    key = jax.random.key(0)
    x = jax.random.normal(key, (8, 64, 32))  # (BSD)
    eqn = "BSD,3KDH->3BSKH"

    # Ensure that lora parameters are not initialized when LoRA is not used.
    params = einsum.init(key, eqn, x)
    assert "lora_a" not in params["params"]
    assert "lora_b" not in params["params"]

    # Check that default axes work.
    params_lora0 = lora0.init(key, eqn, x)
    assert params_lora0["params"]["lora_a"].shape == (3, 8, 32, 2)
    assert params_lora0["params"]["lora_b"].shape == (3, 8, 2, 4)

    # Check that user provided axes work.
    params_lora1 = lora1.init(key, eqn, x)
    assert params_lora1["params"]["lora_a"].shape == (3, 8, 2, 4)
    assert params_lora1["params"]["lora_b"].shape == (3, 2, 32, 4)


# 【test_lora_einsum_same_output】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：jnp.allclose(output, output_lora)。
# 内部调用线索：lora.Einsum → lora.LoRAConfig → jax.random.key → jax.random.normal → einsum.init（含分支中的调用，实际路径由条件决定）。
def test_lora_einsum_same_output():
    shape = (3, 8, 32, 4)  # (3KDH)
    einsum = lora.Einsum(shape)
    einsum_lora = lora.Einsum(shape, lora_config=lora.LoRAConfig(rank=2, init_fn=nn.initializers.zeros))

    key = jax.random.key(0)
    x = jax.random.normal(key, (8, 64, 32))  # (BSD)
    eqn = "BSD,3KDH->3BSKH"

    params = einsum.init(key, eqn, x)
    output = einsum.apply(params, eqn, x)

    params_lora = einsum_lora.init(key, eqn, x)
    output_lora = einsum_lora.apply(params_lora, eqn, x)

    # Results are the same since the LoRA parameters are initialized to zeros.
    assert jnp.allclose(output, output_lora)


# 【test_lora_ffn_params_shape】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：params['params']['gating_einsum'].shape == (2, 8, 32)。
# 内部调用线索：lora.FeedForward → lora.LoRAConfig → jax.random.key → jax.random.normal → ffn.init（含分支中的调用，实际路径由条件决定）。
def test_lora_ffn_params_shape():
    ffn = lora.FeedForward(features=8, hidden_dim=32)
    ffn_lora = lora.FeedForward(
        features=8,
        hidden_dim=32,
        lora_config=lora.LoRAConfig(rank=2),
    )

    key = jax.random.key(0)
    x = jax.random.normal(key, (2, 8))

    params = ffn.init(key, x)
    assert params["params"]["gating_einsum"].shape == (2, 8, 32)
    assert params["params"]["linear"].shape == (32, 8)

    params_lora = ffn_lora.init(key, x)
    assert params_lora["params"]["gating_einsum"].shape == (2, 8, 32)
    assert params_lora["params"]["linear"].shape == (32, 8)
    assert params_lora["params"]["gating_einsum_lora_a"].shape == (2, 8, 2)
    assert params_lora["params"]["gating_einsum_lora_b"].shape == (2, 2, 32)
    assert params_lora["params"]["linear_lora_a"].shape == (32, 2)
    assert params_lora["params"]["linear_lora_b"].shape == (2, 8)


# 【test_lora_ffn_same_output】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：jnp.allclose(output, output_lora)。
# 内部调用线索：lora.FeedForward → lora.LoRAConfig → jax.random.key → jax.random.normal → ffn.init（含分支中的调用，实际路径由条件决定）。
def test_lora_ffn_same_output():
    ffn = lora.FeedForward(features=8, hidden_dim=32)
    ffn_lora = lora.FeedForward(
        features=8,
        hidden_dim=32,
        lora_config=lora.LoRAConfig(rank=2, init_fn=nn.initializers.zeros),
    )

    key = jax.random.key(0)
    x = jax.random.normal(key, (2, 8))

    params = ffn.init(key, x)
    output = ffn.apply(params, x)

    params_lora = ffn_lora.init(key, x)
    output_lora = ffn_lora.apply(params_lora, x)

    assert jnp.allclose(output, output_lora)
