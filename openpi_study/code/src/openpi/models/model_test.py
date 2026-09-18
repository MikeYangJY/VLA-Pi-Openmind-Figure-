# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：验证与示例｜用小样本和断言检查model相关行为，是理解预期输入输出的例子。
# 阅读顺序：先看test函数里的构造输入，再看被测调用和assert/数值比较。
# 重点边界：测试覆盖的是列出的行为，不等于完整机器人部署或任务成功率验证。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
from flax import nnx
import jax
import pytest

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.shared import download
from openpi.shared import nnx_utils


# 【test_pi0_model】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：loss.shape == (batch_size, config.action_horizon)。
# 内部调用线索：jax.random.key → pi0_config.Pi0Config → config.create → config.fake_obs → config.fake_act（含分支中的调用，实际路径由条件决定）。
def test_pi0_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


# 【test_pi0_lora_model】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：loss.shape == (batch_size, config.action_horizon)。
# 内部调用线索：jax.random.key → pi0_config.Pi0Config → config.create → config.fake_obs → config.fake_act（含分支中的调用，实际路径由条件决定）。
def test_pi0_lora_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


# 【test_pi0_fast_model】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：loss.shape == (batch_size,)。
# 内部调用线索：jax.random.key → pi0_fast.Pi0FASTConfig → config.create → config.fake_obs → config.fake_act（含分支中的调用，实际路径由条件决定）。
def test_pi0_fast_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)


# 【test_pi0_fast_lora_model】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：loss.shape == (batch_size,)。
# 内部调用线索：jax.random.key → pi0_fast.Pi0FASTConfig → config.create → config.fake_obs → config.fake_act（含分支中的调用，实际路径由条件决定）。
def test_pi0_fast_lora_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)

    lora_filter = nnx_utils.PathRegex(".*lora.*")
    model_state = nnx.state(model)

    lora_state_elems = list(model_state.filter(lora_filter))
    assert len(lora_state_elems) > 0


# 【test_model_restore】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：loss.shape == (batch_size, config.action_horizon)。
# 内部调用线索：jax.random.key → pi0_config.Pi0Config → config.fake_obs → config.fake_act → config.load（含分支中的调用，实际路径由条件决定）。
@pytest.mark.manual
def test_model_restore():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    model = config.load(
        _model.restore_params(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"))
    )

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)
