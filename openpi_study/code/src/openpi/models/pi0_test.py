# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：验证与示例｜用小样本和断言检查pi0相关行为，是理解预期输入输出的例子。
# 阅读顺序：先看test函数里的构造输入，再看被测调用和assert/数值比较。
# 重点边界：测试覆盖的是列出的行为，不等于完整机器人部署或任务成功率验证。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import flax.nnx as nnx
import jax

import openpi.models.pi0_config as _pi0_config


# 【_get_frozen_state】本函数位于“验证与示例”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 nnx.eval_shape → jax.random.key → config.get_freeze_filter 追踪具体实现。
# 输入接口：config:_pi0_config.Pi0Config。
# 返回类型：nnx.State；类型/shape约定需与调用方配套。
# 内部调用线索：nnx.eval_shape → jax.random.key → config.get_freeze_filter → nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_stat… → nnx.state（含分支中的调用，实际路径由条件决定）。
def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


# 【test_pi0_full_finetune】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：len(state) == 0。
# 内部调用线索：_pi0_config.Pi0Config → _get_frozen_state（含分支中的调用，实际路径由条件决定）。
def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


# 【test_pi0_gemma_lora】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：len(state) == 9。
# 内部调用线索：_pi0_config.Pi0Config → _get_frozen_state → all（含分支中的调用，实际路径由条件决定）。
def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


# 【test_pi0_action_expert_lora】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：len(state) == 8。
# 内部调用线索：_pi0_config.Pi0Config → _get_frozen_state → all → any（含分支中的调用，实际路径由条件决定）。
def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


# 【test_pi0_all_lora】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：len(state) == 17。
# 内部调用线索：_pi0_config.Pi0Config → _get_frozen_state → all（含分支中的调用，实际路径由条件决定）。
def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
