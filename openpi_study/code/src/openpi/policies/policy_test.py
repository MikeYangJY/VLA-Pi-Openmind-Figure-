# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：验证与示例｜用小样本和断言检查policy相关行为，是理解预期输入输出的例子。
# 阅读顺序：先看test函数里的构造输入，再看被测调用和assert/数值比较。
# 重点边界：测试覆盖的是列出的行为，不等于完整机器人部署或任务成功率验证。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
from openpi_client import action_chunk_broker
import pytest

from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


# 【test_infer】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：result['actions'].shape == (config.model.action_horizon, 14)。
# 内部调用线索：_config.get_config → _policy_config.create_trained_policy → aloha_policy.make_aloha_example → policy.infer（含分支中的调用，实际路径由条件决定）。
@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


# 【test_broker】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：outputs['actions'].shape == (14,)。
# 内部调用线索：_config.get_config → _policy_config.create_trained_policy → action_chunk_broker.ActionChunkBroker → aloha_policy.make_aloha_example → broker.infer（含分支中的调用，实际路径由条件决定）。
@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)
