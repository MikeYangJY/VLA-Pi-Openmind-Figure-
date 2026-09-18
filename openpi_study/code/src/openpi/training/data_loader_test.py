# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：验证与示例｜用小样本和断言检查data_loader相关行为，是理解预期输入输出的例子。
# 阅读顺序：先看test函数里的构造输入，再看被测调用和assert/数值比较。
# 重点边界：测试覆盖的是列出的行为，不等于完整机器人部署或任务成功率验证。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import dataclasses

import jax

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


# 【test_torch_data_loader】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：len(batches) == 2。
# 内部调用线索：pi0_config.Pi0Config → _data_loader.FakeDataset → _data_loader.TorchDataLoader → all → jax.tree.leaves（含分支中的调用，实际路径由条件决定）。
def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


# 【test_torch_data_loader_infinite】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。数值比较或预期异常给出通过条件。
# 内部调用线索：pi0_config.Pi0Config → _data_loader.FakeDataset → _data_loader.TorchDataLoader → iter → next（含分支中的调用，实际路径由条件决定）。
def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


# 【test_torch_data_loader_parallel】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：len(batches) == 2。
# 内部调用线索：pi0_config.Pi0Config → _data_loader.FakeDataset → _data_loader.TorchDataLoader → all → jax.tree.leaves（含分支中的调用，实际路径由条件决定）。
def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


# 【test_with_fake_dataset】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：len(batches) == 2。
# 内部调用线索：_config.get_config → _data_loader.create_data_loader → all → jax.tree.leaves（含分支中的调用，实际路径由条件决定）。
def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


# 【test_with_real_dataset】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：loader.data_config().repo_id == config.data.repo_id。
# 内部调用线索：_config.get_config → dataclasses.replace → _data_loader.create_data_loader → loader.data_config（含分支中的调用，实际路径由条件决定）。
def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
