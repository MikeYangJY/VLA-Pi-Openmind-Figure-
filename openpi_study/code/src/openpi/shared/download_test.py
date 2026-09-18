# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：验证与示例｜用小样本和断言检查download相关行为，是理解预期输入输出的例子。
# 阅读顺序：先看test函数里的构造输入，再看被测调用和assert/数值比较。
# 重点边界：测试覆盖的是列出的行为，不等于完整机器人部署或任务成功率验证。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import pathlib

import pytest

import openpi.shared.download as download


# 【set_openpi_data_home】本函数位于“验证与示例”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 tmp_path_factory.mktemp → pytest.MonkeyPatch().context → pytest.MonkeyPatch 追踪具体实现。
# 输入接口：tmp_path_factory。
# 内部调用线索：tmp_path_factory.mktemp → pytest.MonkeyPatch().context → pytest.MonkeyPatch → mp.setenv（含分支中的调用，实际路径由条件决定）。
@pytest.fixture(scope="session", autouse=True)
def set_openpi_data_home(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("openpi_data")
    with pytest.MonkeyPatch().context() as mp:
        mp.setenv("OPENPI_DATA_HOME", str(temp_dir))
        yield


# 【test_download_local】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：result == local_path。
# 输入接口：tmp_path:pathlib.Path。
# 内部调用线索：local_path.touch → download.maybe_download → pytest.raises（含分支中的调用，实际路径由条件决定）。
def test_download_local(tmp_path: pathlib.Path):
    local_path = tmp_path / "local"
    local_path.touch()

    result = download.maybe_download(str(local_path))
    assert result == local_path

    with pytest.raises(FileNotFoundError):
        download.maybe_download("bogus")


# 【test_download_gs_dir】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：local_path.exists()。
# 内部调用线索：download.maybe_download → local_path.exists（含分支中的调用，实际路径由条件决定）。
def test_download_gs_dir():
    remote_path = "gs://openpi-assets/testdata/random"

    local_path = download.maybe_download(remote_path)
    assert local_path.exists()

    new_local_path = download.maybe_download(remote_path)
    assert new_local_path == local_path


# 【test_download_gs】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：local_path.exists()。
# 内部调用线索：download.maybe_download → local_path.exists（含分支中的调用，实际路径由条件决定）。
def test_download_gs():
    remote_path = "gs://openpi-assets/testdata/random/random_512kb.bin"

    local_path = download.maybe_download(remote_path)
    assert local_path.exists()

    new_local_path = download.maybe_download(remote_path)
    assert new_local_path == local_path


# 【test_download_fsspec】验证本文件相关行为；按“构造输入→调用被测对象→比较结果”读。关键断言：local_path.exists()。
# 内部调用线索：download.maybe_download → local_path.exists（含分支中的调用，实际路径由条件决定）。
def test_download_fsspec():
    remote_path = "gs://big_vision/paligemma_tokenizer.model"

    local_path = download.maybe_download(remote_path, gs={"token": "anon"})
    assert local_path.exists()

    new_local_path = download.maybe_download(remote_path, gs={"token": "anon"})
    assert new_local_path == local_path
