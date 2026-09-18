# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：无机器人接口测试｜构造随机观测并向推理服务发送请求，收集耗时统计。
# 阅读顺序：按环境生成匹配字段→等待服务→调用infer→记录统计。
# 重点边界：随机输入能检查接口，不代表机器人任务成功率或真实闭环性能。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import dataclasses
import enum
import logging
import pathlib
import time

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import polars as pl
import rich
import tqdm
import tyro

logger = logging.getLogger(__name__)


# 【EnvMode】构造随机观测并向推理服务发送请求，收集耗时统计。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


# 【Args】构造随机观测并向推理服务发送请求，收集耗时统计。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclasses.dataclass
class Args:
    """Command line arguments."""

    # Host and port to connect to the server.
    host: str = "0.0.0.0"
    # Port to connect to the server. If None, the server will use the default port.
    port: int | None = 8000
    # API key to use for the server.
    api_key: str | None = None
    # Number of steps to run the policy for.
    num_steps: int = 20
    # Path to save the timings to a parquet file. (e.g., timing.parquet)
    timing_file: pathlib.Path | None = None
    # Environment to run the policy in.
    env: EnvMode = EnvMode.ALOHA_SIM


# 【TimingRecorder】构造随机观测并向推理服务发送请求，收集耗时统计。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
class TimingRecorder:
    """Records timing measurements for different keys."""

    # 【TimingRecorder.__init__】初始化TimingRecorder的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._timings。
    # 返回类型：None；类型/shape约定需与调用方配套。
    def __init__(self) -> None:
        self._timings: dict[str, list[float]] = {}

    # 【TimingRecorder.record】本函数位于“无机器人接口测试”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self._timings[key].append 追踪具体实现。
    # 输入接口：key:str；time_ms:float。
    # 返回类型：None；类型/shape约定需与调用方配套。
    def record(self, key: str, time_ms: float) -> None:
        """Record a timing measurement for the given key."""
        if key not in self._timings:
            self._timings[key] = []
        self._timings[key].append(time_ms)

    # 【TimingRecorder.get_stats】本函数位于“无机器人接口测试”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.mean → np.std → np.quantile 追踪具体实现。
    # 输入接口：key:str。
    # 返回类型：dict[str, float]；类型/shape约定需与调用方配套。
    # 内部调用线索：np.mean → np.std → np.quantile（含分支中的调用，实际路径由条件决定）。
    def get_stats(self, key: str) -> dict[str, float]:
        """Get statistics for the given key."""
        times = self._timings[key]
        return {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "p25": float(np.quantile(times, 0.25)),
            "p50": float(np.quantile(times, 0.50)),
            "p75": float(np.quantile(times, 0.75)),
            "p90": float(np.quantile(times, 0.90)),
            "p95": float(np.quantile(times, 0.95)),
            "p99": float(np.quantile(times, 0.99)),
        }

    # 【TimingRecorder.print_all_stats】本函数位于“无机器人接口测试”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 rich.table.Table → table.add_column → sorted 追踪具体实现。
    # 返回类型：None；类型/shape约定需与调用方配套。
    # 内部调用线索：rich.table.Table → table.add_column → sorted → self._timings.keys → self.get_stats（含分支中的调用，实际路径由条件决定）。
    def print_all_stats(self) -> None:
        """Print statistics for all keys in a concise format."""

        table = rich.table.Table(
            title="[bold blue]Timing Statistics[/bold blue]",
            show_header=True,
            header_style="bold white",
            border_style="blue",
            title_justify="center",
        )

        # Add metric column with custom styling
        table.add_column("Metric", style="cyan", justify="left", no_wrap=True)

        # Add statistical columns with consistent styling
        stat_columns = [
            ("Mean", "yellow", "mean"),
            ("Std", "yellow", "std"),
            ("P25", "magenta", "p25"),
            ("P50", "magenta", "p50"),
            ("P75", "magenta", "p75"),
            ("P90", "magenta", "p90"),
            ("P95", "magenta", "p95"),
            ("P99", "magenta", "p99"),
        ]

        for name, style, _ in stat_columns:
            table.add_column(name, justify="right", style=style, no_wrap=True)

        # Add rows for each metric with formatted values
        for key in sorted(self._timings.keys()):
            stats = self.get_stats(key)
            values = [f"{stats[key]:.1f}" for _, _, key in stat_columns]
            table.add_row(key, *values)

        # Print with custom console settings
        console = rich.console.Console(width=None, highlight=True)
        console.print(table)

    # 【TimingRecorder.write_parquet】本函数位于“无机器人接口测试”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 logger.info → pl.DataFrame → path.parent.mkdir 追踪具体实现。
    # 输入接口：path:pathlib.Path。
    # 返回类型：None；类型/shape约定需与调用方配套。
    # 内部调用线索：logger.info → pl.DataFrame → path.parent.mkdir → frame.write_parquet（含分支中的调用，实际路径由条件决定）。
    def write_parquet(self, path: pathlib.Path) -> None:
        """Save the timings to a parquet file."""
        logger.info(f"Writing timings to {path}")
        frame = pl.DataFrame(self._timings)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)


# 【main】本脚本入口：构造随机观测并向推理服务发送请求，收集耗时统计。 按环境生成匹配字段→等待服务→调用infer→记录统计。
# 输入接口：args:Args。
# 返回类型：None；类型/shape约定需与调用方配套。
# 内部调用线索：_websocket_client_policy.WebsocketClientPolicy → logger.info → policy.get_server_metadata → policy.infer → obs_fn（含分支中的调用，实际路径由条件决定）。
def main(args: Args) -> None:
    obs_fn = {
        EnvMode.ALOHA: _random_observation_aloha,
        EnvMode.ALOHA_SIM: _random_observation_aloha,
        EnvMode.DROID: _random_observation_droid,
        EnvMode.LIBERO: _random_observation_libero,
    }[args.env]

    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )
    logger.info(f"Server metadata: {policy.get_server_metadata()}")

    # Send a few observations to make sure the model is loaded.
    for _ in range(2):
        policy.infer(obs_fn())

    timing_recorder = TimingRecorder()

    for _ in tqdm.trange(args.num_steps, desc="Running policy"):
        inference_start = time.time()
        action = policy.infer(obs_fn())
        timing_recorder.record("client_infer_ms", 1000 * (time.time() - inference_start))
        for key, value in action.get("server_timing", {}).items():
            timing_recorder.record(f"server_{key}", value)
        for key, value in action.get("policy_timing", {}).items():
            timing_recorder.record(f"policy_{key}", value)

    timing_recorder.print_all_stats()

    if args.timing_file is not None:
        timing_recorder.write_parquet(args.timing_file)


# 【_random_observation_aloha】本函数位于“无机器人接口测试”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.ones → np.random.randint 追踪具体实现。
# 返回类型：dict；类型/shape约定需与调用方配套。
# 内部调用线索：np.ones → np.random.randint（含分支中的调用，实际路径由条件决定）。
def _random_observation_aloha() -> dict:
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


# 【_random_observation_droid】本函数位于“无机器人接口测试”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.random.randint → np.random.rand 追踪具体实现。
# 返回类型：dict；类型/shape约定需与调用方配套。
# 内部调用线索：np.random.randint → np.random.rand（含分支中的调用，实际路径由条件决定）。
def _random_observation_droid() -> dict:
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


# 【_random_observation_libero】本函数位于“无机器人接口测试”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.random.rand → np.random.randint 追踪具体实现。
# 返回类型：dict；类型/shape约定需与调用方配套。
# 内部调用线索：np.random.rand → np.random.randint（含分支中的调用，实际路径由条件决定）。
def _random_observation_libero() -> dict:
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
