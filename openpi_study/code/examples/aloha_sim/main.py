# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：仿真运行组装｜把仿真环境、WebSocket策略与视频保存订阅者接入runtime。
# 阅读顺序：按Args创建环境和策略，设置控制频率与episode数，再运行。
# 重点边界：仿真客户端和模型服务可以用不同Python环境。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import dataclasses
import logging
import pathlib

import env as _env
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import saver as _saver
import tyro


# 【Args】把仿真环境、WebSocket策略与视频保存订阅者接入runtime。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclasses.dataclass
class Args:
    out_dir: pathlib.Path = pathlib.Path("data/aloha_sim/videos")

    task: str = "gym_aloha/AlohaTransferCube-v0"
    # 字段含义：随机种子，帮助复查初始化和数据顺序。
    seed: int = 0

    # 字段含义：一次预测未来多少个控制时刻，与一次任务持续多久不同。
    action_horizon: int = 10

    host: str = "0.0.0.0"
    port: int = 8000

    display: bool = False


# 【main】本脚本入口：把仿真环境、WebSocket策略与视频保存订阅者接入runtime。 按Args创建环境和策略，设置控制频率与episode数，再运行。
# 输入接口：args:Args。
# 返回类型：None；类型/shape约定需与调用方配套。
# 内部调用线索：_runtime.Runtime → _env.AlohaSimEnvironment → _policy_agent.PolicyAgent → action_chunk_broker.ActionChunkBroker → _websocket_client_policy.WebsocketClientPolicy（含分支中的调用，实际路径由条件决定）。
def main(args: Args) -> None:
    runtime = _runtime.Runtime(
        environment=_env.AlohaSimEnvironment(
            task=args.task,
            seed=args.seed,
        ),
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=_websocket_client_policy.WebsocketClientPolicy(
                    host=args.host,
                    port=args.port,
                ),
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=[
            _saver.VideoSaver(args.out_dir),
        ],
        max_hz=50,
    )

    runtime.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
