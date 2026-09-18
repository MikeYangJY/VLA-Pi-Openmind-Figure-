# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：ALOHA运行组装｜把真机环境、远程策略、动作缓存和runtime连接起来。
# 阅读顺序：Args配置连接；main创建环境、策略代理与显示订阅者，然后进入runtime。
# 重点边界：阅读时对照接口即可；运行会连接真实机器人。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import dataclasses
import logging

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from examples.aloha_real import env as _env


# 【Args】把真机环境、远程策略、动作缓存和runtime连接起来。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    # 字段含义：一次预测未来多少个控制时刻，与一次任务持续多久不同。
    action_horizon: int = 25

    num_episodes: int = 1
    max_episode_steps: int = 1000


# 【main】本脚本入口：把真机环境、远程策略、动作缓存和runtime连接起来。 Args配置连接；main创建环境、策略代理与显示订阅者，然后进入runtime。
# 输入接口：args:Args。
# 返回类型：None；类型/shape约定需与调用方配套。
# 内部调用线索：_websocket_client_policy.WebsocketClientPolicy → logging.info → ws_client_policy.get_server_metadata → _runtime.Runtime → _env.AlohaRealEnvironment（含分支中的调用，实际路径由条件决定）。
def main(args: Args) -> None:
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

    metadata = ws_client_policy.get_server_metadata()
    runtime = _runtime.Runtime(
        environment=_env.AlohaRealEnvironment(reset_position=metadata.get("reset_pose")),
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=[],
        max_hz=50,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    runtime.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
