# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：部署入口｜读取命令行指定的模型配置和权重，创建Policy并启动WebSocket服务。
# 阅读顺序：Args→create_policy→create_trained_policy→WebsocketPolicyServer→serve_forever。
# 重点边界：启动服务只提供模型推理；机器人环境循环由客户端负责。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


# 【EnvMode】读取命令行指定的模型配置和权重，创建Policy并启动WebSocket服务。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


# 【Checkpoint】读取命令行指定的模型配置和权重，创建Policy并启动WebSocket服务。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


# 【Default】读取命令行指定的模型配置和权重，创建Policy并启动WebSocket服务。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


# 【Args】读取命令行指定的模型配置和权重，创建Policy并启动WebSocket服务。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


# 【create_default_policy】本函数位于“部署入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 DEFAULT_CHECKPOINT.get → _policy_config.create_trained_policy → _config.get_config 追踪具体实现。
# 输入接口：env:EnvMode；default_prompt:str | None。
# 返回类型：_policy.Policy；类型/shape约定需与调用方配套。
# 内部调用线索：DEFAULT_CHECKPOINT.get → _policy_config.create_trained_policy → _config.get_config → ValueError（含分支中的调用，实际路径由条件决定）。
def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


# 【create_policy】本函数位于“部署入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 _policy_config.create_trained_policy → _config.get_config → create_default_policy 追踪具体实现。
# 输入接口：args:Args。
# 返回类型：_policy.Policy；类型/shape约定需与调用方配套。
# 内部调用线索：_policy_config.create_trained_policy → _config.get_config → create_default_policy（含分支中的调用，实际路径由条件决定）。
def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


# 【main】本脚本入口：读取命令行指定的模型配置和权重，创建Policy并启动WebSocket服务。 Args→create_policy→create_trained_policy→WebsocketPolicyServer→serve_forever。
# 输入接口：args:Args。
# 返回类型：None；类型/shape约定需与调用方配套。
# 内部调用线索：create_policy → _policy.PolicyRecorder → socket.gethostname → socket.gethostbyname → logging.info（含分支中的调用，实际路径由条件决定）。
def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
