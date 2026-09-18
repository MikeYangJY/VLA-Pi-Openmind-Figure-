# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：远程推理客户端｜连接模型服务，将观测编码发送并把返回值解码为动作。
# 阅读顺序：等待连接→读取metadata→infer发送/接收→调用方执行动作。
# 重点边界：网络失败、模型错误与机器人失败是三个不同层级。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import logging
import time
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


# 【WebsocketClientPolicy】将策略调用通过网络发给服务端。
class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    # 【WebsocketClientPolicy.__init__】初始化WebsocketClientPolicy的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._uri、self._packer、self._api_key。
    # 输入接口：host:str；port:Optional[int]；api_key:Optional[str]。
    # 返回类型：None；类型/shape约定需与调用方配套。
    # 内部调用线索：host.startswith → msgpack_numpy.Packer → self._wait_for_server（含分支中的调用，实际路径由条件决定）。
    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    # 【WebsocketClientPolicy.get_server_metadata】本函数位于“远程推理客户端”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。
    # 返回类型：Dict；类型/shape约定需与调用方配套。
    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    # 【WebsocketClientPolicy._wait_for_server】本函数位于“远程推理客户端”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 logging.info → websockets.sync.client.connect → msgpack_numpy.unpackb 追踪具体实现。
    # 返回类型：Tuple[websockets.sync.client.ClientConnection, Dict]；类型/shape约定需与调用方配套。
    # 内部调用线索：logging.info → websockets.sync.client.connect → msgpack_numpy.unpackb → conn.recv → time.sleep（含分支中的调用，实际路径由条件决定）。
    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    # 【WebsocketClientPolicy.infer】MessagePack编码观测→通过连接发送→接收返回值→解码数组；字符串返回按服务错误处理。
    # 输入接口：obs:Dict。
    # 返回类型：Dict；类型/shape约定需与调用方配套。
    # 内部调用线索：self._packer.pack → self._ws.send → self._ws.recv → isinstance → RuntimeError（含分支中的调用，实际路径由条件决定）。
    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    # 【WebsocketClientPolicy.reset】清理上一episode/调用周期留下的状态，或调用底层环境进行重置。
    # 返回类型：None；类型/shape约定需与调用方配套。
    @override
    def reset(self) -> None:
        pass
