# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：网络服务｜接收客户端观测，反序列化后调用Policy.infer，并发回动作与计时。
# 阅读顺序：run启动服务器；_handler先发送metadata，再循环接收/推理/发送；health_check供探活。
# 重点边界：数据序列化和网络等待在模型之外；部署延迟需要把这些部分也计入。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import asyncio
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


# 【WebsocketPolicyServer】模型服务端；真正的机器人控制循环在客户端。
class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    # 【WebsocketPolicyServer.__init__】初始化WebsocketPolicyServer的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._policy、self._host、self._port、self._metadata。
    # 输入接口：policy:_base_policy.BasePolicy；host:str；port:int | None；metadata:dict | None。
    # 返回类型：None；类型/shape约定需与调用方配套。
    # 内部调用线索：logging.getLogger('websockets.server').setLevel → logging.getLogger（含分支中的调用，实际路径由条件决定）。
    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    # 【WebsocketPolicyServer.serve_forever】本函数位于“网络服务”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 asyncio.run → self.run 追踪具体实现。
    # 返回类型：None；类型/shape约定需与调用方配套。
    # 内部调用线索：asyncio.run → self.run（含分支中的调用，实际路径由条件决定）。
    def serve_forever(self) -> None:
        asyncio.run(self.run())

    # 【WebsocketPolicyServer.run】本函数位于“网络服务”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 _server.serve → server.serve_forever 追踪具体实现。
    # 内部调用线索：_server.serve → server.serve_forever（含分支中的调用，实际路径由条件决定）。
    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    # 【WebsocketPolicyServer._handler】每个连接先发metadata，然后循环收观测、执行Policy.infer、编码发回动作；异常路径反馈错误。
    # 输入接口：websocket:_server.ServerConnection。
    # 内部调用线索：logger.info → msgpack_numpy.Packer → websocket.send → packer.pack → time.monotonic（含分支中的调用，实际路径由条件决定）。
    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


# 【_health_check】本函数位于“网络服务”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 connection.respond 追踪具体实现。
# 输入接口：connection:_server.ServerConnection；request:_server.Request。
# 返回类型：_server.Response | None；类型/shape约定需与调用方配套。
def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
