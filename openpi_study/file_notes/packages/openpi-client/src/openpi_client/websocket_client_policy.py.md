# `packages/openpi-client/src/openpi_client/websocket_client_policy.py` 中文阅读说明

**定位：** 远程推理客户端。

连接模型服务，将观测编码发送并把返回值解码为动作。

**建议读法：** 等待连接→读取metadata→infer发送/接收→调用方执行动作。

**易错点：** 网络失败、模型错误与机器人失败是三个不同层级。

[注释源码](../../../../../code/packages/openpi-client/src/openpi_client/websocket_client_policy.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/packages/openpi-client/src/openpi_client/websocket_client_policy.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [WebsocketClientPolicy](../../../../../code/packages/openpi-client/src/openpi_client/websocket_client_policy.py#L19) | 将策略调用通过网络发给服务端。 |
| [WebsocketClientPolicy.__init__](../../../../../code/packages/openpi-client/src/openpi_client/websocket_client_policy.py#L29) | 初始化WebsocketClientPolicy的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._uri、self._packer、self._api_key。 |
| [WebsocketClientPolicy.get_server_metadata](../../../../../code/packages/openpi-client/src/openpi_client/websocket_client_policy.py#L42) | 本函数位于“远程推理客户端”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。 |
| [WebsocketClientPolicy._wait_for_server](../../../../../code/packages/openpi-client/src/openpi_client/websocket_client_policy.py#L48) | 本函数位于“远程推理客户端”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 logging.info → websockets.sync.client.connect → msgpack_numpy.unpackb 追踪具体实现。 |
| [WebsocketClientPolicy.infer](../../../../../code/packages/openpi-client/src/openpi_client/websocket_client_policy.py#L67) | MessagePack编码观测→通过连接发送→接收返回值→解码数组；字符串返回按服务错误处理。 |
| [WebsocketClientPolicy.reset](../../../../../code/packages/openpi-client/src/openpi_client/websocket_client_policy.py#L79) | 清理上一episode/调用周期留下的状态，或调用底层环境进行重置。 |
