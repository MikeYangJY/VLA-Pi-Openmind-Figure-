# `src/openpi/serving/websocket_policy_server.py` 中文阅读说明

**定位：** 网络服务。

接收客户端观测，反序列化后调用Policy.infer，并发回动作与计时。

**建议读法：** run启动服务器；_handler先发送metadata，再循环接收/推理/发送；health_check供探活。

**易错点：** 数据序列化和网络等待在模型之外；部署延迟需要把这些部分也计入。

[注释源码](../../../../code/src/openpi/serving/websocket_policy_server.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/serving/websocket_policy_server.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [WebsocketPolicyServer](../../../../code/src/openpi/serving/websocket_policy_server.py#L22) | 模型服务端；真正的机器人控制循环在客户端。 |
| [WebsocketPolicyServer.__init__](../../../../code/src/openpi/serving/websocket_policy_server.py#L32) | 初始化WebsocketPolicyServer的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._policy、self._host、self._port、self._metadata。 |
| [WebsocketPolicyServer.serve_forever](../../../../code/src/openpi/serving/websocket_policy_server.py#L48) | 本函数位于“网络服务”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 asyncio.run → self.run 追踪具体实现。 |
| [WebsocketPolicyServer.run](../../../../code/src/openpi/serving/websocket_policy_server.py#L53) | 本函数位于“网络服务”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 _server.serve → server.serve_forever 追踪具体实现。 |
| [WebsocketPolicyServer._handler](../../../../code/src/openpi/serving/websocket_policy_server.py#L67) | 每个连接先发metadata，然后循环收观测、执行Policy.infer、编码发回动作；异常路径反馈错误。 |
| [_health_check](../../../../code/src/openpi/serving/websocket_policy_server.py#L108) | 本函数位于“网络服务”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 connection.respond 追踪具体实现。 |
