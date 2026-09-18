# `packages/openpi-client/src/openpi_client/runtime/subscriber.py` 中文阅读说明

**定位：** 运行事件接口。

定义episode开始、每步和结束的回调，供视频与日志工具订阅。

**建议读法：** 先看三个回调，再看VideoSaver/VideoDisplay的实现。

**易错点：** 旁路记录器可以观察控制过程，但不是策略本体。

[注释源码](../../../../../../code/packages/openpi-client/src/openpi_client/runtime/subscriber.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/packages/openpi-client/src/openpi_client/runtime/subscriber.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [Subscriber](../../../../../../code/packages/openpi-client/src/openpi_client/runtime/subscriber.py#L11) | 定义episode开始、每步和结束的回调，供视频与日志工具订阅。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [Subscriber.on_episode_start](../../../../../../code/packages/openpi-client/src/openpi_client/runtime/subscriber.py#L20) | 处理runtime广播的事件，更新本订阅者的显示/记录状态。 |
| [Subscriber.on_step](../../../../../../code/packages/openpi-client/src/openpi_client/runtime/subscriber.py#L27) | 处理runtime广播的事件，更新本订阅者的显示/记录状态。 |
| [Subscriber.on_episode_end](../../../../../../code/packages/openpi-client/src/openpi_client/runtime/subscriber.py#L33) | 处理runtime广播的事件，更新本订阅者的显示/记录状态。 |
