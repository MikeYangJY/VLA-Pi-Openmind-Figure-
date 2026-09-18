# `examples/aloha_sim/main.py` 中文阅读说明

**定位：** 仿真运行组装。

把仿真环境、WebSocket策略与视频保存订阅者接入runtime。

**建议读法：** 按Args创建环境和策略，设置控制频率与episode数，再运行。

**易错点：** 仿真客户端和模型服务可以用不同Python环境。

[注释源码](../../../code/examples/aloha_sim/main.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/aloha_sim/main.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [Args](../../../code/examples/aloha_sim/main.py#L22) | 把仿真环境、WebSocket策略与视频保存订阅者接入runtime。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [main](../../../code/examples/aloha_sim/main.py#L42) | 本脚本入口：把仿真环境、WebSocket策略与视频保存订阅者接入runtime。 按Args创建环境和策略，设置控制频率与episode数，再运行。 |
