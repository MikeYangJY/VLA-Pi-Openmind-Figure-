# `examples/aloha_real/main.py` 中文阅读说明

**定位：** ALOHA运行组装。

把真机环境、远程策略、动作缓存和runtime连接起来。

**建议读法：** Args配置连接；main创建环境、策略代理与显示订阅者，然后进入runtime。

**易错点：** 阅读时对照接口即可；运行会连接真实机器人。

[注释源码](../../../code/examples/aloha_real/main.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/aloha_real/main.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [Args](../../../code/examples/aloha_real/main.py#L21) | 把真机环境、远程策略、动作缓存和runtime连接起来。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [main](../../../code/examples/aloha_real/main.py#L36) | 本脚本入口：把真机环境、远程策略、动作缓存和runtime连接起来。 Args配置连接；main创建环境、策略代理与显示订阅者，然后进入runtime。 |
