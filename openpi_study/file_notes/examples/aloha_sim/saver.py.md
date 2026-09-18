# `examples/aloha_sim/saver.py` 中文阅读说明

**定位：** 评测记录。

订阅运行事件，把每步图像累积成episode视频。

**建议读法：** 开始清空帧列表；每步收集帧；结束写视频。

**易错点：** 视频是观察行为的辅助材料，不能替代逐次试验结果。

[注释源码](../../../code/examples/aloha_sim/saver.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/aloha_sim/saver.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [VideoSaver](../../../code/examples/aloha_sim/saver.py#L17) | 订阅运行事件，把每步图像累积成episode视频。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [VideoSaver.__init__](../../../code/examples/aloha_sim/saver.py#L23) | 初始化VideoSaver的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._out_dir、self._images、self._subsample。 |
| [VideoSaver.on_episode_start](../../../code/examples/aloha_sim/saver.py#L32) | 处理runtime广播的事件，更新本订阅者的显示/记录状态。 |
| [VideoSaver.on_step](../../../code/examples/aloha_sim/saver.py#L40) | 处理runtime广播的事件，更新本订阅者的显示/记录状态。 |
| [VideoSaver.on_episode_end](../../../code/examples/aloha_sim/saver.py#L49) | 处理runtime广播的事件，更新本订阅者的显示/记录状态。 |
