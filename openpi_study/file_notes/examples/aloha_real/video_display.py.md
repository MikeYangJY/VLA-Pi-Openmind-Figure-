# `examples/aloha_real/video_display.py` 中文阅读说明

**定位：** 观测可视化。

订阅episode与step事件，实时展示相机画面。

**建议读法：** episode开始初始化窗口；每步更新画面；结束清理。

**易错点：** 显示画面不修改策略权重，也不能代替成功率评价。

[注释源码](../../../code/examples/aloha_real/video_display.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/aloha_real/video_display.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [VideoDisplay](../../../code/examples/aloha_real/video_display.py#L14) | 订阅episode与step事件，实时展示相机画面。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [VideoDisplay.__init__](../../../code/examples/aloha_real/video_display.py#L19) | 初始化VideoDisplay的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._ax、self._plt_img。 |
| [VideoDisplay.on_episode_start](../../../code/examples/aloha_real/video_display.py#L27) | 处理runtime广播的事件，更新本订阅者的显示/记录状态。 |
| [VideoDisplay.on_step](../../../code/examples/aloha_real/video_display.py#L37) | 处理runtime广播的事件，更新本订阅者的显示/记录状态。 |
| [VideoDisplay.on_episode_end](../../../code/examples/aloha_real/video_display.py#L53) | 处理runtime广播的事件，更新本订阅者的显示/记录状态。 |
