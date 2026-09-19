# RTC / Training-time RTC

<!-- reading-nav-start -->
[首页](../../README.md) · [PI入口](../pi/README.md) · [按问题查找](../FIND_BY_QUESTION.md)
<!-- reading-nav-end -->

**核心大纲：** 旧动作持续执行 + 新动作前缀约束 + 延迟对齐。

**来源类型：** 论文两篇 · [原始来源](https://arxiv.org/abs/2506.07339) · [展开阅读](../pi/08_supporting_work.md)

## 1）要解决的问题

预测动作块需要时间；同步等待会停顿，朴素异步切换又可能造成轨迹不连续。

## 2）方法

原始RTC在推理时把已承诺执行的前缀作为约束，补全新动作块。后续Training-time RTC在训练时模拟延迟并条件化动作前缀，减少推理时处理开销。

## 3）实验与结论

在不同延迟条件下比较同步、异步与RTC方案，分别看成功率、完成时间及计算开销。两篇应按是否修改训练区分；π0.7采用后续训练时方案。

**必须保留的边界：** 简单缓存一个action chunk并逐步执行不是RTC；openpi的ActionChunkBroker不要据此称为RTC实现。

<!-- reading-footer-start -->
[接着读：机制与数据](../MECHANISMS.md) · [返回PI入口](../pi/README.md)
<!-- reading-footer-end -->
