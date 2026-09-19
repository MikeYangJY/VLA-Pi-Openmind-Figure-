# Helix 2.5

<!-- reading-nav-start -->
[首页](../../README.md) · [Figure入口](../figure/README.md) · [按问题查找](../FIND_BY_QUESTION.md)
<!-- reading-nav-end -->

**核心大纲：** Index预训练 → 三种任务分别适配 → 未见家庭测试。

**来源类型：** 官方技术说明（2026-09-17） · [原始来源](https://www.figure.ai/news/helix-2-5-zero-shot-30-home-generalization) · [展开阅读](../figure/03_data_and_helix25.md)

## 1）要解决的问题

人形机器人能否把训练中学到的全身行为迁移到没有采集数据的新家庭和物体？

## 2）方法

基座从随机初始化用Index预训练，再分别适配整理、折毛巾和铺床；每个任务固定checkpoint跨家庭评估。

## 3）实验与结论

30个家庭测试，完整任务成功才计分；官方控制下游数据、架构和训练设置，对比随机初始化与Index预训练，报告9%→56%。另外的数据scaling实验测动作预测loss，并非商业成功率定律。

**必须保留的边界：** Zero-shot指环境与物体；不是任务完全未训练，也不是已证明单一checkpoint包办三项任务。

<!-- reading-footer-start -->
[前一篇：Index](14_index.md) · [接着读：DeepMind](../deepmind/README.md) · [返回Figure入口](../figure/README.md)
<!-- reading-footer-end -->
