# Helix 02

<!-- reading-nav-start -->
[首页](../../README.md) · [Figure入口](../figure/README.md) · [按问题查找](../FIND_BY_QUESTION.md)
<!-- reading-nav-end -->

**核心大纲：** S2目标 → S1全身关节目标 → S0执行器命令。

**来源类型：** 官方技术说明（2026-01-27） · [原始来源](https://www.figure.ai/news/helix-02) · [展开阅读](../figure/02_helix02.md)

## 1）要解决的问题

行走、操作和平衡相互耦合，分别控制再用状态机拼接容易不连贯。

## 2）方法

新增S0 learned whole-body controller，使用重定向人体运动和仿真RL；S1融合头部/掌部视觉、触觉与本体状态。S1约200Hz，S0约1kHz，承担不同层级职责。

## 3）实验与结论

官方展示约4分钟、61个移动操作步骤的洗碗机任务，以及触觉/掌部视觉支持的精细操作。它证明了一些完整行为可实现，但单次长演示不足以确定平均成功率和商业可靠性。

**必须保留的边界：** Figure 02是机器人代号，Helix 02是模型系统；S0也不等于普通PD控制器。

<!-- reading-footer-start -->
[前一篇：Helix](11_helix.md) · [接着读：Index](14_index.md) · [返回Figure入口](../figure/README.md)
<!-- reading-footer-end -->
