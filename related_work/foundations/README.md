# 必要背景：知道每个设计从哪里来

<!-- reading-nav-start -->
[首页](../../README.md) · [背景入口](../README.md) · [按问题查找](../FIND_BY_QUESTION.md)
<!-- reading-nav-end -->

这些工作用于建立技术坐标，不需要把所有背景论文先读完再读主线。PDF 收录状态见 [SOURCES](../SOURCES.md)。

| 工作 | 机制与 workflow | 与主线的联系 | 读什么 / 不要误读 |
|---|---|---|---|
| [ACT / ALOHA](https://arxiv.org/abs/2304.13705) · [PDF](https://arxiv.org/pdf/2304.13705) | 遥操作示范→条件生成动作块→滚动执行 | action chunking、双臂操作 | 看动作块与 temporal ensembling；ACT 不是完整互联网预训练 VLA |
| [Diffusion Policy](https://arxiv.org/abs/2303.04137) · [PDF](https://arxiv.org/pdf/2303.04137) | 视觉条件→迭代去噪动作序列→receding horizon | π0 连续生成与多模态动作分布的背景 | 看生成目标与闭环执行；diffusion 与 flow 的公式不能随意互换 |
| [RT-2](https://arxiv.org/abs/2307.15818) · [PDF](../papers/foundations/rt2.pdf) | 视觉语言与机器人数据共同训练→动作离散 token | 网络知识迁移到动作策略 | 区分语义泛化和学到全新运动技能 |
| [OpenVLA](https://arxiv.org/abs/2406.09246) · [PDF](../papers/foundations/openvla.pdf) | 预训练视觉/语言组件→多机器人动作训练→适配 | 可读代码的 VLA 对照路线 | 原始版本与后续 action-chunking 改进需分开；不能自动沿用新版本能力 |
| [Octo](https://arxiv.org/abs/2405.12213) · [PDF](../papers/foundations/octo.pdf) | 多机器人预训练→语言/目标图像条件→适配新接口 | 通用 policy、灵活观测/动作空间 | 通用机器人策略不必等同于“大 LLM 加动作 token” |

上表是对各原始论文的简要归纳，没有把不同测试集上的报告数字合成排行榜。

## 数据与评估入口

- [Open X-Embodiment](https://robotics-transformer-x.github.io/)：跨本体数据背景。下载某些数据与复现完整预训练配方是不同任务。
- [DROID](https://droid-dataset.github.io/)：真实机器人多场景数据；与 openpi 的公开适配路径相关。
- [LIBERO](https://libero-project.github.io/)：模拟环境中的任务/知识迁移评估；适合建立闭环实验流程，但不能直接代表真实家庭效果。

## 基础概念自测

1. 相机帧率、policy 推理频率、action 执行频率为什么可能不同？
2. 绝对关节目标、关节增量、末端位姿增量能否共用同一归一化？
3. 长期任务成功为什么不能只用单步动作 MSE 衡量？
4. 示范数据训练上的小 loss 为什么不保证自主 rollout 稳定？
5. goal image、language subtask 和 action chunk 分别处于哪一层？

答不出 1/2 时先补控制接口；答不出 3/4 时先补闭环评估；答不出 5 时回到 [统一 workflow](../WORKFLOW.md)。

<!-- reading-footer-start -->
[返回背景入口](../README.md)
<!-- reading-footer-end -->
