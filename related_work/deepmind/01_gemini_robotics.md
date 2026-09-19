# Gemini Robotics：先区分 ER 与 VLA

<!-- reading-nav-start -->
[首页](../../README.md) · [DeepMind入口](README.md) · [按问题查找](../FIND_BY_QUESTION.md)
<!-- reading-nav-end -->

[原报告](../papers/deepmind/gemini_robotics.pdf) · [arXiv](https://arxiv.org/abs/2503.20020)。团队是 **Google DeepMind**。依据报告的 Embodied Reasoning、Gemini Robotics 和评估章节。

## 核心问题

视觉语言模型中的世界知识，如何转化成可用于真实机器人操作的空间理解和动作？报告同时提出具身推理模型 Gemini Robotics-ER 与直接控制机器人的 Gemini Robotics VLA。

## 两条路线不要混写

| 对象 | 输入输出与能力 | 研究层级 |
|---|---|---|
| Gemini Robotics-ER | 从视觉与语言推理空间关系、指点、轨迹/抓取等，亦可生成调用机器人接口的代码 | 世界理解、规划与接口调用 |
| Gemini Robotics | 用视觉和指令条件产生机器人动作 | 学习得到的动作策略 |

ER 生成代码调用已有控制 API，与 VLA 端到端输出动作，是不同的控制路径。ERQA 问答成绩也不能直接换算为真实机器人成功率。

## 训练 workflow

Gemini 2.0 的多模态基础能力 → 强化/扩展具身空间与时间理解 → 加入机器人动作数据形成 VLA → 按需求对新任务或本体追加微调。报告展示了少量示范适配；这里的“少量”依赖具体任务和已有模型，不是从零学习机器人控制所需的全部数据。

公开报告未提供让外部研究者完整重建所有网络、数据混合与训练系统的全部细节。不能仅根据 PI 的熟悉架构推定 Gemini 使用同样的 flow loss 或 action expert 规格。

## 运行 workflow

观测与任务 → 语义/空间理解 → 动作生成或调用既有机器人能力 → 新观测 → 继续执行。学习时分别画 ER + 工具控制、VLA 直接控制两张图；1.5 的完整 agent 组合方式放在下一篇。

## 实验怎么读

先分开 generality、interactivity、dexterity 三种目标，再检查新物体、视觉变化、语言变化、动作/任务变化的测试边界。适配到新本体的展示，不自动意味着基础 checkpoint 对任意本体无需训练即可运行。

报告引入的 ERQA 有 400 道视觉选择题，覆盖空间、轨迹、动作、状态等推理。这个 benchmark 对研究 ER 有价值，但没有包含所有接触动力学与闭环执行问题。

## 你的阅读重点

你已粗读过 Gemini，建议略读背景，直接比较“ER 可以预测一个抓取点”和“VLA 实际完成抓取”的输入输出差异。然后进入 [Gemini Robotics 1.5](02_gemini_robotics_15.md) 的 orchestrator 与 Thinking VLA 消融。

<!-- reading-footer-start -->
[接着读：Gemini Robotics 1.5](02_gemini_robotics_15.md) · [返回DeepMind入口](README.md)
<!-- reading-footer-end -->
