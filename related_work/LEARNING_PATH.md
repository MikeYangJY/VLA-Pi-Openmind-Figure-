# 学习路线：按六步走，每步只带走一个结果

这是全库统一的阅读顺序。即使已经读过论文，如果action expert、flow matching、ER等术语仍然混乱，先读[零基础入门](00_START_HERE.md)，把“模块、训练方法、数据”分开。**当前目标是理解并能调研/访谈；实验复现放在后面选做。**

## 主线

| 顺序 | 必读入口 | 本轮做到什么就够了 | 深读到哪里 |
|---|---|---|---|
| ① 框架 | [系统总览](00_OVERVIEW.md) | 用杯子例子分清训练、推理、执行、成功判断 | 不熟悉时再查术语 |
| ② 对比 | [八模型表](MODEL_COMPARISON_8.md) | 指出相对π0的主要变化；识别不可直接比较的成功率 | 需要某数字时再点原文 |
| ③ 公司 | [Figure](figure/README.md) → [DeepMind](deepmind/README.md) → [PI](pi/README.md) | 每项工作写出“问题、方法、实验”三句话 | 简洁笔记 → 详细笔记 → 原文相关图表 |
| ④ 机制 | [机制与数据路线](MECHANISMS.md) | 解释动作标签、噪声、记忆、未来目标图各是什么 | 完成flow手算与一条轨迹拆样本 |
| ⑤ 代码 | [openpi入口](../openpi_study/README.md) | 跟通一次推理和一次训练更新，指出π0.5差异 | 只沿π0/π0.5主线，其他文件按需查 |
| ⑥ 访谈 | [访谈准备](INTERVIEW_PREP.md) | 为一个客户问题列出公开答案、边界和待问项 | 按证据与未知点补问题，不先猜答案 |

## 你下一次学习直接这样开始

1. 先读[零基础入门](00_START_HERE.md)并回答末尾四个问题，再扫[系统图](00_OVERVIEW.md)。[八模型表](MODEL_COMPARISON_8.md)留到需要查具体数字时。
2. 进入[Figure路线](figure/README.md)：Helix → Helix 02 → Index → Helix 2.5。先理解架构，再理解为什么扩展数据。
3. 带着同样问题读[DeepMind路线](deepmind/README.md)：谁规划、谁生成动作、跨机器人迁移怎么证明。
4. 回[PI路线](pi/README.md)补KI、RECAP、MEM及世界模型；已熟悉的π0/π0.5/π0.7只需回查。
5. 进入[机制路线](MECHANISMS.md)，先算懂flow，再读代码。

准备近期访谈时，可以提前使用⑥；它不要求你先读完所有源码。

## 每一项工作只读三层

| 层 | 用哪种文件 | 什么时候停 |
|---|---|---|
| 第一层：知道它做什么 | [简洁笔记](quick_notes/README.md) | 能复述问题、方法、实验即可 |
| 第二层：理解为什么 | `pi/`、`deepmind/`、`figure/`里的详细笔记 | 能画出数据流，并指出证据边界 |
| 第三层：核对原文 | 笔记链接的论文/官方页 | 核对与你问题有关的章节、图表、单位和条件 |

简洁笔记与详细笔记是同一项工作的两种阅读深度；**不需要把它们当成两份独立课程重读**。文件名编号是定位用的，阅读顺序按各入口页。

## 卡住时退到哪里？

- 听不懂名词 → [术语表](../openpi_study/GLOSSARY.md)。
- 不知道模型到底改了什么 → [三路线流程比较](COMPARISON.md)。
- 不明白训练需要哪些字段 → [flow训练样本](../openpi_study/07_FLOW_MATCHING_FROM_ZERO.md)与[数据质量三概念](pi/11_data_quality_three_concepts.md)。
- 代码太多 → [只跟一次推理](../openpi_study/02_START_HERE.md)。
- 想核对某个具体问题 → [按问题找资料](FIND_BY_QUESTION.md)。

## 主线完成后再选做

[背景论文](foundations/README.md)按缺口补；[小实验](RESEARCH_PRACTICE.md)、[研究问题](RESEARCH_QUESTIONS.md)、[复现路线](REPRODUCTION.md)用于下一阶段。无需为了准备调研先从头训练一个VLA。

[开始①系统总览](00_OVERVIEW.md) · [回首页](../README.md)
