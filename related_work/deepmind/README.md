# DeepMind：从零看懂机器人，到能与专家讨论

[全库首页](../../README.md) · [按问题找答案](../FIND_BY_QUESTION.md) · [本模块原文与来源](08_sources.md)

**先理解“谁负责什么”，再理解“怎么学会”，最后判断“实验究竟证明了什么”。** 本模块把前期学习中的 SayCan、RT 系列、Gemini Robotics、ER coding、数据与 ICL 串成一条线。这里的团队是 **Google DeepMind**。

## 第一次读，按这个顺序

| 顺序 | 阅读内容 | 看完要能回答 |
|---|---|---|
| 1，先读 | [用收杯子看懂 ER、VLA 和控制系统](00_beginner_map.md) | 谁看懂、谁规划、谁生成动作、谁执行？ |
| 2 | [技术路线：SayCan → RT-1 → PaLM-E → RT-2／RT-X](00_technical_lineage.md) | 为什么需要从“选已有技能”走向通用动作学习？ |
| 3 | [Gemini Robotics 初代](01_gemini_robotics.md) | ER 代码控制、ICL 与 VLA 是哪几条不同路径？ |
| 4，重点 | [Gemini Robotics 1.5](02_gemini_robotics_15.md) | ER 编排、Thinking VLA、Motion Transfer 各改变什么？ |
| 5 | [Gemini Robotics 2 与部署／推理分支](03_updates.md) | GR 2、ER 2、On-Device 2 怎样分工，哪些能力已披露？ |
| 6，重点 | [官方 ER notebook 逐段导读](04_er_notebook_walkthrough.md) | 图片怎样变成点、框、成功判断和工具调用？ |
| 7 | [训练和数据](05_training_and_data.md) | 网络数据、遥操作、人类视频分别教什么，缺什么？ |
| 8 | [ICL、世界模型与评测](06_icl_world_models_and_evaluation.md) | 什么真的学了？什么只是上下文？成功率能否比较？ |
| 9，访谈前读 | [专家沟通与客户问题映射](07_expert_conversation.md) | 公开资料已有答案是什么，下一句该追问什么？ |

**时间有限：** 第 1 → 4 → 6 → 9 步先建立骨架，再按不懂的概念回查。**系统学习：** 分几次走完整条线，每次能回答表中的一个问题即可；不必一次读完所有原文。

文件编号保留旧链接兼容，**阅读顺序以这张表为准**。

## 先抓住模型家族的分工

    用户任务 + 场景反馈
            ↓
    ER：理解、规划、选择工具、检查结果
            ↓ 短指令或工具请求
    VLA／已有机器人技能：产生具体动作
            ↓
    机器人控制系统：执行 → 新观测 → 返回上层

这画的是一种常见组合。VLA 也可直接接收指令；ER 可调用自定义工具。[初代](01_gemini_robotics.md)和[1.5](02_gemini_robotics_15.md)分别解释实际公开架构。

## 带着问题查，直接跳转

| 你卡住的地方 | 对应页面 |
|---|---|
| Transformer、VLM 和动作生成方法是什么关系？ | [基础分工](00_beginner_map.md) · [Transformer 概念课](../00_START_HERE.md#transformer) |
| RT-2 和 π0 为什么都叫 VLA，却不一样？ | [技术演进第 3 节](00_technical_lineage.md) |
| 云端 VLA 与本地 action decoder 怎样配合？ | [初代架构与 50 Hz 的解释](01_gemini_robotics.md) |
| 框住物体，为什么还不能直接抓？ | [ER 代码里的坐标手算](04_er_notebook_walkthrough.md) |
| 几个示范就能适配，是 ICL 吗？ | [ICL 与微调、实验边界](06_icl_world_models_and_evaluation.md) |
| Motion Transfer 是否只是把动作坐标统一？ | [数据与跨本体训练](05_training_and_data.md) |
| Genie 已经与 Gemini Robotics 联训了吗？ | [世界模型：已知与待确认](06_icl_world_models_and_evaluation.md) |
| 工业与家庭具体做过什么？ | [工业清单](../SCENARIOS_BY_MODEL.md#deepmind-industry) · [家庭清单](../SCENARIOS_BY_MODEL.md#deepmind-home) |
| 发布访谈有什么增量？ | [访谈观点、原链接与时间点](09_public_interviews.md) |
| 要与专家讨论，但怕听不懂术语 | [术语→白话→一句追问](07_expert_conversation.md) |

## 怎样使用证据

本模块分别标注 **论文结果、官方披露、访谈观点、教学解释和未公开项**。例如，ER 问答准确率不等于机器人任务成功率；On-Device 适配所需示范数不等于基座训练数据总量；公开视频演示不等于长期商业部署。

查每个数字时，都看它属于哪个模型、哪台机器人、什么任务、是否额外训练、用什么指标。原文统一在[来源索引](08_sources.md)。最近核对：**2026-09-24**。

## 原有简洁笔记继续保留

[初代一页版](../quick_notes/08_gemini.md) · [1.5 一页版](../quick_notes/09_gemini15.md) · [2 系列一页版](../quick_notes/10_gemini2.md)。它们是同一内容的速读入口，不必与详细笔记重复从头阅读。

[开始第 1 步](00_beginner_map.md) · [返回全库学习路线](../LEARNING_PATH.md) · [Figure](../figure/README.md) · [PI](../pi/README.md)
