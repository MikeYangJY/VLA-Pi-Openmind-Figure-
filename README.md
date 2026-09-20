# VLA 机器人学习与调研

**从“机器人怎么学会做事”，走到“能看懂论文、代码，并问清专家”。** 覆盖 Physical Intelligence（PI）、Google DeepMind 和 Figure；代码只学习 **π0、π0.5**。

## 第一次读，先从这里开始

**[零基础入门：从“看见杯子”到“把杯子放进柜子”](related_work/00_START_HERE.md)**

先记一条线：**看见环境和听懂指令 → 决定当前做什么 → 生成动作数字 → 控制器执行 → 再看结果。** 各家的差别在于怎样分工，以及用什么数据把这些能力教出来。

| 你会遇到的名字 | 先这样理解 |
|---|---|
| **[Transformer](related_work/00_START_HERE.md#transformer)** | 让文字、图像等数字表示结合上下文、逐层加工的核心网络架构 |
| PI的 **action expert** | 专门生成动作的模型模块 |
| **flow matching** | 教模型把随机数字逐步变成动作的方法；随机数字不会先被机器人执行 |
| Gemini的 **ER** | 理解真实场景、安排步骤、检查进度的具身推理模型 |
| Helix的 **S2 / S1 / S0** | 理解意图 / 产生运动目标 / 协调全身和平衡；S0在Helix 02中加入 |

入门页按 **基本分工 → Transformer → 训练是什么 → PI → Gemini → Figure → 数据差异** 讲解，每个主要模型都有“核心结构＋训练数据”的白话对照。先读它，再看下面的论文、对比和代码。具体依据及未公开边界均列在[入门正文](related_work/00_START_HERE.md)。

## 入门之后，按这六步深入

| 顺序 | 点这里 | 看完要能回答 |
|---|---|---|
| ① 建立框架 | [一张图看懂机器人系统](related_work/00_OVERVIEW.md) | 训练、规划、动作生成、控制和评估各做什么？ |
| ② 看核心差异 | [八模型对照表](related_work/MODEL_COMPARISON_8.md) | 相对π0，各家改了哪里？哪些数字能比？ |
| ③ 按公司读 | [Figure](related_work/figure/README.md) → [DeepMind](related_work/deepmind/README.md) → [PI](related_work/pi/README.md) | 每项工作解决什么问题，用什么方法，证据是什么？ |
| ④ 学懂关键机制 | [训练、数据、记忆与世界模型](related_work/MECHANISMS.md) | 动作怎么学、数据怎么处理、过去和未来信息怎么用？ |
| ⑤ 跟一次代码 | [openpi：只学π0与π0.5](openpi_study/README.md) | 一条观测如何变成动作？一次训练更新在哪里发生？ |
| ⑥ 准备访谈 | [从公开证据到专家问题](related_work/INTERVIEW_PREP.md) | 已知什么、还缺什么，如何追问到可核对的答案？ |

即使已经读过π0、π0.5、π0.7，也可以先用入门页重新串起术语。理解分工后，从[Figure学习入口](related_work/figure/README.md)继续；需要补机制时回④。完整安排及每一步的自测见[学习路线](related_work/LEARNING_PATH.md)。

## 带着问题来，直接查

- **想听懂 Figure 每代变了什么：** [白话迭代说明](related_work/figure/04_evolution_plain_language.md)，按“遇到什么问题 → 增加什么 → 解决到哪里”阅读。
- **想弄懂 Index 怎样处理数据：** [官方管线逐步拆解](related_work/figure/05_index_pipeline.md)，用叠毛巾例子解释过滤、反作弊、去重、再平衡与标注。
- **想查各家具体做过什么：** [工业 / 家庭任务清单](related_work/SCENARIOS_BY_MODEL.md)，逐模型区分具体任务、评测、演示和现场运行。
- **准备采访 PI：** [60 分钟 PI 访谈提纲](related_work/pi/12_pi_interview_guide.md)：12 个必问覆盖全部客户需求；每题只看“对应客户、主问、漏答时补齐”。
- **只想快速看一项工作：** [18篇简洁笔记](related_work/quick_notes/README.md)，每篇都是“大纲 → 问题 → 方法 → 实验”。
- **想比较模型：** [八模型表](related_work/MODEL_COMPARISON_8.md) · [Excel可打开的CSV](related_work/data/model_comparison_8.csv) · [系统流程比较](related_work/COMPARISON.md)。
- **卡在某个概念：** [按问题查找](related_work/FIND_BY_QUESTION.md) · [术语速查](openpi_study/GLOSSARY.md)。
- **要找原论文或某个文件：** [论文与官方来源](related_work/SOURCES.md) · [完整内容目录](related_work/CONTENTS.md)。

## 文件分工

```text
README.md                 白话速览与阅读入口
related_work/             学习路线、对比、机制与访谈准备
  00_START_HERE.md        零基础：架构、训练与数据差异
  quick_notes/           每项工作先读这一页
  figure/ deepmind/ pi/   分公司精读笔记，各有阅读入口
  papers/                原始PDF，按笔记需要回查
openpi_study/             π0 / π0.5代码学习
  code/                  带中文注释的固定版本源码
  file_notes/            每个文件的说明，通过FILE_INDEX查找
```

本库区分**论文报告、官方披露、学习解释、未披露项**。入门更新日期：2026-09-20；技术事实以各篇注明的资料版本为准。仓库名保留Openmind，本文的Gemini Robotics均指 **Google DeepMind**。

[资料索引](related_work/README.md) · [更新记录](related_work/CHANGELOG.md)
