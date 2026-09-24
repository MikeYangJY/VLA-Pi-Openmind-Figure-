# 按问题查找：不知道文件名，也能找到答案

| 你现在想知道 | 先读哪一页 | 继续到哪里 |
|---|---|---|
| Transformer到底做什么，token和attention是什么？ | [零基础Transformer](00_START_HERE.md#transformer) | 同节选读Q/K/V，再连接VLM、action expert与flow matching |
| 模型术语太多，我想从零理清架构和数据 | [零基础入门](00_START_HERE.md) | PI → Gemini → Figure，每代结构与数据一起看 |
| VLA、规划器、动作expert、控制器是什么关系？ | [系统总览](00_OVERVIEW.md) | [系统流程比较](COMPARISON.md) |
| 八个模型的架构、数据和能力差在哪？ | [核心对照表](MODEL_COMPARISON_8.md) | [完整CSV](data/model_comparison_8.csv) |
| 我想先学Figure，从哪一代看起？ | [Figure路线](figure/README.md) | Helix → 02 → Index → 2.5 |
| Figure 每代到底增加什么、解决什么问题？ | [白话迭代说明](figure/04_evolution_plain_language.md) | 模型、机器人身体、数据计划分开理解 |
| Index 具体怎样清洗、去重和平衡数据？ | [五步管线拆解](figure/05_index_pipeline.md) | 官方图示、叠毛巾例子、数据字段与未披露项 |
| 三家各模型做过哪些工业、家庭任务？ | [场景任务清单](SCENARIOS_BY_MODEL.md) | 公司 → 模型 → 具体任务 → 证据类型与适配条件 |
| Helix的S2/S1/S0分别负责什么？ | [Helix 02简洁笔记](quick_notes/12_helix02.md) | [全身控制解释](figure/02_helix02.md) |
| Index人类数据为什么有用？ | [Index简洁笔记](quick_notes/14_index.md) | [Helix 2.5训练/评估路径](figure/03_data_and_helix25.md) |
| Gemini的ER与VLA，谁规划、谁输出动作？ | [DeepMind路线](deepmind/README.md) | [Gemini 1.5详解](deepmind/02_gemini_robotics_15.md) |
| SayCan、RT-1、PaLM-E、RT-2 与 Gemini 是什么关系？ | [DeepMind 技术演进](deepmind/00_technical_lineage.md) | 每一步的问题、方法、数据与边界 |
| ER coding 怎么看？Bounding box 怎样变成坐标？ | [ER notebook 逐段导读](deepmind/04_er_notebook_walkthrough.md) | 手算坐标、mock 工具、成功判断与视频进度 |
| Gemini 怎样训练，RT-X 数据够不够？ | [训练与数据](deepmind/05_training_and_data.md) | 动作标签、跨本体差异、MT 和未披露配方 |
| Gemini 的 ICL、微调、Genie 世界模型有何区别？ | [机制与实验口径](deepmind/06_icl_world_models_and_evaluation.md) | ICL 原表、参数更新、world model 用途、success 与 progress |
| 怎样准备与 Google DeepMind 专家沟通？ | [客户问题→已知→追问](deepmind/07_expert_conversation.md) | 一小时路线、术语速查与访谈前自测 |
| π0.6、π0.6*、MEM是不是三个依次升级的版本？ | [家族区别](pi/09_pi06_family.md) | [RECAP](pi/05_recap.md) · [MEM](pi/06_mem.md) |
| flow matching是什么，需要哪些动作数据？ | [零基础教程](../openpi_study/07_FLOW_MATCHING_FROM_ZERO.md) | [π0代码讲解](../openpi_study/03_PI0_WALKTHROUGH.md) |
| π0.7世界模型和动作expert训练有什么不同？ | [两种生成目标对照](pi/10_pi07_world_model_vs_action_expert.md) | [π0.7整体结构](pi/07_pi07.md) |
| FAST和KI为什么出现？ | [FAST](quick_notes/02_fast.md) → [KI](quick_notes/04_ki.md) | [PI阅读入口](pi/README.md) |
| Kinematics sanity check检查什么？ | [数据质量教程第3节](pi/11_data_quality_three_concepts.md#3-kinematics-sanity-check检查运动记录是否说得通) | 同页延迟分析与访谈问题 |
| 多摄像头怎样判断成功？ | [数据质量教程第4节](pi/11_data_quality_three_concepts.md#4-多摄像头状态变化验证成功从变化到目标成立) | 任务条件、遮挡、时序与误判 |
| 视觉归一化到底在哪些代码执行？ | [数据质量教程第5节](pi/11_data_quality_three_concepts.md#5-视觉归一化沿-openpi-的真实代码看一遍) | 同页固定版本代码链接 |
| 模型预测了一段动作，机器人怎么逐步执行？ | [π0代码讲解](../openpi_study/03_PI0_WALKTHROUGH.md) | [RTC](quick_notes/16_rtc.md) |
| 我是代码初学者，先看哪个文件？ | [代码阅读顺序](../openpi_study/02_START_HERE.md) | [代码树](../openpi_study/01_CODE_TREE.md) |
| 某个openpi文件是做什么的？ | [逐文件目录](../openpi_study/FILE_INDEX.md) | 对应中文说明与注释源码 |
| π0.5在代码里改了什么？ | [沿开关看差异](../openpi_study/04_PI05_DIFF.md) | [训练与部署](../openpi_study/05_TRAINING_AND_DEPLOYMENT.md) |
| 一个术语是什么意思？ | [术语表](../openpi_study/GLOSSARY.md) | 不用先读完整张表 |
| 如何把这些知识用于访谈？ | [访谈准备](INTERVIEW_PREP.md) | 公开回答 → 边界 → 具体追问 |
| 这次采访 PI，怎样覆盖客户问题并追问？ | [60 分钟 PI 访谈提纲](pi/12_pi_interview_guide.md) | 12 个必问覆盖全部需求；逐题标注客户原问题与漏答核对项 |
| 想找PDF、出处、图号或许可信息？ | [原始来源](SOURCES.md) | [源码版本/许可](../openpi_study/README.md) |
| 读完以后能做什么小实验？ | [可选小实验](RESEARCH_PRACTICE.md) | [研究问题](RESEARCH_QUESTIONS.md) · [复现](REPRODUCTION.md) |

[完整内容目录](CONTENTS.md) · [顺序学习](LEARNING_PATH.md) · [首页](../README.md)
