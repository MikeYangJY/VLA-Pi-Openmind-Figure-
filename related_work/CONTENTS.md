# 完整内容目录

这是查文件用的全目录；顺序学习请走[六步路线](LEARNING_PATH.md)，不知道文件名时用[按问题查找](FIND_BY_QUESTION.md)。

## 先分清文件类型

| 类型 | 作用 | 使用方式 |
|---|---|---|
| 入口/路线 | 告诉你顺序和重点 | 从首页或公司README进入 |
| 简洁笔记 | 一页建立工作框架 | 问题→方法→实验 |
| 详细笔记/专题 | 解释机制和证据边界 | 卡住时展开，按目录跳读 |
| 原始论文/官方页 | 核对事实 | 从笔记中的来源链接进入 |
| 代码教程/逐文件说明 | 串起源码调用关系 | 先跟主线，再查文件 |
| 实验/研究建议 | 把理解变成可测试问题 | 主线后选读 |

## 主线与查找

- [零基础入门：机器人怎么从“看见”到“动起来”](00_START_HERE.md)
- [① 先用一个任务看懂机器人系统](00_OVERVIEW.md)
- [学习路线：按六步走，每步只带走一个结果](LEARNING_PATH.md)
- [八个模型核心对照：以 π0 为架构基准](MODEL_COMPARISON_8.md)
- [三条路线：在同一 workflow 下比较](COMPARISON.md)
- [④ 机制路线：把动作、数据、记忆和未来目标分开学](MECHANISMS.md)
- [⑥ 访谈准备：把客户问题转成可核对的技术问题](INTERVIEW_PREP.md)
- [按问题查找：不知道文件名，也能找到答案](FIND_BY_QUESTION.md)

## 公司入口

- [③-A Figure：先架构，再数据，最后泛化](figure/README.md)
- [③-B DeepMind：先分清ER与VLA，再看版本变化](deepmind/README.md)
- [③-C PI：以π0为基准，按“改变了什么”阅读](pi/README.md)

## Figure详细笔记

- [Helix：连续 latent 连接语义与高速控制](figure/01_helix.md)
- [Helix 02：把语义、移动操作与全身控制分层](figure/02_helix02.md)
- [Figure 数据主线：Go-Big → Index → Helix 2.5](figure/03_data_and_helix25.md)

## DeepMind详细笔记

- [Gemini Robotics：先区分 ER 与 VLA](deepmind/01_gemini_robotics.md)
- [Gemini Robotics 1.5：编排器、Thinking VLA 与跨本体迁移](deepmind/02_gemini_robotics_15.md)
- [DeepMind 补充更新：部署、ER 1.6 与 GR 2](deepmind/03_updates.md)

## PI详细笔记与专题

- [π0：把 VLM 的语义表征接到连续动作生成](pi/01_pi0.md)
- [FAST：动作表示决定 VLA 学起来是否高效](pi/02_fast.md)
- [π0.5：从动作能力到新环境泛化](pi/03_pi05.md)
- [Knowledge Insulation：把梯度路径当成设计对象](pi/04_knowledge_insulation.md)
- [π*0.6 / RECAP：怎样从部署经验改进 VLA](pi/05_recap.md)
- [MEM：短期视觉记忆与长期语义记忆](pi/06_mem.md)
- [π0.7：丰富上下文如何变成可引导的通用策略](pi/07_pi07.md)
- [把 PI 主线补完整的五项工作](pi/08_supporting_work.md)
- [π0.6、π0.6*、π0.6-MEM：分别改变了什么？](pi/09_pi06_family.md)
- [π0.7 世界模型与 action expert：训练方法和数据有什么不同？](pi/10_pi07_world_model_vs_action_expert.md)
- [三项数据质量概念：运动学检查、多摄像头成功验证、视觉归一化](pi/11_data_quality_three_concepts.md)
- [PI 专家访谈问卷：15 个主问题与现场追问](pi/12_pi_interview_guide.md)

## 简洁笔记

- [简洁笔记：先用一页看清一项工作](quick_notes/README.md)
- [π0](quick_notes/01_pi0.md)
- [FAST / π0-FAST](quick_notes/02_fast.md)
- [π0.5](quick_notes/03_pi05.md)
- [Knowledge Insulation（KI）](quick_notes/04_ki.md)
- [π*0.6 / RECAP](quick_notes/05_recap.md)
- [MEM / π0.6-MEM](quick_notes/06_mem.md)
- [π0.7](quick_notes/07_pi07.md)
- [Gemini Robotics（原始报告）](quick_notes/08_gemini.md)
- [Gemini Robotics 1.5](quick_notes/09_gemini15.md)
- [Gemini Robotics 2 系列](quick_notes/10_gemini2.md)
- [Helix](quick_notes/11_helix.md)
- [Helix 02](quick_notes/12_helix02.md)
- [Helix 2.5](quick_notes/13_helix25.md)
- [Index / Project Go-Big](quick_notes/14_index.md)
- [Hi Robot](quick_notes/15_hirobot.md)
- [RTC / Training-time RTC](quick_notes/16_rtc.md)
- [Human-to-Robot Transfer](quick_notes/17_human_transfer.md)
- [RL Token](quick_notes/18_rlt.md)

## 选读、调研方法与维护

- [必要背景：知道每个设计从哪里来](foundations/README.md)
- [可选进阶：从阅读到小实验](RESEARCH_PRACTICE.md)
- [可以收敛成研究课题的问题](RESEARCH_QUESTIONS.md)
- [从阅读到最小复现](REPRODUCTION.md)
- [统一调研 workflow](WORKFLOW.md)
- [论文 / 技术工作名称](templates/PAPER_NOTE.md)
- [论文与官方资料目录](SOURCES.md)
- [资料库更新记录](CHANGELOG.md)

## openpi学习

- [⑤ openpi代码学习：只学π0与π0.5](../openpi_study/README.md)
- [代码树：先分清五层](../openpi_study/01_CODE_TREE.md)
- [小白应该先看哪里，再看哪里](../openpi_study/02_START_HERE.md)
- [从零理解：action expert 的 flow matching 在学什么？](../openpi_study/07_FLOW_MATCHING_FROM_ZERO.md)
- [π0：一条观测如何变成动作](../openpi_study/03_PI0_WALKTHROUGH.md)
- [从数据、训练到部署：每一步做什么](../openpi_study/05_TRAINING_AND_DEPLOYMENT.md)
- [π0.5：只沿着开关追三条路径](../openpi_study/04_PI05_DIFF.md)
- [两个计算后端与底层组件怎样读](../openpi_study/06_BACKENDS_AND_COMPONENTS.md)
- [逐文件中文阅读目录](../openpi_study/FILE_INDEX.md)
- [随用随查的术语](../openpi_study/GLOSSARY.md)
- [验证说明](../openpi_study/VERIFICATION.md)

140个源码/配置文件及其说明统一从[FILE_INDEX](../openpi_study/FILE_INDEX.md)进入；完整文件树在[代码树](../openpi_study/01_CODE_TREE.md)。

## 原始资料与数据文件

- [论文与官方来源目录](SOURCES.md)：含15份PDF及仅链接收录的资料。
- [完整八模型CSV](data/model_comparison_8.csv)。
- [来源登记JSON](sources.json) · [BibTeX引用](references.bib)。
- [上游源码版本](../openpi_study/UPSTREAM.json) · [注释记录](../openpi_study/ANNOTATION_MANIFEST.json)。

[资料索引](README.md) · [首页](../README.md)
