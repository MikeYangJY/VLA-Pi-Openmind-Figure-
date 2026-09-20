# ③-C PI：以π0为基准，按“改变了什么”阅读

你已精读π0、π0.5、π0.7，因此这里优先帮助你补齐中间机制。**先看工作分工，计算与数据细节再到④机制路线展开。**

**准备采访 PI：** [60 分钟问卷：9 个必问＋6 个补充](12_pi_interview_guide.md)。先问★必问主线，再用余量补问；每题标有时间上限和必追一句，附超时删减顺序。

## 核心主线

| 顺序 | 先读 | 展开阅读 | 核心问题 |
|---|---|---|---|
| 1，回查 | [π0简洁笔记](../quick_notes/01_pi0.md) | [π0详细笔记](01_pi0.md) | 怎样把VLM与连续动作连接？ |
| 2，回查 | [π0.5简洁笔记](../quick_notes/03_pi05.md) | [π0.5详细笔记](03_pi05.md) | 为什么异构数据和高层子任务有助于新家庭泛化？ |
| 3，补机制 | [FAST](../quick_notes/02_fast.md) → [KI](../quick_notes/04_ki.md) | [FAST详解](02_fast.md) · [KI详解](04_knowledge_insulation.md) | 动作如何编码，训练时哪些梯度应该相互影响？ |
| 4，先分名字 | [π0.6 / π0.6* / MEM对照](09_pi06_family.md) | [RECAP](05_recap.md) · [MEM](06_mem.md) | 基础策略、从经验学习、记忆增强分别增加什么？ |
| 5，重新串联 | [π0.7简洁笔记](../quick_notes/07_pi07.md) | [π0.7详细笔记](07_pi07.md) | 历史、metadata、语言和图像目标如何配合？ |

第4步如果对照页读不顺，可先看[RECAP简洁笔记](../quick_notes/05_recap.md)和[MEM简洁笔记](../quick_notes/06_mem.md)。

## 三个初学者重点专题

| 现在卡在哪里 | 直接去这里 |
|---|---|
| action expert为什么能从噪声得到动作 | [flow matching手算教程](../../openpi_study/07_FLOW_MATCHING_FROM_ZERO.md) |
| 世界模型和动作expert各学什么、需要什么数据 | [π0.7两种训练目标对照](10_pi07_world_model_vs_action_expert.md) |
| 数据怎样清洗、验证和变成图像输入 | [运动学、多摄像头与视觉归一化](11_data_quality_three_concepts.md) |

## 补充工作按问题选读

| 问题 | 简洁笔记 | 详细入口 |
|---|---|---|
| 谁把开放指令拆成子任务 | [Hi Robot](../quick_notes/15_hirobot.md) | [补充工作](08_supporting_work.md) |
| 模型计算时怎样连续执行 | [RTC与Training-time RTC](../quick_notes/16_rtc.md) | [补充工作](08_supporting_work.md) |
| 人类数据怎样帮助机器人 | [Human-to-Robot](../quick_notes/17_human_transfer.md) | [补充工作](08_supporting_work.md) |
| 怎样用小模块做在线改进 | [RL Token](../quick_notes/18_rlt.md) | [补充工作](08_supporting_work.md) |

**看完要能做到：** 把每项工作归到数据、表示/训练、记忆、经验学习或执行中的具体位置，而非只记版本号。代码学习仍只覆盖[π0与π0.5](../../openpi_study/README.md)。

**上一站：** [DeepMind](../deepmind/README.md) · **下一步：** [④ 机制与数据](../MECHANISMS.md) · [返回学习路线](../LEARNING_PATH.md)
