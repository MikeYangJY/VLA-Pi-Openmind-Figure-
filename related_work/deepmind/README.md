# ③-B DeepMind：先分清ER与VLA，再看版本变化

Gemini Robotics在本库指 **Google DeepMind**。这一轮重点是模型如何分工、如何跨机器人迁移，以及实验究竟证明了什么。

**先查具体做过什么：** [DeepMind 工业任务](../SCENARIOS_BY_MODEL.md#deepmind-industry) / [DeepMind 家庭任务](../SCENARIOS_BY_MODEL.md#deepmind-home)。工业重点查工具取放、连接、皮带装配、配套与插入；家庭重点查收纳、擦拭、打包和全身取放。独立 VLA 与 ER＋VLA 整系统分开记录。

## 按这个顺序读

| 顺序 | 先读：简洁笔记 | 想弄懂时展开 | 本轮只回答 |
|---|---|---|---|
| 1 | [Gemini Robotics](../quick_notes/08_gemini.md) | [ER与VLA的边界](01_gemini_robotics.md) | ER理解/推理与VLA生成动作分别负责什么？ |
| 2，重点 | [Gemini Robotics 1.5](../quick_notes/09_gemini15.md) | [编排器、Thinking VLA、Motion Transfer](02_gemini_robotics_15.md) | 三种机制各改善什么，如何通过实验区分？ |
| 3 | [Gemini Robotics 2系列](../quick_notes/10_gemini2.md) | [GR 2及相关更新](03_updates.md) | VLA、ER 2、On-Device 2分别是什么，哪些数字属于哪个模型？ |

先走主线，再按需要查[On-Device与ER 1.6](03_updates.md)。它们是部署/推理分支，不必混进一条简单的“动作模型版本升级”链。

## 三个需要反复核对的边界

| 容易混的内容 | 阅读时问自己 |
|---|---|
| ER模型与动作VLA | 这个结果测的是规划/成功判断，还是实际完成任务？ |
| Thinking与Motion Transfer | 一个是动作前的推理能力，一个是跨本体学习机制；消融分别怎么做？ |
| progress与success | 正文图和附录图是否用了不同指标？ |

核对成功率直接看[八模型表](../MODEL_COMPARISON_8.md)与其中的论文图号；不要从完成进度推算整任务成功率。

## 看完的产出

画出“用户任务 → ER组织任务 → VLA动作 → 机器人反馈”的图，再用一条旁注解释Thinking VLA。随后与Figure的S2/S1/S0比较：**模块名字不决定它们能否一一对应，输入输出和职责才决定。** [已有流程对照](../COMPARISON.md)

**上一站：** [Figure](../figure/README.md) · **下一站：** [③-C PI](../pi/README.md) · [返回学习路线](../LEARNING_PATH.md)
