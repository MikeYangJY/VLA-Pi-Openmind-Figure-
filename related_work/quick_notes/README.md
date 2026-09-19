# 简洁笔记：先用一页看清一项工作

每篇统一为 **核心大纲 → 要解决的问题 → 方法 → 实验与结论边界**。先读这层，卡住再点篇首的“展开阅读”。以下按学习顺序分组，文件名前缀仅用于定位。

## 先读Figure

| 顺序 | 笔记 | 核心问题 |
|---|---|---|
| 1 | [Helix](11_helix.md) | S2语义如何经latent连接S1动作？ |
| 2 | [Helix 02](12_helix02.md) | S0怎样加入全身控制？ |
| 3 | [Index / Go-Big](14_index.md) | 怎样获得与利用人类行为数据？ |
| 4 | [Helix 2.5](13_helix25.md) | 预训练怎样支持未见家庭泛化？ |

详细解释和原始来源由[Figure入口](../figure/README.md)串联。

## 再读DeepMind

| 顺序 | 笔记 | 核心问题 |
|---|---|---|
| 1 | [Gemini Robotics](08_gemini.md) | ER具身推理与VLA动作分别负责什么？ |
| 2 | [Gemini Robotics 1.5](09_gemini15.md) | 编排、Thinking VLA、Motion Transfer各改了什么？ |
| 3 | [Gemini Robotics 2系列](10_gemini2.md) | VLA、ER 2与On-Device 2分别怎么定位？ |

详细解释和原始来源由[DeepMind入口](../deepmind/README.md)串联。

## 回到PI，补齐中间机制

| 顺序 | 笔记 | 核心问题 |
|---|---|---|
| 1，回查 | [π0](01_pi0.md) | VLM如何连接连续动作flow？ |
| 2，回查 | [π0.5](03_pi05.md) | 异构共训与高层子任务怎样支持新家庭？ |
| 3 | [FAST / π0-FAST](02_fast.md) | 连续动作块如何压缩成离散token？ |
| 4 | [KI](04_ki.md) | 为什么控制动作梯度流向？ |
| 5 | [π*0.6 / RECAP](05_recap.md) | 怎样从示范、自主经验和纠错中改进？ |
| 6 | [MEM](06_mem.md) | 短期视觉与长期文本分别记什么？ |
| 7，串联 | [π0.7](07_pi07.md) | 如何用丰富上下文整合多种能力？ |

读RECAP/MEM前可先看[π0.6家族对照](../pi/09_pi06_family.md)；完整顺序见[PI入口](../pi/README.md)。

## 这四篇按问题选读

| 笔记 | 什么时候看 |
|---|---|
| [Hi Robot](15_hirobot.md) | 想知道开放指令怎样变成可执行子任务 |
| [RTC / Training-time RTC](16_rtc.md) | 想解决推理延迟与动作块连续执行 |
| [Human-to-Robot](17_human_transfer.md) | 想理解人类数据怎样迁移到机器人 |
| [RL Token](18_rlt.md) | 想理解冻结VLA后的小模块在线学习 |

共18篇；数字按各篇材料的测试范围解读，未公开的实验不补写。

[机制学习](../MECHANISMS.md) · [八模型对照](../MODEL_COMPARISON_8.md) · [完整目录](../CONTENTS.md) · [首页](../../README.md)
