# 重点工作：简洁调研笔记

每篇按“核心大纲 → 问题 → 方法 → 实验”阅读。数字保留测试范围，未公开的实验不补写。官方发布与技术报告明确区分。核对日期：2026-09-18。

先看Figure：Helix 2.5 → Index → Helix 02 → Helix；再回看PI的KI、MEM、RECAP与π0.7，对照数据、上下文和经验学习。

| 工作 | 核心大纲 |
|---|---|
| [π0](01_pi0.md) | VLM → 连续动作专家 → flow matching → 通用预训练/专项后训练 |
| [FAST / π0-FAST](02_fast.md) | 连续动作块 → 时间压缩 → 离散token → 自回归学习 |
| [π0.5](03_pi05.md) | 异构共训 → 高层子任务 → 连续动作 → 新家庭泛化 |
| [Knowledge Insulation（KI）](04_ki.md) | 离散监督训练主干 + 梯度隔离 + 连续专家快速输出 |
| [π*0.6 / RECAP](05_recap.md) | 示范/自主经验/纠错 → 价值与优势 → 条件策略 → 再部署 |
| [MEM / π0.6-MEM](06_mem.md) | 短期视觉历史 + 长期语言记忆 → 有状态的决策 |
| [π0.7](07_pi07.md) | 更丰富context → 异构经验共训 → 可引导的generalist |
| [Gemini Robotics（原始报告）](08_gemini.md) | Gemini知识 → ER具身能力 → VLA动作 → 专项适配 |
| [Gemini Robotics 1.5](09_gemini15.md) | ER编排 + Thinking VLA + Motion Transfer |
| [Gemini Robotics 2 系列](10_gemini2.md) | ER 2规划协作 + GR 2全身动作 + On-Device 2本地适配 |
| [Helix](11_helix.md) | S2语义latent → S1连续动作 → 上半身控制 |
| [Helix 02](12_helix02.md) | S2目标 → S1全身关节目标 → S0执行器命令 |
| [Helix 2.5](13_helix25.md) | Index预训练 → 三种任务分别适配 → 未见家庭测试 |
| [Index / Project Go-Big](14_index.md) | 人类行为采集 → 过滤/去重/标注 → 预训练 → 机器人迁移 |
| [Hi Robot](15_hirobot.md) | 总体意图/反馈 → 高层子任务 → 低层VLA闭环执行 |
| [RTC / Training-time RTC](16_rtc.md) | 旧动作持续执行 + 新动作前缀约束 + 延迟对齐 |
| [Human-to-Robot Transfer](17_human_transfer.md) | 人类视频/手轨迹/语言 → 联合训练 → 机器人迁移 |
| [RL Token](18_rlt.md) | 冻结VLA表征 + 参考动作 → 小actor-critic在线学习 |

已有详细笔记、PDF与来源索引仍保留在[调研总入口](../README.md)。代码学习见[openpi学习入口](../../openpi_study/README.md)。

进一步理解 PI：看 [π0.6 / π0.6* / MEM 区别](../pi/09_pi06_family.md) → [action expert 与 flow matching](../../openpi_study/07_FLOW_MATCHING_FROM_ZERO.md) → [π0.7 世界模型和数据对照](../pi/10_pi07_world_model_vs_action_expert.md)。
