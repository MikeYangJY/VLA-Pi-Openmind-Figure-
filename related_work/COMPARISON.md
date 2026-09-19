# 三条路线：在同一 workflow 下比较

<!-- reading-nav-start -->
[首页](../README.md) · [横向对比入口](README.md) · [按问题查找](FIND_BY_QUESTION.md)
<!-- reading-nav-end -->

核对日期：2026-09-18。下表是对各篇原文的结构化归纳；它不是统一 benchmark 上的排名。

新增：[八模型核心对照表](MODEL_COMPARISON_8.md)，按π0基准架构、数据类型/规模、泛化成功率、速度/时长、问题与不足详细比较。此页保留workflow视角。

## 训练与执行的关键差异

| 工作 | 数据/学习重点 | 高层与低层接口 | 动作与执行 | 最应检查的证据 |
|---|---|---|---|---|
| [π0](pi/01_pi0.md) | 多本体示范、VLM 初始化、后训练 | 观测和语言条件 | 连续 flow action chunk | 预训练对后训练的帮助、灵巧操作表现 |
| [FAST](pi/02_fast.md) | 频域压缩后的动作 token | 离散序列 | AR token 解码后恢复动作块 | 同一数据和算力下收敛速度、解码成本 |
| [π0.5](pi/03_pi05.md) | 多源 co-training、分阶段训练 | 自然语言子任务；原文高低层分布由同一模型表示 | flow expert | 未见家庭、数据来源消融 |
| [KI](pi/04_knowledge_insulation.md) | CE + flow，隔离 expert → backbone 梯度 | backbone 表征供 action expert 读取 | 连续动作推理 | 去掉 stop-gradient、去掉 VLM 数据的对照 |
| [π*0.6](pi/05_recap.md) | 价值估计、advantage conditioning、自主经验和纠错 | 子任务与优势条件 | flow；部署收集数据后再更新 | 吞吐、成功率、纠错与自主数据贡献 |
| [MEM](pi/06_mem.md) | 训练视觉历史编码和文本记忆更新 | 长期文本状态 → 子任务；短期视觉 → 动作 | 有历史的 VLA | 无记忆/只保留一种记忆的消融 |
| [π0.7](pi/07_pi07.md) | 多模态上下文、质量标签、经验蒸馏 | 子任务 + 视觉子目标 + metadata | flow + training-time RTC | 不同 prompting 配置、跨本体和组合泛化 |
| [Gemini 1.5](deepmind/02_gemini_robotics_15.md) | 多本体训练、Motion Transfer、embodied thinking | ER 编排器向 VLA 发自然语言指令；VLA 内也可思考 | VLA 输出机器人动作；具体实现部分未披露 | MT、thinking、编排器替换的消融 |
| [Helix](figure/01_helix.md) | 端到端连续动作回归 | S2 的连续 latent → S1 | S1 200 Hz；S2 7–9 Hz（官方） | 新物体泛化与端侧异步控制 |
| [Helix 02](figure/02_helix02.md) | 模仿与全身控制的组合 | S2 latent → S1 关节目标 → S0 | S1 200 Hz，S0 1 kHz（官方） | 全身移动操作、触觉、长任务完整评估 |
| [Helix 2.5](figure/03_data_and_helix25.md) | Index 人类数据预训练后行为适配 | 详细模块接口尚未完整披露 | 全身任务；不能直接沿用旧版全部参数 | 新家庭/物体 zero-shot 与预训练消融 |

## 同一任务的概念映射

设任务是“把桌面整理好，但保留正在使用的杯子”。以下是本库用于理解接口的例子，不是已经做过的跨模型测试。

| 环节 | PI 系列的对应机制 | Gemini 1.5 的对应机制 | Helix 系列的对应机制 |
|---|---|---|---|
| 理解约束 | 高层语言 policy / coaching | ER 编排器推理与工具使用 | S2 将场景、指令转成 latent |
| 决定下一步 | 子任务文本；π0.7 可加子目标图 | ER 下发短期指令；VLA 可继续分解 | S2 latent 条件影响 S1 |
| 记住已完成事项 | MEM 的长期文本记忆 | agent 的任务上下文与记忆任务能力 | 公开行为能体现持续任务状态，但不据此断言采用 MEM |
| 执行抓取 | flow action expert | VLA 动作模型 | 快速 S1；全身动作由 S0 进一步执行 |
| 延迟处理 | RTC 约束动作块重叠前缀 | 具体执行栈按报告披露范围记录 | 不同时间尺度系统异步运行 |
| 从失误中学习 | RECAP 或 RLT 的显式训练机制 | 在线参数更新机制不能从纠错演示推定 | 不能仅凭“self-correction”断言在线 RL |

## 五个需要纠正的直觉

**分层结构不代表相同算法。** π0.5 的共享模型分层推理、Gemini 的 ER/VLA 编排、Helix 的 latent 双系统，接口和参数共享方式不同。

**动作 token 的作用不只在部署。** FAST 可以提供训练时的动作监督；部署时可改由连续 expert 生成动作。要分别问训练监督形式和推理输出形式。

**KI 与 Helix 的梯度方向不同。** KI 阻止动作 expert 的损失更新 backbone；原始 Helix 官方说明 S1 梯度经 latent 传入 S2。不能因两者都有“大模型 + 小控制模型”便写成同一种知识隔离。

**记忆与 RL 改变不同的东西。** MEM 在执行中更新上下文；RECAP 通过收集经验和训练更新模型；RLT 将适配集中在较小 actor-critic；它们可以组合，但互不等价。

**世界模型不是整个控制栈。** π0.7 的视觉子目标为动作生成提供条件。不能由此推定机器人在完整物理模拟中搜索计划，也不能把生成图像当作未来必然发生的状态。

## 对你的研究起点的判断

本库建议以 **openpi 的数据接口和模型运行流程** 建立动手基础，以 **Gemini 的编排器消融** 学习长任务评估，以 **Helix 的系统频率划分** 检查部署假设。理由是它们分别暴露了可操作的不同层面；没有理由在硬件、预算和目标任务未知时直接选出唯一“最佳模型”。

来源由上表各阅读卡片链接到原文。开放实现边界另见 [复现指南](REPRODUCTION.md)。

<!-- reading-footer-start -->
[接着读：Figure](figure/README.md) · [返回横向对比入口](README.md)
<!-- reading-footer-end -->
