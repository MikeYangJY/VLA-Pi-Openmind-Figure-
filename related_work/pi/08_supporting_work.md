# 把 PI 主线补完整的五项工作

<!-- reading-nav-start -->
[首页](../../README.md) · [PI入口](README.md) · [按问题查找](../FIND_BY_QUESTION.md)
<!-- reading-nav-end -->

这些是机制连接，不是额外的版本号。每项都沿用问题→训练→运行→证据→研究问题的 workflow。

## Hi Robot：开放指令与执行中的语言纠正

[论文](../papers/pi/hirobot.pdf) · [arXiv](https://arxiv.org/abs/2502.19417) · [官方说明](https://www.pi.website/research/hirobot)。精读 Section 4 与分层/合成数据消融。

**问题：** 简单动作策略难以直接处理隐含约束、指代和用户中途修改意图。

**训练：** 在机器人轨迹与子任务标签基础上构造高层图像—语言训练例，包含人工标注和合成复杂指令/交互；训练高层策略，下层使用执行具体语言子任务的 VLA。

**运行：** 当前场景 + 总体任务 + 用户补充 → 高层推理与下一子任务 → 低层 π0 动作 → 新观测。它体现的是带闭环反馈的分层推理，而非一次性生成全部脚本。

**证据：** 比较 flat VLA、通用 VLM 高层及训练过的高层策略。重点看复杂指令跟随和任务进展，不把“会输出解释”直接当作动作成功。

**本库问题：** 收益来自更好的子任务分解，还是训练过的语境理解？固定下层 policy，替换高层，是最直接的检验。

## RTC：模型思考时机器人仍在执行

[论文](../papers/pi/rtc.pdf) · [arXiv](https://arxiv.org/abs/2506.07339) · [官方说明](https://www.pi.website/research/real_time_chunking)。重点读动作重叠/前缀约束和延迟实验。

**问题：** 同步策略在动作块之间停顿；朴素异步切换又可能产生不连续动作。

**方法：** 将新动作块预测视为带前缀约束的 inpainting。推理期间已经承诺执行的旧动作不能更改；剩余重叠部分以约束维持轨迹一致，同时允许后半段响应新观测。

**运行：** 执行旧块 → 同时预测下一块 → 按推理延迟对齐时间索引 → 切换到新块尚未执行的部分。原始 inference-time RTC 可作用于 flow/diffusion policy 而不改训练。

**证据与限制：** 看成功率与完成时间随延迟的变化；不要把 temporal ensembling、线性平滑与 RTC 写成同义词。

**本库问题：** 延迟估计不准时冻结了错误长度的前缀，会怎样？应加入时间戳抖动和延迟突增测试。

## Training-time RTC：训练时就让模型看见动作前缀

[论文](../papers/pi/rtc_training.pdf) · [arXiv](https://arxiv.org/abs/2512.05964)。

**训练：** 模拟推理延迟，并把已经确定的动作前缀作为条件；让模型学习补全后续动作。

**运行：** 直接条件化生成，减少原始推理时 inpainting 的额外计算。它改变训练配方，不能仍叫“完全不需要训练修改”。π0.7 正文明确使用这一后续版本。

**评估重点：** 用相同动作模型、数据与延迟设置比较两种 RTC；既看较大延迟的效果，也看推理开销。

## Human-to-robot：人类视频何时能帮助机器人

[论文](https://www.pi.website/download/human_to_robot.pdf) · [官方项目页](https://www.pi.website/research/human_to_robot)。

**训练 workflow：** 第一人称人类视频 → 3D 手部轨迹和密集子任务语言标签 → 将人视为额外 embodiment → 与相关机器人数据共同微调预训练 VLA。

原文的重点是：随着机器人预训练的场景、任务和本体多样性提高，这种简单混合配方更能利用人类数据。它没有证明任何互联网上的视频都可直接替代机器人数据。

**证据：** 固定下游设置，改变机器人预训练多样性并比较加/不加人类数据；考察只在人类示范中出现的目标场景。不要把手部轨迹标签、数据筛选与已有机器人基础能力省略掉。

**本库问题：** 几何轨迹与语义标签各贡献多少？加入仅视频、仅语言、视频+手轨迹对照能更清楚定位迁移通道。

## RL Token：用小模块做快速在线改进

[论文](https://www.pi.website/download/rlt.pdf) · [官方说明](https://www.pi.website/research/rlt)。正式标题为 **RL Token: Bootstrapping Online RL with Vision-Language-Action Models**。

**训练 workflow：** 先适配 VLA 并训练压缩表征的 encoder/decoder；之后冻结 VLA 与表征模块，在 RL token 上训练轻量 actor-critic。actor 同时接收参考 VLA action chunk，并通过正则保持与先验行为的联系。

**运行：** VLA 提供状态表征与参考动作 → 小 actor 输出动作块 → 环境交互进入 replay buffer → 更新 actor-critic。不能把它写成“给基础动作直接加一个手调比例的 residual”；原文专门区分了这一点。

**证据：** 原文四项精细操作实验针对任务最难阶段改善速度与成功率。“最多 3×”指特定阶段/设置的提速，不是整个家务任务或所有模型的平均提升。

**本库问题：** 压缩 token 丢失什么接触信息？参考动作条件会不会让 actor 只复制 VLA？这比笼统问“RL 是否有用”更容易形成实验。

<!-- reading-footer-start -->
[接着读：机制与数据](../MECHANISMS.md) · [返回PI入口](README.md)
<!-- reading-footer-end -->
