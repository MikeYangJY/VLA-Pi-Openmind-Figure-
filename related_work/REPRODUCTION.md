# 从阅读到最小复现

<!-- reading-nav-start -->
[首页](../README.md) · [选读入口](RESEARCH_PRACTICE.md) · [按问题查找](FIND_BY_QUESTION.md)
<!-- reading-nav-end -->

这是一份下一阶段的操作路线。本次没有下载模型权重、训练模型、调用付费 API 或控制机器人。

## 先划清可获取的对象

| 对象 | 本次核对到的公开入口 | 能合理开展的工作 |
|---|---|---|
| π0 / π0-FAST / π0.5 | [openpi](https://github.com/Physical-Intelligence/openpi) 列出代码和 checkpoint | 跑通公开推理/微调流程，研究数据与动作接口 |
| FAST tokenizer | [官方 Hugging Face](https://huggingface.co/physical-intelligence/fast) | tokenization/重建分析 |
| π*0.6 / MEM / π0.7 | 论文与官方说明；本次 openpi README 未列这些 checkpoint | 方法分析、在可用基线做受启发的机制实验 |
| Gemini Robotics-ER | 官方开发者入口 | 在实际账号可用时研究具身推理/编排 |
| Gemini Robotics 1.5 VLA | 官方模型页列受限伙伴访问 | 报告分析；不能声称已能公开下载并复现 |
| Figure Helix 家族 | 官方技术页和演示 | 系统分工/数据/评估研究；非完整训练复现 |

openpi 核对版本：[`215abfb217dbac7d5f1273282331b9b1866c0479`](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479)。开放状态会改变，后续实验先重新核对。

## 先完成代码阅读，再运行

阅读顺序统一见[openpi学习入口](../openpi_study/README.md)：跟推理 → 理解训练 → 比较π0.5 → 补工程。这里接着讨论运行时的验收，不另设一套阅读顺序。

先核对数据映射、归一化统计、模型配置与平台动作单位，再按固定版本文档准备环境。官方README指向Linux/NVIDIA环境；JAX与PyTorch功能支持也有差别。

## 第一个可验收的里程碑

**输入一条合法观测，得到正确形状和单位的动作块，并解释每个字段的含义。** 这还不是任务成功，但能排除大量接口错误。

之后才跑完整模拟 rollout。至少记录 checkpoint/代码提交、任务、种子、成功定义、终止原因、推理 p50/p95 延迟、控制频率、执行前缀、模型设备和数据版本。

## 最小评估表

| 实验 | 固定项 | 唯一主要变量 | 指标 |
|---|---|---|---|
| 基线正确性 | 模型、数据、任务集、环境版本 | 无，先重复确认 | 完整任务成功率、失败视频、接口检查 |
| 延迟 | 同一 checkpoint 与种子 | 注入延迟/调度机制 | 成功率、耗时、动作块边界不连续性 |
| 记忆 | 相同训练预算与场景 | 历史/文本状态的可见性 | 有歧义场景准确率、长任务成功率 |
| 语义跟随 | 同一场景与对象 | 指令限制条件 | 选对对象、约束违反、任务完成 |

只报告离线动作误差不足以证明闭环任务能力。模型损失下降、成功率上升、速度提高，可能对应不同原因。

<!-- reading-footer-start -->
[返回选读入口](RESEARCH_PRACTICE.md)
<!-- reading-footer-end -->
