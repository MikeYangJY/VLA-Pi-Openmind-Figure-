# DeepMind 补充更新：部署与 ER 1.6

核对日期：2026-09-18。这里记录与主线直接相关的官方更新，不作为穷尽所有机器人研究的目录。

## Gemini Robotics On-Device（2025-06-24）

[官方发布](https://deepmind.google/blog/gemini-robotics-on-device-brings-ai-to-local-robotic-devices/) · [Gemini Robotics 报告](https://arxiv.org/abs/2503.20020)。

官方披露的目标是本地运行、降低网络依赖，并通过少量示范适配任务。其发布说明列出 SDK 与 trusted tester 访问方式。它与旗舰 Gemini Robotics、ER 模型属于不同部署/功能定位，不能将名称中的 On-Device 理解为所有 Gemini 机器人能力已开放下载。

**Workflow：** 本地观测与指令 → 本地动作策略 → 执行与反馈；任务适配另外经过数据收集与微调。研究重点是延迟、断网鲁棒性和任务适配效率。设备端模型大小、目标硬件、可获取权重必须以实际访问文档为准，不能按博客演示推定适合任意消费级电脑。

## Gemini Robotics-ER 1.6（2026-04-14）

[官方发布](https://deepmind.google/blog/gemini-robotics-er-1-6/)。

官方将其定位为高层具身推理升级，强调空间理解、多视角、指点/计数、成功检测与仪表读数，并说明可通过 Gemini API 和 AI Studio 访问。它可以调用 VLA 或用户定义的工具；这不代表它本身替代高频动作控制器，也不应把 ER 1.6 写成已发布同名 “Gemini Robotics VLA 1.6”。

**研究连接：** 若你关心 Gemini 1.5 的 agent 架构，最值得追踪的是高层结束检测能否可靠判断“已经完成”，以及工具返回信息如何进入下一步计划。官方特别说明部分仪表读取评估启用 agentic vision，而其他评估设置不同；不同设置数字不能直接横比。

## 访问状态的边界

本次检查的 [Gemini Robotics 1.5 模型页面](https://deepmind.google/en/models/gemini-robotics/gemini-robotics/) 将动作模型列为 private preview / partners。ER 的开发者访问不等于原 VLA 权重、训练数据或完整机器人系统开放。

这些是文档核查结果；本库未申请 tester、调用付费 API 或验证账号实际可用权限。
