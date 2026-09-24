# DeepMind 来源索引：要核对什么，就去哪里

[学习入口](README.md) · 核对日期：**2026-09-24**。

**来源层级：** 论文／技术报告看方法与实验；官方产品页和模型卡看当前能力与访问状态；官方代码看实际接口；访谈看观点。本文学习例子与提问建议不冒充官方实验。

## 前置技术路线

| 资料 | 直达原文 | 优先看什么 |
|---|---|---|
| SayCan | [项目与论文入口](https://say-can.github.io/) | Language grounding、affordance 与技能选择 |
| RT-1 | [项目](https://robotics-transformer1.github.io/) · [论文](https://arxiv.org/abs/2212.06817) | 模型概览、数据量、已见／未见任务实验 |
| PaLM-E | [项目](https://palm-e.github.io/) · [论文](https://arxiv.org/abs/2303.03378) | 连续感知向量如何进入语言模型；文本与低层策略的分工 |
| RT-2 | [项目与论文入口](https://robotics-transformer2.github.io/) | 动作 token 与视觉语言共同微调 |
| Open X-Embodiment／RT-X | [项目、数据及论文入口](https://robotics-transformer-x.github.io/) | 区分数据集合和模型；跨机器人实验 |

## Gemini Robotics 原始报告

| 资料 | 原文 | 对应学习页／定位 |
|---|---|---|
| Gemini Robotics，2025 | [论文](https://arxiv.org/abs/2503.20020) · [用户提供的 PDF](../papers/deepmind/gemini_robotics.pdf) | [初代笔记](01_gemini_robotics.md)；§2.3 Tables 5–6 / Fig.13：ICL；§3.1 / Fig.14：架构、数据与延迟 |
| Gemini Robotics 1.5，2025 | [论文](https://arxiv.org/abs/2510.03342) · [用户提供的 PDF](../papers/deepmind/gemini_robotics_15.pdf) | [1.5 笔记](02_gemini_robotics_15.md)；§2：分工与训练；§3.2–3.3：MT 与 Thinking；§5 / Table 1 / App.D：长任务与失败 |
| 1.5 官方发布 | [文章](https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/) | 先看概览，涉及实验数字再回论文 |

## 更新、模型卡与开发入口

| 资料 | 链接 | 用来核对 |
|---|---|---|
| 模型家族总入口 | [Gemini Robotics](https://deepmind.google/models/gemini-robotics/) | 三类模型与合作入口 |
| 初代 On-Device | [2025-06-24 发布](https://deepmind.google/blog/gemini-robotics-on-device-brings-ai-to-local-robotic-devices/) | 本地部署分支 |
| ER 1.6 | [官方文章](https://deepmind.google/blog/gemini-robotics-er-1-6/) | 具身推理与视觉能力更新 |
| GR 2 系列 | [2026-07-30 发布](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/) | 整体分工和能力范围 |
| GR 2 VLA | [模型页](https://deepmind.google/models/gemini-robotics/vla/) | 分任务评测与 private preview |
| ER 2 | [模型页](https://deepmind.google/models/gemini-robotics/embodied-reasoning/) · [模型卡](https://deepmind.google/models/model-cards/gemini-robotics-er-2/) | 推理、数据披露和 API 分发；基础模型版本以模型卡为准 |
| On-Device 2 | [模型页](https://deepmind.google/models/gemini-robotics/on-device/) | 少示例适配、本地运行、trusted testers |
| ER 开发文档 | [Robotics overview](https://ai.google.dev/gemini-api/docs/robotics-overview) | API 型号、输入输出、可用功能、接口限制 |
| ER 官方示例库 | [robotics-samples](https://github.com/google-gemini/robotics-samples) | 示例组织与后续更新 |
| 用户指定的 ER notebook | [最新](https://github.com/google-gemini/robotics-samples/blob/main/Getting%20Started/gemini_robotics_er.ipynb) · [固定版本](https://github.com/google-gemini/robotics-samples/blob/c51cbab6e6efffff8738ecf9ce41ba85034d9654/Getting%20Started/gemini_robotics_er.ipynb) | [逐段中文导读](04_er_notebook_walkthrough.md)，不是 VLA 训练源码 |
| Genie 3 | [官方文章](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/) | 世界模型能力；不能据此推定与 Robotics 已怎样联训 |

## 访谈入口

- [Release Notes：Introducing Gemini Robotics 2，节目与文字稿](https://google-ai-release-notes.podigee.io/34-introducing-gemini-robotics-2)，2026-08-02；[观点与时间点](09_public_interviews.md)。
- [谭捷访谈，节目 121](https://www.youtube.com/watch?v=2o281Zy5aZE)：回听入口，本轮不新增未经逐字稿核实的精确引语。

## 维护口径

- **固定代码版本：** `c51cbab6e6efffff8738ecf9ce41ba85034d9654`。导读按该版 83 个 cell 核对；上游 `main` 会变化。
- **网页会变化：** 本页记录核对日期；接口状态按具体端点核对，普通与 streaming 端点的功能可能不同。
- **数字要带上下文：** 任务、机器人、是否微调、样本数、指标和实机／仿真条件缺一项，都可能改变解释。
- **未做的验证：** 本轮核查文献、模型页和示例源码；未独立复现机器人实验、训练闭源模型或验证个人账号的 API 权限。

[全库来源目录](../SOURCES.md) · [返回 DeepMind](README.md)
