# Gemini Robotics 2：全身动作、ER 协调与本地部署

[学习入口](README.md) · [前一篇：1.5](02_gemini_robotics_15.md) · [下一篇：ER 代码](04_er_notebook_walkthrough.md)

核对日期：**2026-09-24**。本页主要依据官方发布、模型页和模型卡；不要把产品披露当成公开了完整训练算法的论文。

## 1. 这一代想解决什么？

从桌面双臂操作进一步走向 **全身移动与操作、更复杂的手部动作、多机器人协调及易部署的模型**。阅读时仍要拆成三个对象。[2026-07-30 官方发布](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/)

| 模型 | 白话职责 | 主要变化 | 官方列出的访问定位 |
|---|---|---|---|
| [Gemini Robotics 2](https://deepmind.google/models/gemini-robotics/vla/) | 根据观察与指令产生动作 | 全身控制、更复杂灵巧操作 | Private preview |
| [Gemini Robotics ER 2](https://deepmind.google/models/gemini-robotics/embodied-reasoning/) | 理解现场、规划、工具调用与协调 | 时序／进度理解、多机器人任务组织 | AI Studio／Gemini API public preview；企业平台另有 private preview |
| [Gemini Robotics On-Device 2](https://deepmind.google/models/gemini-robotics/on-device/) | 本地运行的动作 VLA | 减少网络依赖，快速适配新身体 | Trusted testers |

**它们不是必须按 ER 2 → GR 2 → On-Device 2 串起来的三个步骤。** 两个动作模型有不同部署定位；ER 可以组织调用动作模型或其他已有工具。能调用 ER API，不等于拿到了完整 VLA 权重。

## 2. 方法与数据：知道到哪里？

- **动作侧：** 官方展示全身动作与操作配合，但没有完整开放所有控制层级、动作表示、损失函数和数据混合。
- **ER 侧：** 模型卡明确基于 **Gemini 3.5 Flash**，加具身推理任务数据；不要根据评测对照模型名称反推它的基座版本。[ER 2 模型卡](https://deepmind.google/models/model-cards/gemini-robotics-er-2/)
- **本地适配：** On-Device 2 官方称可用少于 200 个例子、几小时训练适配新身体。这是已有基座后的适配，**不是从零训练数据总量，也不是“不更新参数的 ICL”**。[官方模型页](https://deepmind.google/models/gemini-robotics/on-device/)

用什么数据分别教语义、动作和结果判断，见[训练专题](05_training_and_data.md)。

## 3. 公开评测告诉了什么？

下面只摘取帮助理解口径的例子，不能汇总成一个“公司总成功率”。

| 对象／平台 | 官方页面指标例子 | 怎样解释 |
|---|---|---|
| GR 2；Apollo + Inspire hands | 桌上拿取 68.4%，地面拿取 45.7%，架上拿取 76.3% | 不同任务难度仍有差异；这是指定平台和测试项的 accuracy，不能代表全天候家务可靠性 |
| ER 2；ERQA | 78.5% | 具身视觉问答准确率，不是机器人抓取成功率 |
| ER 2；success detection，image | 87.7% | 对图像中的任务成功状态做判断的准确率，不是动作完成率 |

来源：[GR 2 模型页评测图](https://deepmind.google/models/gemini-robotics/vla/) · [ER 2 模型页评测图](https://deepmind.google/models/gemini-robotics/embodied-reasoning/)。独立复现实验、长期连续运行和不同客户工位的统计不能从这些数字自动推出。

具体工业／家庭任务仍从[场景清单](../SCENARIOS_BY_MODEL.md)查；本页不把演示升级成客户生产部署证据。

## 4. 两个旁支，放回正确位置

| 更新 | 为什么要单列 | 原文 |
|---|---|---|
| On-Device，2025-06-24 | 在 2 之前就已有本地动作模型方向；不是 ER 的别名 | [官方发布](https://deepmind.google/blog/gemini-robotics-on-device-brings-ai-to-local-robotic-devices/) |
| ER 1.6，2026-04-14 | 高层具身推理更新，包含空间／多视角、成功检测与仪表理解；不代表同名动作 VLA 已发布 | [官方发布](https://deepmind.google/blog/gemini-robotics-er-1-6/) |

版本号相近，不等于功能层级相同。你给的 notebook 当前已经使用 **ER 2**，详细解释在[代码导读](04_er_notebook_walkthrough.md)。

## 5. 合作与商业落地：什么已知，什么还需要问？

官方家族页面提供模型接入、合作伙伴与开发者生态入口。对新公司而言，可以分别研究 **ER API 接入、动作模型合作／测试、硬件适配、部署与评测支持**。[官方入口](https://deepmind.google/models/gemini-robotics/)

页面列出 Agile Robots、Apptronik、Boston Dynamics 等研究合作伙伴、100 多家 trusted testers，以及支持早期 physical AI 公司的 Google DeepMind Accelerator。这说明已有多种合作入口；**tester 数不是付费量产客户数**，也不能据此认定各家获得相同模型和交付支持。[官方合作与生态栏目](https://deepmind.google/models/gemini-robotics/)

尚不能仅凭这些页面回答：谁承担每个客户的数据采集成本、定制与通用交付各占多少、收费和验收方式、失败由谁兜底。访谈应请对方用一个可分享的案例走完“接入 → 适配 → 验收 → 运行反馈”，见[客户问题映射](07_expert_conversation.md)。

API 功能还应核对具体端点：普通 ER 2 与 streaming 端点的支持项并不完全相同，不要把模型家族页面的能力都抄成某个端点可用。[当前开发文档](https://ai.google.dev/gemini-api/docs/robotics-overview)

[下一篇：实际代码怎样调用 ER](04_er_notebook_walkthrough.md) · [原文索引](08_sources.md)
