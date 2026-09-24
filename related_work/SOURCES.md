# 论文与官方资料目录

**DeepMind 补充来源（2026-09-24）：** [按学习顺序整理的论文、模型卡、API、固定版本 ER notebook 和公开访谈](deepmind/08_sources.md)。新增早期技术路线与 ER 代码导读均从该索引回查原文。

<!-- reading-nav-start -->
[首页](../README.md) · [资料入口](README.md) · [按问题查找](FIND_BY_QUESTION.md)
<!-- reading-nav-end -->

原资料核对日期：2026-09-18；原专题补充日期：2026-09-20。下方原登记表包含 **20 篇论文（15 篇附 PDF：用户提供 9 篇、补充归档 6 篇；另外 5 篇提供原始链接）**，以及 **21 项官方网页、模型卡、代码或数据入口**。2026-09-24 新增的 DeepMind 链接单列于上方专题索引，不并入这份原表的计数。

用户提供 PDF 保留原字节内容。上游 arXiv 可能更新版本；本库的准确存档由 [sources.json](sources.json) 中的 SHA-256、字节数与页数确定。文件名仅用于检索，不能推断它必然是上游最新修订。

## 专题证据入口

- [Figure 每代变化](figure/04_evolution_plain_language.md) · [Index 管线正文与官方图示](figure/05_index_pipeline.md)：逐项区分官方披露、教学解释与未知细节。
- [工业 / 家庭任务清单](SCENARIOS_BY_MODEL.md)：每行链接官方发布或本地论文页码，硬件型号与模型版本分别归属。
- [八模型对照表的原文定位](MODEL_COMPARISON_8.md#4-数字与原文定位)：各数字对应论文章节/图号或官方技术页。
- [数据质量专题](pi/11_data_quality_three_concepts.md)：另在正文直接链接ROS状态/约束接口、相机标定和固定openpi代码，作为概念解释参考；不将其列为新增VLA论文。

## 论文归档

| ID | 正式标题 | 年份 | 获取方式 | PDF / 页数 | 笔记 |
|---|---|---:|---|---|---|
| flow_matching | [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) | 2022 | 官方链接 | [原始 PDF](https://arxiv.org/pdf/2210.02747) / 链接收录 | [零基础教程](../openpi_study/07_FLOW_MATCHING_FROM_ZERO.md) |
| pi0 | [π0: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164) | 2024 | 用户提供 | [PDF](papers/pi/pi0.pdf) / 17 | [阅读卡片](pi/01_pi0.md) |
| pi05 | [π0.5: a Vision-Language-Action Model with Open-World Generalization](https://arxiv.org/abs/2504.16054) | 2025 | 用户提供 | [PDF](papers/pi/pi05.pdf) / 19 | [阅读卡片](pi/03_pi05.md) |
| pi07 | [π0.7: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities](https://arxiv.org/abs/2604.15483) | 2026 | 用户提供 | [PDF](papers/pi/pi07.pdf) / 25 | [阅读卡片](pi/07_pi07.md) |
| fast | [FAST: Efficient Action Tokenization for Vision-Language-Action Models](https://arxiv.org/abs/2501.09747) | 2025 | 用户提供 | [PDF](papers/pi/fast.pdf) / 19 | [阅读卡片](pi/02_fast.md) |
| knowledge_insulation | [Knowledge Insulating Vision-Language-Action Models: Train Fast, Run Fast, Generalize Better](https://arxiv.org/abs/2505.23705) | 2025 | 用户提供 | [PDF](papers/pi/knowledge_insulation.pdf) / 18 | [阅读卡片](pi/04_knowledge_insulation.md) |
| pi06_star | [π*0.6: a VLA That Learns From Experience](https://arxiv.org/abs/2511.14759) | 2025 | 用户提供 | [PDF](papers/pi/pi06_star.pdf) / 18 | [阅读卡片](pi/05_recap.md) |
| mem | [MEM: Multi-Scale Embodied Memory for Vision Language Action Models](https://arxiv.org/abs/2603.03596) | 2026 | 用户提供 | [PDF](papers/pi/mem.pdf) / 15 | [阅读卡片](pi/06_mem.md) |
| gemini_robotics | [Gemini Robotics: Bringing AI into the Physical World](https://arxiv.org/abs/2503.20020) | 2025 | 用户提供 | [PDF](papers/deepmind/gemini_robotics.pdf) / 64 | [阅读卡片](deepmind/01_gemini_robotics.md) |
| gemini_robotics_15 | [Gemini Robotics 1.5: Pushing the Frontier of Generalist Robots with Advanced Embodied Reasoning, Thinking, and Motion Transfer](https://arxiv.org/abs/2510.03342) | 2025 | 用户提供 | [PDF](papers/deepmind/gemini_robotics_15.pdf) / 62 | [阅读卡片](deepmind/02_gemini_robotics_15.md) |
| hirobot | [Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models](https://proceedings.mlr.press/v267/shi25d.html) | 2025 | 本次补充 | [PDF](papers/pi/hirobot.pdf) / 15 | [阅读卡片](pi/08_supporting_work.md) |
| rtc | [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/pdf/2506.07339) | 2025 | 本次补充 | [PDF](papers/pi/rtc.pdf) / 25 | [阅读卡片](pi/08_supporting_work.md) |
| rtc_training | [Training-Time Action Conditioning for Efficient Real-Time Chunking](https://arxiv.org/pdf/2512.05964) | 2025 | 本次补充 | [PDF](papers/pi/rtc_training.pdf) / 6 | [阅读卡片](pi/08_supporting_work.md) |
| human_to_robot | [Emergence of Human to Robot Transfer in Vision-Language-Action Models](https://www.pi.website/download/human_to_robot.pdf) | 2025 | 本次补充 | [原始 PDF](https://www.pi.website/download/human_to_robot.pdf) / 链接收录 | [阅读卡片](pi/08_supporting_work.md) |
| rl_tokens | [RL Token: Bootstrapping Online RL with Vision-Language-Action Models](https://www.pi.website/download/rlt.pdf) | 2026 | 本次补充 | [原始 PDF](https://www.pi.website/download/rlt.pdf) / 链接收录 | [阅读卡片](pi/08_supporting_work.md) |
| diffusion_policy | [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/pdf/2303.04137) | 2023 | 本次补充 | [原始 PDF](https://arxiv.org/pdf/2303.04137) / 链接收录 | [阅读卡片](foundations/README.md) |
| act | [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/pdf/2304.13705) | 2023 | 本次补充 | [原始 PDF](https://arxiv.org/pdf/2304.13705) / 链接收录 | [阅读卡片](foundations/README.md) |
| openvla | [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/pdf/2406.09246) | 2024 | 本次补充 | [PDF](papers/foundations/openvla.pdf) / 37 | [阅读卡片](foundations/README.md) |
| rt2 | [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://proceedings.mlr.press/v229/zitkovich23a.html) | 2023 | 本次补充 | [PDF](papers/foundations/rt2.pdf) / 19 | [阅读卡片](foundations/README.md) |
| octo | [Octo: An Open-Source Generalist Robot Policy](https://arxiv.org/abs/2405.12213) | 2024 | 本次补充 | [PDF](papers/foundations/octo.pdf) / 17 | [阅读卡片](foundations/README.md) |

## 官方网页与开放资源

Figure 本次检索到的关键技术材料主要为官方网页。本库保存可追溯链接与原创笔记，不将它们冒充论文。DeepMind 的模型访问状态与 ER/VLA 区分见对应笔记。

| 工作 / 资料 | 发布日期 | 原始入口 | 阅读卡片 |
|---|---|---|---|
| π0.6 Model Card | 2025-11-17 | [官方模型卡](https://website.pi-asset.com/pi06star/PI06_model_card.pdf) | [家族区别](pi/09_pi06_family.md) |
| BAGEL 图像 flow 实现（固定 commit） | 2026-09-19 核查 | [官方代码](https://github.com/ByteDance-Seed/Bagel/blob/a2fa77dd8caeefc41e6607ae0ec17408d3f4ee9f/modeling/bagel/bagel.py) | [世界模型对照](pi/10_pi07_world_model_vs_action_expert.md) |
| Gemini Robotics 2 brings whole-body intelligence to robots | 2026-07-30 | [官方来源](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/) | [简洁笔记](quick_notes/10_gemini2.md) |
| Helix: A Vision-Language-Action Model for Generalist Humanoid Control | 2025-02-20 | [官方来源](https://www.figure.ai/news/helix) | [笔记](figure/01_helix.md) |
| Introducing Helix 02: Full-Body Autonomy | 2026-01-27 | [官方来源](https://www.figure.ai/news/helix-02) | [笔记](figure/02_helix02.md) |
| Helix 2.5: Zero-Shot 30-Home Generalization | 2026-09-17 | [官方来源](https://www.figure.ai/news/helix-2-5-zero-shot-30-home-generalization) | [笔记](figure/03_data_and_helix25.md) |
| Project Go-Big: Internet-Scale Humanoid Pretraining and Direct Human-to-Robot Transfer | 2025-09-18 | [官方来源](https://www.figure.ai/news/project-go-big) | [笔记](figure/03_data_and_helix25.md) |
| Introducing Index: Building The World’s Largest and Most Diverse Physical Dataset | 2026-08-25 | [官方来源](https://www.figure.ai/news/introducing-index) | [五步管线与图示核对](figure/05_index_pipeline.md) |
| Index App | 持续更新；2026-09-20 核查 | [官方入口](https://www.figure.ai/index-app) | [数据采集边界](figure/05_index_pipeline.md) |
| Scaling Helix: a New State of the Art in Humanoid Logistics | 2025-06-07 | [官方来源](https://www.figure.ai/news/scaling-helix-logistics) | [迭代说明](figure/04_evolution_plain_language.md) |
| Helix 02 Bedroom Tidy | 2026-05-08 | [官方来源](https://www.figure.ai/news/helix-02-bedroom-tidy) | [笔记](figure/03_data_and_helix25.md) |
| Figure 02 production at BMW | 2025-11-19 | [官方部署报告](https://www.figure.ai/news/production-at-bmw) | [硬件与模型归属](SCENARIOS_BY_MODEL.md#figure-industry) |
| Figure 03 at BMW，Helix 02 | 2026-06-30 | [官方现场演示](https://www.figure.ai/news/f-03-at-bmw) | [工业任务分类](SCENARIOS_BY_MODEL.md#figure-industry) |
| Gemini Robotics On-Device brings AI to local robotic devices | 2025-06-24 | [官方来源](https://deepmind.google/blog/gemini-robotics-on-device-brings-ai-to-local-robotic-devices/) | [笔记](deepmind/03_updates.md) |
| Gemini Robotics-ER 1.6: Powering real-world robotics tasks through enhanced embodied reasoning | 2026-04-14 | [官方来源](https://deepmind.google/blog/gemini-robotics-er-1-6/) | [笔记](deepmind/03_updates.md) |
| Gemini Robotics 1.5 model information | 持续更新 | [官方来源](https://deepmind.google/en/models/gemini-robotics/gemini-robotics/) | [笔记](deepmind/03_updates.md) |
| openpi official repository | 持续更新 | [官方来源](https://github.com/Physical-Intelligence/openpi) | [笔记](REPRODUCTION.md) |
| FAST official tokenizer | 持续更新 | [官方来源](https://huggingface.co/physical-intelligence/fast) | [笔记](pi/02_fast.md) |
| Open X-Embodiment project | 持续更新 | [官方来源](https://robotics-transformer-x.github.io/) | [笔记](foundations/README.md) |
| DROID project | 持续更新 | [官方来源](https://droid-dataset.github.io/) | [笔记](foundations/README.md) |
| LIBERO project | 持续更新 | [官方来源](https://libero-project.github.io/) | [笔记](foundations/README.md) |

## 用户文件名映射

| 原文件名 | 仓库文件 |
|---|---|
| Pi_0.pdf | [papers/pi/pi0.pdf](papers/pi/pi0.pdf) |
| Pi_0.5.pdf | [papers/pi/pi05.pdf](papers/pi/pi05.pdf) |
| Pi_0.7.pdf | [papers/pi/pi07.pdf](papers/pi/pi07.pdf) |
| pi_Fast.pdf | [papers/pi/fast.pdf](papers/pi/fast.pdf) |
| pi_KI.pdf | [papers/pi/knowledge_insulation.pdf](papers/pi/knowledge_insulation.pdf) |
| pi_06_star.pdf | [papers/pi/pi06_star.pdf](papers/pi/pi06_star.pdf) |
| pi0.6_Mem.pdf | [papers/pi/mem.pdf](papers/pi/mem.pdf) |
| Gemini Robotics- Bringing AI into the Physical World.pdf | [papers/deepmind/gemini_robotics.pdf](papers/deepmind/gemini_robotics.pdf) |
| Gemini Robotics 1.5.pdf | [papers/deepmind/gemini_robotics_15.pdf](papers/deepmind/gemini_robotics_15.pdf) |

## 引用与复用

[references.bib](references.bib) 是标题/年份/URL 的最小书目索引，未伪造完整作者名单或会议信息；正式论文写作应从原始发表页面导出完整 BibTeX。PDF、原论文中的图表与原始代码沿用各自的权利和许可，本库没有对第三方材料重新授权。

补充归档的 6 篇 PDF 使用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，作者署名保留在未改动的 PDF 中，许可来源记录在 sources.json。Hi Robot 使用 Shi et al. (2025), ICML, PMLR 267:54919–54933 的[正式发表版本](https://proceedings.mlr.press/v267/shi25d.html)；RT-2 使用 Zitkovich et al. (2023), CoRL, PMLR 229:2165–2183 的[正式发表版本](https://proceedings.mlr.press/v229/zitkovich23a.html)。两者适用 [PMLR 发表许可](https://proceedings.mlr.press/pmlr-license-agreement.pdf)。ACT、Diffusion Policy、Human-to-Robot、RL Token 提供原始下载链接与原创笔记，不在本公开库复制全文。

<!-- reading-footer-start -->
[返回资料入口](README.md)
<!-- reading-footer-end -->
