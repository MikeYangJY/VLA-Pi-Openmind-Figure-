# Figure 数据主线：Go-Big → Index → Helix 2.5

<!-- reading-nav-start -->
[首页](../../README.md) · [Figure入口](README.md) · [按问题查找](../FIND_BY_QUESTION.md)
<!-- reading-nav-end -->

核对日期：2026-09-18。**Helix 2.5 于2026-09-17发布**，以下来自官方技术页面，尚不能当作独立复现结论。当前发现的材料是网页，不伪造 PDF 论文。

## Go-Big（2025-09-18）

[官方说明](https://www.figure.ai/news/project-go-big)提出扩展第一人称人类数据，并展示人类视频向语言引导导航迁移。这里的“只用人类视频”应限制在其报告的导航学习设置，不能扩大成整个机器人控制栈从未使用机器人数据或预训练。

**本库阅读问题：** 哪部分监督来自视觉，哪部分来自位姿/运动或已有控制能力？必须区分“学习去哪里”和“学会如何稳定走过去”。

## Index（2026-08-25）

[官方说明](https://www.figure.ai/news/introducing-index)将数据采集、质量控制与规模扩展作为预训练基础。它是 Figure 的专属数据管线；网页中的规模主张不代表外部研究者可直接下载的开源数据集。

**本库阅读问题：** 数据量增加时，场景、行为、相机和标注质量分别如何变化？仅报告视频数量不足以判断有效监督量。

## Helix 2.5（2026-09-17）

[官方技术页](https://www.figure.ai/news/helix-2-5-zero-shot-30-home-generalization)报告的训练/评估路径：**从随机初始化在 Index 预训练 → 分别适配整理客厅、叠毛巾、铺床 → 在 30 个未见家庭测试**。每项任务使用一个固定 checkpoint；不能改写成同一个最终 checkpoint 无差别完成全部任务。

官方对照保持下游数据与训练/评估设置相同，报告有 Index 预训练的成功率为 56%，从头做任务训练为 9%。这里成功要求整项任务完成，不给部分进展分；zero-shot 特指测试家庭与物体，任务已在别处采集数据并适配。与原始 Helix 不同，该页明确本轮预训练从随机初始化开始；不能套用“从 7B VLM 初始化”的旧版描述。[来源](https://www.figure.ai/news/helix-2-5-zero-shot-30-home-generalization)

## 本库分析：应怎样和 π0.5 / π0.7 对读

比较价值在于**泛化切分与数据来源**：哪些测试条件被留出、哪些行为仍在训练分布内、任务适配成本如何计算。不同机器人、任务长度和成功定义下的百分比不适合做公司排名。

需要继续验证的事项包括完整试验数与不确定性、checkpoint 选择流程、数据重叠检查、失败分布，以及详细网络/控制接口。即使官方页面给出某些方法说明，也不代表外部能审计原始数据或独立重复全部流程。

## 后续官方资料

- [Scaling Helix：物流任务数据扩展](https://www.figure.ai/news/scaling-helix-logistics)：2025-06-07，作为数据规模与吞吐评估案例。
- [Bedroom Tidy](https://www.figure.ai/news/helix-02-bedroom-tidy)：2026-05-08，作为多机器人互动案例。

以上是拓展阅读入口，本库未将展示视频自行转写为统计性能结论。

<!-- reading-footer-start -->
[前一篇：Helix 02](02_helix02.md) · [接着读：DeepMind](../deepmind/README.md) · [返回Figure入口](README.md)
<!-- reading-footer-end -->
