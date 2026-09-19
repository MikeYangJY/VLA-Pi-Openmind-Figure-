# VLA：Physical Intelligence · Google DeepMind · Figure

面向机器人 VLA 研究入门与进一步精读的中文资料库。按 **数据 → 训练 → 推理 → 控制 → 评估 → 经验回流** 组织，而不是只按版本号罗列论文。

**资料核对日期：2026-09-19。** 仓库名称沿用 Openmind，但本库的 Gemini Robotics 指 **Google DeepMind**；不将其与其他名为 OpenMind 的项目混同。

## 从这里开始

- **八模型横向对照：** [架构、训练数据、数据量、泛化、速度、时长、核心问题与不足](related_work/MODEL_COMPARISON_8.md) · [完整单表CSV](related_work/data/model_comparison_8.csv)。每项附来源，缺失信息标注未披露。
- **三项数据质量概念：** [运动学检查、多摄像头成功验证、视觉归一化](related_work/pi/11_data_quality_three_concepts.md)。从具体例子到openpi代码，并列出访谈追问。

- **2026-09-19 学习专题：** [π0.6 / π0.6* / MEM 的区别](related_work/pi/09_pi06_family.md) → [从零理解 action expert 与 flow matching](openpi_study/07_FLOW_MATCHING_FROM_ZERO.md) → [与 π0.7 世界模型的训练、数据对照](related_work/pi/10_pi07_world_model_vs_action_expert.md)。含逐步公式、手算例子、同一轨迹拆成不同训练样本的表格。
- [18 篇简洁调研笔记](related_work/quick_notes/README.md)：每篇按「核心大纲 → 问题 → 方法 → 实验」阅读，先建立框架。
- [openpi 中文代码学习：π0 与 π0.5](openpi_study/README.md)：代码树、初学者阅读顺序、逐文件说明和中文注释源码。
- [Related work 总入口](related_work/README.md)：文献地图、重点工作与阅读顺序。
- [统一调研 workflow](related_work/WORKFLOW.md)：以后增加论文时沿用的分析方法。
- [三条路线的流程比较](related_work/COMPARISON.md)：模型边界、接口、训练方法、评估口径。
- [针对当前基础的学习路线](related_work/LEARNING_PATH.md)：已读 π0、π0.5、π0.7 后，接下来怎么学。
- [文献与官方资料目录](related_work/SOURCES.md)：PDF、官方网站、原始来源和校验信息。

本库区分论文报告、官方技术说明和本库分析。没有公开的训练细节明确标为“未披露”；公司演示不等同于独立复现。原始论文与本库笔记分开放置，版权及许可归原作者或原发布方。
