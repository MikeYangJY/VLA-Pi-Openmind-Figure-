# Index / Project Go-Big

<!-- reading-nav-start -->
[首页](../../README.md) · [Figure入口](../figure/README.md) · [按问题查找](../FIND_BY_QUESTION.md)
<!-- reading-nav-end -->

**核心大纲：** 人类行为采集 → 过滤/去重/标注 → 预训练 → 机器人迁移。

**来源类型：** 官方数据与技术说明 · [原始来源](https://www.figure.ai/news/introducing-index) · [五步管线详解](../figure/05_index_pipeline.md) · [Go-Big 到 Helix 2.5](../figure/03_data_and_helix25.md)

## 1）要解决的问题

机器人数据昂贵且覆盖狭窄，需要规模更大、环境与行为更多样的数据供给。

## 2）方法

Index通过人类采集网络获得视频，经过质量过滤、欺诈审核、去重、再平衡和层级文本标注。Go-Big先前讨论利用人类视频迁移导航行为；两者应按具体数据和任务范围阅读。

## 3）实验与结论

Index发布介绍数据供给与处理机制，本身不是完整控制实验论文；模型贡献看Helix 2.5的预训练对照。Go-Big的早期证据主要是导航迁移，不能推广成所有动作均不需机器人数据。

**必须保留的边界：** 上传小时数不等于有效训练小时；“主要数据来源”需区分采集量、采样权重、成本和能力贡献。

<!-- reading-footer-start -->
[前一篇：Helix 02](12_helix02.md) · [接着读：Helix 2.5](13_helix25.md) · [返回Figure入口](../figure/README.md)
<!-- reading-footer-end -->
