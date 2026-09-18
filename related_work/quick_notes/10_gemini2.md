# Gemini Robotics 2 系列

**核心大纲：** ER 2规划协作 + GR 2全身动作 + On-Device 2本地适配。

**来源类型：** 官方发布（2026-07-30） · [原始来源](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/) · [展开阅读](../deepmind/03_updates.md)

## 1）要解决的问题

从上半身操作扩展到全身移动操作、灵巧手与多机器人协作，同时降低新本体接入成本。

## 2）方法

区分三个产品：ER 2做高层推理与协调；GR 2输出机器人动作；On-Device 2针对本地运行和新本体适配。不能把三个产品的参数、评测和访问权限混用。

## 3）实验与结论

官方给出全身、夹爪和多指任务结果；同一GR 2 checkpoint用于三种硬件配置。多指操作仍有明显困难。On-Device 2报告新双臂本体通常用少于200个示范适配；这是有数据适配，不是新本体zero-shot。

**必须保留的边界：** 发布材料未给完整训练配方，也未证明与Genie已形成机器人控制训练闭环。
