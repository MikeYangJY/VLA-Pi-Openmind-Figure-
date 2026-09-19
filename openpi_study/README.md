# openpi 中文代码学习：只学 π0 与 π0.5

这是官方 [openpi 固定版本](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479) 的中文学习副本。上游版本：`215abfb217dbac7d5f1273282331b9b1866c0479`，核对日期：2026-09-18。

**先读代码树，再跟一次推理，然后理解训练，最后比较π0与π0.5。第一遍不必从第一行读到最后一行。**

如果卡在“噪声为什么能变成动作”，先看新增的 [flow matching 零基础教程](07_FLOW_MATCHING_FROM_ZERO.md)：训练样本、五步计算、两个数字的手算、推理与真实控制周期的区别。再读 [π0.7 世界模型对照](../related_work/pi/10_pi07_world_model_vs_action_expert.md)，理解动作标签与未来图像标签的不同。源码主线仍只覆盖 π0 / π0.5。

想看数据如何变成可信训练样本，补读[运动学检查、多摄像头成功验证与视觉归一化](../related_work/pi/11_data_quality_three_concepts.md)，其中图像部分逐步对应本版本源码。

## 从这六步开始

1. [代码树：每个目录干什么](01_CODE_TREE.md)
2. [小白阅读路线与必要语法](02_START_HERE.md)
3. [π0：跟踪一条观测怎样变成动作](03_PI0_WALKTHROUGH.md)
4. [π0.5：沿着pi05开关看差异](04_PI05_DIFF.md)
5. [从数据、训练到部署的完整流程](05_TRAINING_AND_DEPLOYMENT.md)
6. [PyTorch对照与底层模块阅读](06_BACKENDS_AND_COMPONENTS.md)

## 文件在哪里

- [code/](code/)：保留官方目录结构的源码；Python文件新增中文定位、函数说明和关键计算注释。
- [逐文件目录](FILE_INDEX.md)：140个文件各有对应说明，含配置、测试、文档、notebook与许可。
- [术语速查](GLOSSARY.md)：tensor、batch、token、mask、KV cache、JIT等遇到再查。
- [版本信息](UPSTREAM.json)、[逐文件变更记录](ANNOTATION_MANIFEST.json)、[验证结果](VERIFICATION.md)。

学习主线只覆盖π0、π0.5及共用管线。为保持上游依赖结构，FAST和其他实验文件仍保留，标为“可跳过”；这不要求你学习它们。`tokenizer.py`只需先看PaligemmaTokenizer，`config.py`只看π0/π0.5相关配置。

## 源码与论文的边界

本版本提供π0、π0.5的连续flow动作头训练/推理。上游说明预训练权重使用过KI，但这里没有展示完整论文级异构预训练、FAST联合监督和KI训练配方。学会这份代码，不等于复现了论文全部训练流程。

`π0.5`也不是一份完全独立的`pi05.py`：它主要通过`Pi0Config(pi05=True)`走共享实现的不同分支。π0.6、MEM、RECAP和π0.7不作为此源码的已实现功能。

## 使用与许可

代码阅读无需下载模型或安装GPU环境。真正运行需按本版本[上游README](code/README.md)检查Linux、NVIDIA GPU和依赖；本次未运行真机、未下载大模型权重或执行完整训练。

`code/` 是嵌在调研仓库中的源码快照，安装时应把它视作项目根目录。其内嵌 `.gitmodules` 不会自动成为父仓库的 Git 子模块；涉及 ALOHA/LIBERO 的运行环境还需按 [UPSTREAM.json](UPSTREAM.json) 记录的来源和 commit 单独准备外部项目。第一轮阅读模型无需这一步。

原始[Apache 2.0许可](code/LICENSE)、[Gemma相关条款](code/LICENSE_GEMMA.txt)和源码版权声明完整保留。Python中的新增内容以中文学习注释标记；配置、许可等其他上游文件保持原字节。外部ALOHA/LIBERO子模块只记录来源及固定commit，未把外部项目全部纳入注释范围；见[版本记录](UPSTREAM.json)。

[返回调研资料库](../README.md) · [简洁论文笔记](../related_work/quick_notes/README.md)
