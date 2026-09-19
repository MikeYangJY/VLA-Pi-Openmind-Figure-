# ⑤ openpi代码学习：只学π0与π0.5

**目标：看懂输入怎样变成动作、训练怎样更新模型，以及π0.5改了哪里。** 第一遍从一条数据流出发，随后逐文件补全。

## 按四轮阅读

开始前扫一遍[代码树](01_CODE_TREE.md)。flow概念还不熟时，先看[零基础教程](07_FLOW_MATCHING_FROM_ZERO.md)；不需要先安装GPU环境才能读代码。

| 轮次 | 主入口 | 对应重点 | 读完的结果 |
|---|---|---|---|
| 1：跟一次推理 | [初学者阅读顺序](02_START_HERE.md) | 从LIBERO样本 → Policy.infer → Observation → sample_actions → 动作块消费 | 画出一条完整输入输出链 |
| 2：理解训练 | [π0逐步解释](03_PI0_WALKTHROUGH.md) + [数据、训练与部署](05_TRAINING_AND_DEPLOYMENT.md) | compute_loss、训练样本、归一化、train_step | 分清计算loss与更新参数 |
| 3：比较π0.5 | [沿三个开关看差异](04_PI05_DIFF.md) | state输入、时间条件、配置差异 | 在π0路径上标出变化 |
| 4：补工程与组件 | [训练与部署](05_TRAINING_AND_DEPLOYMENT.md) → [后端与底层组件](06_BACKENDS_AND_COMPONENTS.md) | 先选一个平台，再补JAX/PyTorch、视觉/语言模块和工程工具 | 理解依赖关系，按需要深入 |

文件数字用于定位；按上面的阅读轮次走。第4轮可按当前需要选择，不要求先把两个后端都学完。

## 文件太多时，用这三个入口

- [代码树](01_CODE_TREE.md)：看目录分别做什么。
- [逐文件目录](FILE_INDEX.md)：140个文件都有说明，找到具体文件后再打开注释源码。
- [术语表](GLOSSARY.md)：遇到shape、mask、KV cache、gradient等词再查。

想先弄懂数据清洗与图像输入，补读[运动学检查、多摄像头成功验证、视觉归一化](../related_work/pi/11_data_quality_three_concepts.md)。

## 范围与版本

这是[官方openpi固定版本](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479)的中文学习副本，上游commit为`215abfb217dbac7d5f1273282331b9b1866c0479`。源码与事实按该版本解释。

`π0.5`主要通过`Pi0Config(pi05=True)`走共享实现的不同分支。完整快照保留依赖结构；FAST、FSQ、RoboArena/Polaris等旁支按目录标记选读。本学习主线不扩展为π0.6、MEM、RECAP或π0.7的代码复现。

**论文与开源实现有边界：** 本版本提供连续flow动作头训练/推理；上游说明预训练权重使用过KI，但这里没有展示完整论文级异构预训练、FAST联合监督和KI训练配方。

<details>
<summary>准备真正运行时，再看环境、版本验证与许可</summary>

`code/`是项目根目录。按[固定版本上游README](code/README.md)检查Linux、NVIDIA GPU与依赖；本库未运行真机或完成全量模型训练。涉及ALOHA/LIBERO的外部Git子模块需按[UPSTREAM.json](UPSTREAM.json)所记来源/commit另行准备。

[版本信息](UPSTREAM.json) · [注释变更记录](ANNOTATION_MANIFEST.json) · [验证结果](VERIFICATION.md)。91个Python文件增加中文注释；原始Python逻辑经一致性验证，其余上游文件保持原字节。

[Apache 2.0许可](code/LICENSE)、[Gemma条款](code/LICENSE_GEMMA.txt)及源码版权声明保留。外部子模块记录来源与版本，未展开为本库逐文件注释范围。

</details>

**上一阶段：** [④ 机制与数据](../related_work/MECHANISMS.md) · **下一阶段：** [⑥ 访谈准备](../related_work/INTERVIEW_PREP.md) · [首页](../README.md)
