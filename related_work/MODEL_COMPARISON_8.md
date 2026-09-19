# 八个模型核心对照：以 π0 为架构基准

<!-- reading-nav-start -->
[首页](../README.md) · [横向对比入口](README.md) · [按问题查找](FIND_BY_QUESTION.md)
<!-- reading-nav-end -->

核对日期：**2026-09-19**。覆盖用户指定的八个模型；Gemini Robotics 2、Helix 02 使用官方名称。**未披露≠没有该能力；以下数字也不是跨公司统一排行榜。**

**核心大纲：架构与数据 → 已见/未见任务成功率 → 速度与时长 → 核心问题与不足 → 数字出处。**

[下载完整单表 CSV](data/model_comparison_8.csv)（8个模型、全部维度、来源；可用Excel打开）。下面是同一张表的分栏视图，方便在GitHub阅读。公开资料报告的是“实际用了多少数据”，一般没有证明“至少需要多少数据”。

**第一次读：** 先看第1栏架构和第3栏“核心解决的问题”，建立模型定位；再看数据、成功率和时长，最后按需要核对来源。读完进入[③ Figure路线](figure/README.md)，或返回[学习路线](LEARNING_PATH.md)。

## 阅读前的四个口径

- **完整成功率**：每个测试回合按任务标准判成/败；**progress**：完成了多少步骤，可给部分分。两者不能互换。
- **新任务**、**新环境/物体下的旧任务**、**新任务×机器人组合**分别写明。few-shot适配和实时人工coaching不计为完全自主zero-shot。
- **速度**分秒/任务、成功次数/小时、推理延迟、动作执行Hz；**时长**分单任务持续时间、评测超时、重复任务累计运行时间。
- “需要优化”优先填作者明确局限或实验暴露的缺口；没有公开失败分析就写证据不足，不推测公司内部问题。

## 1. 架构与训练数据

π0基准数据流：**多视角图像 + 语言 + 当前本体状态 → VLM与flow动作expert → 连续动作块 → 机器人控制器**。不要把所有VLA的动作解码器都想成同一种flow网络。

| 模型 | 相对π0的架构差异 | 训练数据类型 | 已披露数据量级 |
| --- | --- | --- | --- |
| π0<br>[P0](papers/pi/pi0.pdf) | 基准：PaliGemma约3B + 约300M连续动作expert；多视角图、语言、当前state → flow matching动作块。语言视觉知识与高频连续控制连接；高层规划器不是此动作网络必备组件。 | 图像、机器人state、连续action、任务/片段语言配对；多机器人遥操作；OXE、DROID、Bridge等公开机器人数据；继承VLM图文预训练。 | 自有约10,000小时、7种机器人配置、68任务，另加公开数据；不是全部来源相加后的精确总时长。下游后训练约5至100+小时/任务，不能视为统一最低需求。 |
| π0.5<br>[P05](papers/pi/pi05.pdf) | 保留PaliGemma + flow动作expert；增加高层文字子任务→低层动作的层级推理和异构任务共同训练。原论文先训练离散FAST/文字，再加入连续expert联合后训练；不要与后续KI版checkpoint混为一谈。 | MM移动操作、ME多家庭固定臂、CE跨本体实验室/公开机器人轨迹；高层子任务与物体定位标签；网页caption/VQA；后训练加入专家语言干预。机器人动作训练仍需要state/action配对。 | MM约400小时、约100个家庭环境；这只是一个数据子集。完整机器人+网页总量未披露，不能称“π0.5只需400小时”。 |
| π0.7<br>[P07](papers/pi/pi07.pdf) | Gemma 3 4B + 860M flow expert（VLA约5B）；KI训练、MEM式历史；增加metadata、可选视觉子目标、控制模式及RTC。另有高层语言policy和BAGEL初始化世界模型，不计入上述VLA约5B。 | 跨本体示范、失败/次优轨迹、先前策略自主执行与纠错、人类第一视角及非机器人数据；细粒度子任务、质量/速度/错误metadata；世界模型另用当前→子任务末尾图像对。 | 完整小时数、episode数和各来源比例未披露。论文的数据子集/扩量消融不是完整语料绝对规模；不能由参数量倒推需要多少数据。 |
| Gemini Robotics<br>[G1](papers/deepmind/gemini_robotics.pdf) / [G15](papers/deepmind/gemini_robotics_15.pdf#page=54) | Gemini-ER蒸馏出的云端VLA backbone + 机器人端action decoder，补偿云端延迟；与π0公开flow结构相比，其decoder损失/精确内部结构及参数量未完整披露，不标成flow或diffusion。 | ALOHA 2遥操作图像/状态/动作；网页文档、代码、图像/音频/视频、具身推理/VQA；新任务/本体的实验另有专项示范。 | 千小时级真实遥操作、数千任务，12个月采集；完整精确总量未披露。专项训练2,000–5,000 episodes/任务；少样本实验≤100示范是适配规模。 |
| Gemini Robotics 1.5<br>[G15](papers/deepmind/gemini_robotics_15.pdf#page=54) | 多本体VLA + Motion Transfer；可启用Thinking VLA，先生成文字推理再动作。完整agent另有ER 1.5做规划/工具调用/成功判断。相对π0新增显式推理和跨本体训练机制；动作decoder具体损失未披露。 | ALOHA、双臂Franka、Apollo多本体机器人数据；公开文字/图像/视频。包含用于动作与具身推理的训练；完整Thinking/MT标注生成细则未公开。 | 数千种任务；完整小时数、轨迹数未披露。“>90%在仿真”指开发评估episode，不能写成90%训练数据来自仿真。 |
| Gemini Robotics 2（用户称2.0）<br>[G2](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/) | 旗舰多本体VLA扩展到全身和多指手控制；ER 2另负责任务组织/多机器人协作。未公开足以对齐π0的内部层次、参数、动作损失；On-Device 2是另一模型。 | 旗舰训练数据构成、各监督字段和来源配比未完整披露；控制三种配置的展示不能反推训练语料。 | 旗舰总小时/轨迹数未披露。官方“通常<200示例适配新本体”属于On-Device 2，不能填为旗舰训练总量。 |
| Helix 02（用户称2.0）<br>[H02](https://www.figure.ai/news/helix-02) | S2语义latent → S1 transformer全身关节目标（200Hz）→ S0全身控制器（1kHz、10M参数）；加入掌部视觉/触觉。相对π0，明确把学习到的平衡/接触控制纳入三级系统；S1动作损失未披露。 | 明确披露：S0用关节级重定向人类运动及仿真RL；S1输入头/掌相机、指尖触觉、本体状态。S1/S2完整训练语料与配方未披露，输入传感器不等于已公开训练数据清单。 | S0 >1,000小时人类运动、>200,000并行仿真环境；后者是并行度，不是示范条数。全系统训练总量未披露。 |
| Helix 2.5<br>[H25](https://www.figure.ai/news/helix-2-5-zero-shot-30-home-generalization) | 核心已披露变化是从随机初始化仅用Index预训练，再分别适配三个行为；02从预训练VLM开始。2.5内部层数、动作loss及是否完全沿用02频率未披露。 | Index人类行为数据 + 各行为的任务适配数据；评估家庭/被操作物体不参与适配。公开说明未给出完整传感器字段与监督构造。 | 预训练/适配绝对小时数未披露；某代表行为适配数据是02的1/2。35分钟/秒是Index新增采集速率，不是本次训练量。 |

## 2. 泛化：把成功率及测试条件放在一起

| 模型 | 已见任务成功率 | 未见任务及其他泛化成功率 |
| --- | --- | --- |
| π0<br>[P0](papers/pi/pi0.pdf) | 未披露覆盖全部任务的统一二元成功率。Fig.7五类基础任务主要报归一化progress；即使均值接近1，也不能直接写成成功率。 | 测试包含新物体/场景变化及新任务后训练；未披露可直接填入的“所有未见任务zero-shot成功率”。后训练后会做新任务，不等于零示范泛化。 |
| π0.5<br>[P05](papers/pi/pi05.pdf) | 主实验针对未见环境；未给出可与其他模型直接比较的“已见任务总体二元成功率”。 | 3个未见真实家庭的厨房/卧室评测主要是分步骤progress（Fig.7），不是整任务二元成功率。主要证明已学家务在新家庭泛化；不能写成任意未见任务的成功率。 |
| π0.7<br>[P07](papers/pi/pi07.pdf) | 结论概括：已见任务常超过90%；不是全任务加权均值，也不是每项任务都>90%。Fig.6另列任务success、progress及吞吐量，指标必须分开。 | 结论概括：未见任务或未见任务×机器人组合约60–80%。跨本体UR5e叠衫80%成功（人类80.6%）；短任务可直接提示，复杂新任务的coaching与高层后训练另计，不能统称自主zero-shot。 |
| Gemini Robotics<br>[G1](papers/deepmind/gemini_robotics.pdf) / [G15](papers/deepmind/gemini_robotics_15.pdf#page=54) | 1.5论文同基准重测：ALOHA分布内63%（Fig.35）。原版论文20任务中一半>80%，不能写成全体平均80%；专项后训练平均79%属于另一实验。 | 1.5论文同基准重测：ALOHA Task Generalization 19%（Fig.35）。原版≤100示范适配后7/8任务>70%，属于few-shot，不能作为zero-shot新任务成功率。 |
| Gemini Robotics 1.5<br>[G15](papers/deepmind/gemini_robotics_15.pdf#page=54) | 二元成功率：ALOHA72%、双臂Franka58%、Apollo64%（Fig.35，分布内）；无须在这些评测任务上再专项后训练。 | Task Generalization：ALOHA39%、Franka33%、Apollo40%（Fig.35）；新指令/物体/初始条件及环境的组合，具体任务见附录。另有跨本体技能迁移，不能等同完全陌生硬件即插即用。 |
| Gemini Robotics 2（用户称2.0）<br>[G2](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/) | 已见/未见拆分未披露。官方未分组成功率示例：Apollo桌面拾取68.4%、地面45.7%、架子76.3%；不能强行归为已见任务。 | 独立“未见任务”成功率未披露。未分组多指任务如拧灯泡36%、拧下92%；同checkpoint覆盖三种配置，不代表对任意新本体zero-shot。 |
| Helix 02（用户称2.0）<br>[H02](https://www.figure.ai/news/helix-02) | 未披露规范测试集的汇总成功率；4分钟成功演示不能写成100%。 | 未披露严格未见任务测试集成功率。 |
| Helix 2.5<br>[H25](https://www.figure.ai/news/helix-2-5-zero-shot-30-home-generalization) | 训练分布内统一成功率未披露。 | 30个未见家庭：三个已训练行为的完整任务成功率56%，相同任务数据从零训练对照9%；新环境/新物体zero-shot，不是从没训练过这三个任务。每任务一个固定checkpoint。 |

**可以做局部同基准比较：** Gemini Robotics 1.5论文Fig.35在ALOHA上同时重测了原版与1.5：分布内 **63%→72%**，Task Generalization **19%→39%**。这组比较有同一论文内的依据；不能把它与Helix 2.5的56%或π0.7结论中的范围直接排序。[G15](papers/deepmind/gemini_robotics_15.pdf#page=54)

**GR 1.5的“新任务”具体是什么？** §3.1定义为新任务/新环境的综合变化；ALOHA附录B.2.1.2列出12种任务，使用未见指令、物体和初始条件。它包含已知技能的重组，并不等于每个原子动作都没有训练过。Franka/Apollo采用各自任务清单，不能把三个百分比平均后充当统一总体成功率。

## 3. 速度、任务时长、问题与不足

| 模型 | 完成速度与推理频率 | 任务时长及条件 | 核心解决的问题 | 公开短板或证据缺口 |
| --- | --- | --- | --- | --- |
| π0<br>[P0](papers/pi/pi0.pdf) | 统一任务耗时未披露；论文控制可达50Hz，属于动作执行频率，不是每秒完成50次任务。 | 后训练后的复杂任务约5–20分钟；部分需要高层policy引导。是实验任务跨度，不是基础模型保证的最长无人干预时长。 | 让预训练VLM生成灵巧、连续、高频的机器人动作，并使多机器人预训练支持任务后训练。 | 论文明确：部分任务仍不可靠；最佳数据组成/权重、达到高可靠性的数据需求和更广跨领域迁移仍未解决。 |
| π0.5<br>[P05](papers/pi/pi05.pdf) | 统一完成任务耗时未披露；实验目标命令50Hz。控制频率不可用于推断比π0快多少。 | 摘要报告10–15分钟多阶段家务；具体定量测试按独立任务和评分规则进行。这不是“连续15分钟必定成功”的统计保证。 | 把异构机器人与非机器人知识迁移到新家庭，减少逐家庭采集/适配。 | 论文明确：陌生把手/难开的柜门、遮挡、重复开关抽屉；复杂指令能力及记忆上下文有限。 |
| π0.7<br>[P07](papers/pi/pi07.pdf) | Fig.6报告相对专项策略的成功次数/小时，部分洗衣/搭盒任务更高；无统一秒/任务。实验50Hz、50步块、5次去噪，执行15/25步前缀；这些不是任务完成时间。 | 新复杂厨房任务最长约5分钟的交互，需要逐步语言coaching；随后可用coaching记录训练高层自主执行。全模型通用最大任务时长未披露。 | 用丰富上下文利用混合质量/多来源数据，把专项灵巧能力、组合泛化和可引导性汇入通用policy。 | 论文明确：未见任务可靠性仍低于已见任务；大规模混合数据下“真正未见”难彻底审计；复杂新长任务仍需引导或适配。 |
| Gemini Robotics<br>[G1](papers/deepmind/gemini_robotics.pdf) / [G15](papers/deepmind/gemini_robotics_15.pdf#page=54) | 原论文：backbone延迟<160ms，端到端约250ms，有效控制50Hz。统一任务完成速度未披露；延迟不是完成一项任务的时间。 | 专项午餐盒任务超过2分钟；需任务后训练。完整通用最大任务时长未披露。 | 把Gemini语义/具身知识转成可实时执行的灵巧动作，并支持少样本适配。 | 同基准新任务19%显示泛化仍有缺口；原论文困难任务/新本体依赖专项适配，后续1.5论文明确这类专用模型的泛化有限。 |
| Gemini Robotics 1.5<br>[G15](papers/deepmind/gemini_robotics_15.pdf#page=54) | 未披露统一秒/任务、可直接比较的整任务加速比或此版本统一动作延迟；不能沿用原版250ms当作1.5测量。 | 论文含多步及ER+VLA长程任务；未给统一最大持续分钟数。ER+VLA整系统结果与单独VLA任务必须分开。 | 同一checkpoint控制多个本体；通过Motion Transfer与Thinking提高技能迁移、复杂指令执行和多步任务能力。 | 新任务完整成功率仍只有33–40%（上述基准）；正文progress高于完整成功率；完整agent仍有规划与执行错误。 |
| Gemini Robotics 2（用户称2.0）<br>[G2](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/) | 统一秒/任务未披露；官方明确移动速度、精度仍需提高。不要借用On-Device 2延迟或旧版频率。 | ER 2 + VLA整系统支持数分钟、数百决策的任务序列；未披露标准化最大时长。 | 全身移动与操作、多指灵巧，以及由ER 2组织的多机器人协作。 | 官方明确：多指操作仍困难，移动速度及精度/速度距人类水平仍有提升空间。 |
| Helix 02（用户称2.0）<br>[H02](https://www.figure.ai/news/helix-02) | 洗碗机演示4分钟；S1 200Hz、S0 1kHz不是整任务速度或快于其他VLA的证据。 | 发布页：4分钟连续自主、61个动作、无重置/干预的演示；不是最长时长统计保证。 | 把步行、平衡、全身操作与触觉灵巧控制结合。 | 官方称早期结果；失败分布、未见任务与长期可靠性量化基准未披露，不能替官方编造具体性能短板。 |
| Helix 2.5<br>[H25](https://www.figure.ai/news/helix-2-5-zero-shot-30-home-generalization) | 统一平均完成时间未披露。玩具1分钟/件、毛巾3分钟/件、枕头/被角1分钟/项是评测超时，不是实际耗时。 | 多阶段整屋任务；未披露统一最长持续时长。不要把Helix 02历史200小时重复物流运行移植为2.5单任务时长。 | 用广泛人类经验预训练，减少新家庭的数据采集与逐环境适配。 | 该基准完整成功率56%，仍有失败；证据范围限三个行为/30家庭，任意新任务及更广部署可靠性未披露。 |

## 4. 数字与原文定位

| 编号 | 原始资料 | 核对位置与用途 |
|---|---|---|
| P0 | [π0论文](papers/pi/pi0.pdf) · [arXiv](https://arxiv.org/abs/2410.24164) | §IV、§V、§VI；PDF第7页明确Fig.7采用允许部分分的normalized score、复杂任务5–20分钟且部分依赖高层引导 |
| P05 | [π0.5论文](papers/pi/pi05.pdf) · [arXiv](https://arxiv.org/abs/2504.16054) | §IV-C/D/E、Fig.7、§VI；400小时仅MM切片；摘要的10–15分钟不应覆盖所有定量测试 |
| P07 | [π0.7论文](papers/pi/pi07.pdf) · [arXiv](https://arxiv.org/abs/2604.15483) | Fig.2架构；Fig.6吞吐/成功/progress；§IX-C及Appendix F跨本体80%；§IX-D新任务/coaching；§X已见>90%、未见60–80%的作者概括 |
| G1 | [Gemini Robotics论文](papers/deepmind/gemini_robotics.pdf) · [arXiv](https://arxiv.org/abs/2503.20020) | §3.1模型/数据/延迟；§3.2原版20任务；§4.1专项后训练；§4.3少样本适配 |
| G15 | [Gemini Robotics 1.5论文](papers/deepmind/gemini_robotics_15.pdf) · [arXiv](https://arxiv.org/abs/2510.03342) | §2模型/数据；§3.1评测分类；**Appendix B.5.1、Fig.35，PDF第54页**才是表内二元成功率；正文Fig.3是progress |
| G2 | [官方发布页](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/) · [旗舰VLA页](https://deepmind.google/models/gemini-robotics/vla/) | Whole-body / dexterity图表及ER 2系统时长；官网把VLA、ER、On-Device分开，表中也分别归属 |
| H02 | [Helix 02官方页](https://www.figure.ai/news/helix-02) | S0/S1/S2、训练数据与Results；S0的1,000+小时不是全模型数据量 |
| H25 | [Helix 2.5官方页](https://www.figure.ai/news/helix-2-5-zero-shot-30-home-generalization) | Key Results、Fig.1、评估条件、Appendix超时规则；56%是全任务不给部分分，每行为各自固定checkpoint |

### 为什么不把所有空格都填成百分比？

π0/π0.5若只从progress柱形图估一个数再标“成功率”，会制造原文不存在的指标。π0.7部分图没有数值标签，本表不从柱高读出伪精确小数，而引用原文明确数字和作者的范围概括。Helix 02发布演示没有测试回合总数，不能以“展示的片段都成功”计算成功率。

对于Gemini Robotics 2，官方图表有成功率，但没有给表内所需的已见/未见拆分，因此保留数值、明确“未分组”，不替其分组。若后续官方补充论文，应补充各数据切分、样本数/误差条和完整训练配方，再修订本表。

### 继续阅读

- [从零理解action expert与flow matching](../openpi_study/07_FLOW_MATCHING_FROM_ZERO.md)
- [π0.6、π0.6*、MEM区别](pi/09_pi06_family.md) · [π0.7世界模型与动作expert的数据区别](pi/10_pi07_world_model_vs_action_expert.md)
- [三项数据质量概念：运动学检查、多摄像头成功验证、视觉归一化](pi/11_data_quality_three_concepts.md)
- [回到调研总入口](README.md)

<!-- reading-footer-start -->
[接着读：Figure](figure/README.md) · [返回横向对比入口](README.md)
<!-- reading-footer-end -->
