# 随用随查的术语

<!-- reading-nav-start -->
[首页](../README.md) · [代码学习入口](README.md) · [按问题查找](../related_work/FIND_BY_QUESTION.md)
<!-- reading-nav-end -->

| 词 | 在本项目中的含义 |
|---|---|
| Transformer | 用attention和前馈网络等逐层加工表示的架构，可用于语言、视觉及动作模块；先看[杯子例子的白话解释](../related_work/00_START_HERE.md#transformer) |
| observation | 当前传感器/状态与指令的集合 |
| policy | 给定观测产生动作的策略；代码中还包含前后处理包装 |
| tensor / array | 带多个轴的数字表；shape决定各轴含义 |
| batch | 同时处理的一组样本；B是样本数 |
| token | 序列中的一个位置，可以表示文本、图像patch、state或一个动作时刻 |
| embedding | 把离散编号或原始数值映射成模型内部的特征向量 |
| prefix / suffix | 条件token段 / 动作专家相关token段，具体组成由模型分支决定 |
| action chunk | 一次预测的多步未来动作序列 |
| action horizon | 动作块包含多少控制时刻 |
| flow time | 从数据与噪声插值的位置；不是机器人运行时钟 |
| flow matching | 学习动作分布上的向量场，推理沿该场把噪声变成动作 |
| MSE | 预测与目标的平方误差平均 |
| attention | 每个query根据与key的匹配，对value内容加权汇集 |
| mask | 控制哪些数据有效，或哪些token彼此可见 |
| KV cache | 缓存已经算好的key/value，避免重复计算不变前缀 |
| RMSNorm / AdaRMS | 均方根归一化 / 用条件调制的归一化 |
| PyTree | 字典、列表、对象等形成的数组嵌套结构 |
| JIT | 将函数编译成更高效的数值执行程序 |
| gradient | loss对参数的变化方向；优化器据此更新权重 |
| LoRA | 用低秩矩阵表示参数更新，减少可训练参数 |
| EMA | 模型参数的指数滑动平均 |
| checkpoint | 磁盘上的参数/训练状态存档 |
| gradient checkpointing | 为省显存而在反传时重算部分激活；不是磁盘存档 |
| DDP / FSDP | 不同的多设备训练策略；具体支持和实现依后端而定 |
| LeRobot / RLDS | 数据组织/读取体系，不是π模型结构 |
| episode | 一次任务尝试及其时间序列 |
| rollout | 用策略在环境中执行得到的轨迹 |
| normalization | 动作/状态可按数据统计缩放，输出需恢复物理尺度；图像另有像素范围约定，见[图像教程](../related_work/pi/11_data_quality_three_concepts.md) |

<!-- reading-footer-start -->
[接着读：访谈准备](../related_work/INTERVIEW_PREP.md) · [返回代码学习入口](README.md)
<!-- reading-footer-end -->
