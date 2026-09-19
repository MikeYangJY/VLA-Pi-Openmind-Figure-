# 小白应该先看哪里，再看哪里

目标不是背文件名，而是能回答四句话：输入是什么、输出是什么、训练为什么能学会、π0.5改了哪里。

## 第一轮：只跟一次推理

| 顺序 | 文件/入口 | 这次只回答一个问题 |
|---|---|---|
| 1 | [libero_policy.py](code/src/openpi/policies/libero_policy.py)：make_libero_example、LiberoInputs、LiberoOutputs | 一条样本有哪些图片、状态和语言？输出为什么裁成7维？ |
| 2 | [policy_config.py](code/src/openpi/policies/policy_config.py)：create_trained_policy | 权重、平台变换和归一化统计怎样装成一个策略？ |
| 3 | [policy.py](code/src/openpi/policies/policy.py)：Policy.infer | 原始字典在调用模型前后经历了什么？ |
| 4 | [model.py](code/src/openpi/models/model.py)：Observation | 模型真正接收的统一格式是什么？ |
| 5 | [pi0.py](code/src/openpi/models/pi0.py)：sample_actions | 噪声为什么会变成一整段动作？ |
| 6 | [action_chunk_broker.py](code/packages/openpi-client/src/openpi_client/action_chunk_broker.py) | 模型预测一段动作后，客户端怎样逐步消费？ |

这一轮可以先把SigLIP/Gemma当成“输入token、输出特征”的模块。把`policy.infer()`画成一条数据流就达标。

## 第二轮：理解模型如何学习

先读 [flow matching 零基础教程](07_FLOW_MATCHING_FROM_ZERO.md)，弄清一条动作样本如何变成带噪输入和监督目标，再进代码。

读[π0逐步讲解](03_PI0_WALKTHROUGH.md)，顺序是：`compute_loss` → `embed_prefix` → `embed_suffix` → `make_attn_mask`。再看[train.py](code/scripts/train.py)里的`train_step`，把“计算误差”和“更新参数”接上。

此时只需认识三种张量：图片、语言token、动作块。先写shape，再看公式，遇到JAX语法查下面表格。

## 第三轮：只看π0.5差异

按[差异表](04_PI05_DIFF.md)追踪`pi05`、`discrete_state_input`、`adarms_cond`。不要把整套代码再读一遍；在已经理解的π0路径上标出变化。

## 第四轮：连上数据与工程

读[训练与部署流程](05_TRAINING_AND_DEPLOYMENT.md)，先选择LIBERO一个例子走通。然后按需要选DROID或ALOHA，最后再研究多GPU分片、第三方模型兼容层和LoRA。

## 看得懂代码所需的最少语法

| 写法 | 用大白话理解 |
|---|---|
| `class Policy` | 把相关函数与状态装在一个对象里；`self`表示这个对象 |
| `@dataclass` | 把配置字段变成容易创建、打印和传递的数据对象 |
| `config.model.create(...)` | 根据配置造模型；不意味着已经训练好 |
| `x.shape` | 数组每一轴有多少元素，先看shape常比先看公式更容易 |
| `[B,H,A]` | B条样本，每条预测H个时刻，每时刻A个动作坐标 |
| `x[:,None,:]` | 插入一个长度为1的轴，方便与其他数组广播 |
| `x[..., :7]` | 保留前面的所有轴，只在最后一轴取前7项 |
| `concatenate(...,axis=1)` | 沿指定轴接起来；接token轴不等于接特征轴 |
| `einsum` | 用字母显式说明哪些轴做乘法、求和和保留 |
| `jax.tree.map` | 对嵌套字典/列表的每个数组执行同一处理 |
| `jax.random.split` | 从一个随机key产生不同子key，避免重复采同一个随机量 |
| `jax.jit` | 把数值函数编译后更快执行；首次调用可能额外编译 |
| `value_and_grad` | 同时求loss和它对参数的导数 |
| `torch.no_grad()` | 推理时不保存用于反向传播的计算图 |
| `yield` | 每次交出一个样本/batch，下一次调用继续往后读 |

## 每读完一个函数做这张小卡片

1. 谁调用它？
2. 输入是什么，shape与单位是什么？
3. 哪一步改变数值，哪一步只改变布局/字段？
4. 输出交给谁？
5. 它属于训练、推理还是环境执行？

卡住时先看该文件顶部中文说明，再点[逐文件目录](FILE_INDEX.md)中的函数定位。第一轮可以跳过FAST、FSQ、RoboArena/Polaris配置，以及与所选机器人无关的真机脚本。
