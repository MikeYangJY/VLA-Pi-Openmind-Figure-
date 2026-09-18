# π0.5：只沿着开关追三条路径

入口：[Pi0Config](code/src/openpi/models/pi0_config.py)。设置`pi05=True`后，模型仍使用[Pi0](code/src/openpi/models/pi0.py)类；PyTorch同理。

## 路径一：state从哪里进入

| 项目 | π0 | π0.5（默认配置） |
|---|---|---|
| state表达 | 连续向量投影为专家的state token | 归一化state分箱后写入语言前缀 |
| 关键代码 | `Pi0.embed_suffix`中的`state_proj` | `ModelTransformFactory`→`TokenizePrompt`→`PaligemmaTokenizer.tokenize` |
| 谁先接触state | 动作专家侧 | VLM前缀侧 |
| 文本最大长度 | 默认48 | 默认200，需容纳任务与离散state |

请按这个顺序点开：

1. [config.py](code/src/openpi/training/config.py)的`ModelTransformFactory`：PI05分支传`discrete_state_input`。
2. [transforms.py](code/src/openpi/transforms.py)的`TokenizePrompt`：决定是否传state。
3. [tokenizer.py](code/src/openpi/models/tokenizer.py)的`PaligemmaTokenizer.tokenize`：将state按256箱编码，写进`Task: …, State: …; Action:`格式。
4. [pi0.py](code/src/openpi/models/pi0.py)的`embed_suffix`：`if not self.pi05`意味着只有π0添加连续state token。

变换顺序里，state先归一化，再token化；token化通常在统一padding之前。因此不要把所有补零维度都想成必然进入state文本。

## 路径二：时间如何告诉动作专家

π0把时间embedding复制到每个动作token，与动作特征拼接后过MLP。π0.5让动作token保留动作信息，时间embedding经过MLP成为`adarms_cond`。

继续追到[gemma.py](code/src/openpi/models/gemma.py)里的`RMSNorm.__call__`：有cond时产生scale、shift和gate，对归一化特征进行调制，并控制残差贡献。它告诉模型“当前动作猜测有多少噪声”。

注意：这里的时间是flow时间，不是任务已经过去几秒，也不是MEM里的历史记忆。

## 路径三：数据配置也会变

[DataConfigFactory.create_base_config](code/src/openpi/training/config.py)默认使π0使用mean/std，而π0.5使用分位数归一化。最终应以配置和checkpoint携带的统计为准。

LIBERO示例也有配置差别：一些旧π0配置启用额外delta变换，π0.5示例不启用。**这是特定数据/checkpoint兼容要求，不能归纳为“π0和π0.5天生输出两种不同物理动作空间”。**

## 哪些没有变

两者在这里都通过连续动作专家学习flow向量场，从噪声采样action chunk；平台输入输出、Policy封装、训练循环和服务接口大量共用。

## 最容易把论文和代码混起来的地方

上游README说明其π0.5预训练checkpoint使用过KI；同时明确公开仓库目前只支持π0.5 flow head训练/推理。你在这条微调路径里不会找到完整的“FAST动作CE + flow loss + KI隔离 + 全部异构共训”的论文级训练系统。

所以要分清：**权重是怎样预训练出来的**，以及**这份开源代码接下来怎样加载、微调和运行权重**。freeze_filter和LoRA也不能自动当成完整KI实现。

**自测：** 只改pi05=True但仍用错误的tokenizer/统计/权重会怎样？为什么π0.5把state离散了，却仍是连续动作策略？
