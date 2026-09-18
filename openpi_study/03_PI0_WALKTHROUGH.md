# π0：一条观测如何变成动作

主文件：[pi0.py](code/src/openpi/models/pi0.py)。本节的数字采用代码默认配置，具体checkpoint/平台可以覆盖它们。

## 1. 从机器人字典到模型输入

以LIBERO为例，原始字典包含环境图像、腕部图像、状态和prompt。`LiberoInputs`把键名统一，给缺少的相机放占位图并设置mask。随后输入归一化、文本token化，并把state/action最后一维补到模型宽度。

| 数据 | 加batch轴后的典型shape | 含义 |
|---|---|---|
| 每路图像 | `[B,224,224,3]`（JAX） | RGB；模型预处理使用约[-1,1]范围 |
| 每路image_mask | `[B]` | 这一路图像对各样本是否有效 |
| state | `[B,32]` | 统一宽度；真实状态不足的维度用padding |
| prompt token | `[B,48]`（π0默认） | 离散文本ID与有效性mask |
| actions | `[B,50,32]`（默认） | 50个未来控制时刻，不是50个语义动作 |

LIBERO最终需要7维动作。模型的32维是统一空间宽度，不能理解成机器人一定有32个关节。

## 2. prefix：把“看见什么、要做什么”编码好

`embed_prefix`把每路图像送入SigLIP，得到一组patch token，再拼上语言embedding。这些是**条件token**：在一次动作采样的多轮积分中，观测不变，因此可以只算一次并缓存K/V。

`input_mask`决定token有效不有效；`ar_mask`决定token属于哪个注意力块。两种mask用途不同。

## 3. suffix：给动作专家一个待完善的动作猜测

π0的`embed_suffix`做三件事：

1. 连续state经线性层变成一个state token。
2. 带噪动作`[B,H,A]`经线性层变成`[B,H,D_expert]`。
3. flow时间经sin/cos编码，与每个动作token拼接，再通过MLP融合。

默认Gemma主干宽度为2048，专家为1024，见[gemma.py](code/src/openpi/models/gemma.py)。它们的hidden width不同，但Q/K/V注意力头尺寸兼容，可以通过注意力交换信息。

## 4. 训练：学“往哪个方向修改动作猜测”

`compute_loss`中最值得逐行读的是：

```python
x_t = t * noise + (1 - t) * actions
u_t = noise - actions
loss = mean((v_t - u_t) ** 2)
```

这里省略broadcast等实现细节。`actions`是示范的归一化动作，`noise`是同shape的高斯噪声。`t=1`是纯噪声，`t=0`是真实动作。模型接收观测、`x_t`和`t`，预测向量场`v_t`，希望它接近插值路径的导数`u_t`。

**v_t是动作表示空间里的更新方向，不一定是机器人的物理速度命令。** 某平台的actions可能表示关节目标，另一平台可能表示末端增量，必须回看平台适配。

JAX版先对动作坐标维取平均，返回`[B,H]`；训练脚本再取标量均值。`compute_loss`本身没有更新权重。更新发生在`train_step`的求导和优化器步骤。

## 5. 注意力：谁能看谁

| Query想读的信息 | 图像/语言prefix | state token | 动作token |
|---|---|---|---|
| prefix | 可以 | 不可以 | 不可以 |
| state（π0） | 可以 | 可以 | 不可以 |
| 动作 | 可以 | 可以 | 整个动作块内可以互看 |

这解释了两个现象：前缀可缓存，因为它不依赖待采样动作；动作块整体预测，因为动作token不是按未来时刻逐个AR生成。`make_attn_mask`通过累计分块编号实现这种关系。

## 6. 推理：从噪声反复细化到动作

`sample_actions`流程：

1. 预处理当前观测。
2. 计算prefix并保存其K/V。
3. 生成`[B,H,A]`高斯噪声，设`t=1`。
4. 每轮构造suffix、读取prefix缓存，得到`v_t`。
5. 执行`x_t = x_t + dt*v_t`，其中`dt=-1/num_steps`。
6. 到`t≈0`输出整段动作，再由Policy做反归一化和平台转换。

默认`num_steps=10`表示积分约10轮；`action_horizon=50`表示每轮在处理一个50步动作块。这两个数字控制不同维度。

## 7. 从预测到真实执行

`Policy.infer`返回动作块；客户端可以选择执行其中一段，再取新观测重规划。`ActionChunkBroker`只是缓存并逐步消费动作，不能据此宣称实现了异步RTC。

动作输出频率、模型调用频率、积分次数、单个任务时长，应分开记录。

**自测：** 如果把action_horizon从50改成25，哪一轴变化？如果num_steps从10变成5，改变的是动作步数还是积分次数？如果换成7维动作的机器人，应改平台适配还是把模型默认32理解成32个关节？
