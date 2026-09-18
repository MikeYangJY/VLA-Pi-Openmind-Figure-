# 两个计算后端与底层组件怎样读

## JAX和PyTorch对照

| 逻辑 | JAX | PyTorch |
|---|---|---|
| 模型 | [models/pi0.py](code/src/openpi/models/pi0.py) | [models_pytorch/pi0_pytorch.py](code/src/openpi/models_pytorch/pi0_pytorch.py) |
| 训练损失 | `compute_loss`返回[B,H] | `forward`返回[B,H,A]；训练循环再平均 |
| 动作采样 | `sample_actions`，`jax.lax.while_loop` | `sample_actions`，`while`与`denoise_step` |
| 梯度更新 | [train.py](code/scripts/train.py)的`value_and_grad`/Optax | [train_pytorch.py](code/scripts/train_pytorch.py)的`backward`/optimizer |
| 大/小分支交互 | [gemma.py](code/src/openpi/models/gemma.py) | [gemma_pytorch.py](code/src/openpi/models_pytorch/gemma_pytorch.py) |
| 多设备 | JAX mesh/sharding | PyTorch DDP；支持范围看固定版本README |

选择一个后端先把公式和数据流看懂。第二个后端用于检验你理解的是算法，还是只记住了某种库的写法。

## SigLIP：从图片到token

看[SigLIP](code/src/openpi/models/siglip.py)的patch嵌入、位置编码和Encoder。`pool_type=none`使主路径保留多个空间token。初学者只需先掌握“每路图片变成一组向量”，再深入每层attention。

## Gemma：不同参数分支怎样交互

先读[gemma.py](code/src/openpi/models/gemma.py)的`Attention.__call__`。各分支可以有自己的投影权重和hidden width；投影到兼容的Q/K/V头空间后拼接token进行注意力，输出再回到各分支。这比“VLM出一句话给动作头”更准确。

接着读RMSNorm与Block，理解π0.5时间条件怎样影响归一化和残差。最后看Module中的层堆叠、扫描和KV cache。

## Transformers补丁为什么这么长

`models_pytorch/transformers_replace/`保留很多标准生成/分类兼容接口。π0主线实际关注视觉塔、Gemma模型、条件归一化和缓存行为。分类头可以最后再看，不必在入门时逐个掌握。

这些文件与上游指定的transformers版本配套。学习快照仅保留并注释源码，没有修改本机Python环境。不要把包含这些文件理解为任意已安装版本都可无条件替换。

## LoRA、分片、下载工具放在最后

- [lora.py](code/src/openpi/models/lora.py)：读懂原线性层后，再理解低秩更新及freeze_filter。
- [sharding.py](code/src/openpi/training/sharding.py)：读懂单次训练更新后，再看参数与batch分布到设备的方式。
- [download.py](code/src/openpi/shared/download.py)：需要排查权重缓存时再读。
- [测试文件](FILE_INDEX.md)：遇到一个小函数不明白，先找它的test输入和断言，常比继续追底层库更直观。
