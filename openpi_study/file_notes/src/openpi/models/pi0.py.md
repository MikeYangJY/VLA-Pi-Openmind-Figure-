# `src/openpi/models/pi0.py` 中文阅读说明

**定位：** 核心模型。

π0和π0.5共用的JAX实现，串起图像/语言条件、动作专家、flow训练和动作采样。

**建议读法：** 先读compute_loss与sample_actions，再读embed_prefix、embed_suffix和make_attn_mask。

**易错点：** t=1是噪声，t=0是真实动作；pi05开关改变状态入口和时间条件，不代表论文全部训练流程都在此实现。

[注释源码](../../../../code/src/openpi/models/pi0.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [make_attn_mask](../../../../code/src/openpi/models/pi0.py#L29) | 把token有效性mask和分块边界mask组成[B,N,N]可见性矩阵；前缀彼此可见，动作可读前缀，但前缀不能偷看带噪动作。 |
| [posemb_sincos](../../../../code/src/openpi/models/pi0.py#L64) | 把一个标量flow时间编码成不同频率的sin/cos向量；时间不是机器人时钟，而是噪声到动作的插值位置。 |
| [Pi0](../../../../code/src/openpi/models/pi0.py#L83) | π0/π0.5的共享flow模型；Pi0Config.pi05决定具体分支。 |
| [Pi0.__init__](../../../../code/src/openpi/models/pi0.py#L87) | 初始化Pi0的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.pi05、self.PaliGemma、self.action_in_proj、self.time_mlp_in、self.time_mlp_out。 |
| [Pi0.embed_prefix](../../../../code/src/openpi/models/pi0.py#L131) | 把多相机图像经SigLIP变成patch token，再拼接语言token。返回[B,P,D_vlm]及有效性/分块mask；π0.5的离散状态已经在语言token里。 |
| [Pi0.embed_suffix](../../../../code/src/openpi/models/pi0.py#L169) | 把带噪动作[B,H,A]投影到专家宽度。π0额外加连续state token并拼接时间embedding；π0.5不加该state token，时间走AdaRMS条件。 |
| [Pi0.compute_loss](../../../../code/src/openpi/models/pi0.py#L228) | 训练：随机采样噪声和时间，把真实动作变为x_t，预测其flow向量场v_t，对目标noise-actions计算MSE。返回每个样本、每个预测时刻的误差。 |
| [Pi0.sample_actions](../../../../code/src/openpi/models/pi0.py#L266) | 推理：固定当前观测，先缓存图像/语言前缀K/V，再从高斯噪声向t=0做多次Euler更新。返回[B,H,A]动作块，执行由外部客户端负责。 |
| [Pi0.sample_actions.step](../../../../code/src/openpi/models/pi0.py#L293) | 一次flow积分：用当前x_t与time构造后缀，读取不变的前缀缓存，预测v_t并更新x_t和time。不是环境里执行了一步机器人动作。 |
| [Pi0.sample_actions.cond](../../../../code/src/openpi/models/pi0.py#L332) | 判断积分是否尚未到达t=0；保留半个步长容差以避免浮点误差多算或少算一步。 |
