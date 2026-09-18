# `src/openpi/models_pytorch/gemma_pytorch.py` 中文阅读说明

**定位：** 共享模型底层。

包装Hugging Face的PaliGemma和Gemma action expert，使两组token共同参与注意力。

**建议读法：** 构造两个模型；分别投影Q/K/V；沿token轴拼接做注意力；按原片段拆回各自分支并计算残差/MLP。

**易错点：** 联合训练前向、仅前缀prefill、仅后缀读cache是三条路径；不要混淆hidden width与head_dim。

[注释源码](../../../../code/src/openpi/models_pytorch/gemma_pytorch.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models_pytorch/gemma_pytorch.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [PaliGemmaWithExpertModel](../../../../code/src/openpi/models_pytorch/gemma_pytorch.py#L18) | 保留VLM与动作专家各自参数，同时实现前缀/后缀交互。 |
| [PaliGemmaWithExpertModel.__init__](../../../../code/src/openpi/models_pytorch/gemma_pytorch.py#L22) | 初始化PaliGemmaWithExpertModel的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.paligemma、self.gemma_expert、self.gemma_expert.model.embed_tokens。 |
| [PaliGemmaWithExpertModel.to_bfloat16_for_selected_params](../../../../code/src/openpi/models_pytorch/gemma_pytorch.py#L76) | 将多数权重用bfloat16保存/计算，同时让指定视觉和归一化参数保持float32，以兼顾内存和数值精度。 |
| [PaliGemmaWithExpertModel.embed_image](../../../../code/src/openpi/models_pytorch/gemma_pytorch.py#L101) | 调用PaliGemma视觉塔/投影，得到能进入语言模型维度空间的图像token。 |
| [PaliGemmaWithExpertModel.embed_language_tokens](../../../../code/src/openpi/models_pytorch/gemma_pytorch.py#L107) | 把整数token ID查表为语言embedding；整数编号本身不是语义向量。 |
| [PaliGemmaWithExpertModel.forward](../../../../code/src/openpi/models_pytorch/gemma_pytorch.py#L114) | 分三种路径：仅前缀建立缓存、仅后缀读取缓存、前后缀联合训练。联合路径用各自参数算Q/K/V，合并注意力后再拆回。 |
| [PaliGemmaWithExpertModel.forward.compute_layer_complete](../../../../code/src/openpi/models_pytorch/gemma_pytorch.py#L185) | 单层联合计算：分别归一化和投影→拼接Q/K/V→RoPE与attention→按token长度拆回→各分支残差/MLP。 |
| [PaliGemmaWithExpertModel.forward.compute_final_norms](../../../../code/src/openpi/models_pytorch/gemma_pytorch.py#L296) | 本函数位于“共享模型底层”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 models[i].norm → outputs_embeds.append 追踪具体实现。 |
