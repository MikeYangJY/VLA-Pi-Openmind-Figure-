# `src/openpi/models/tokenizer.py` 中文阅读说明

**定位：** 文本与状态入口。

π0/π0.5主线只需PaligemmaTokenizer：把指令及可选离散状态转成token和padding mask。

**建议读法：** 看tokenize的state分支：π0只编码指令，π0.5可把归一化状态分箱后写进文本。其余tokenizer可跳过。

**易错点：** 离散化state不是离散化输出action；π0.5在本仓库仍输出连续动作。

[注释源码](../../../../code/src/openpi/models/tokenizer.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/tokenizer.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [PaligemmaTokenizer](../../../../code/src/openpi/models/tokenizer.py#L21) | π0/π0.5主线只需PaligemmaTokenizer：把指令及可选离散状态转成token和padding mask。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [PaligemmaTokenizer.__init__](../../../../code/src/openpi/models/tokenizer.py#L25) | 初始化PaligemmaTokenizer的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._max_len、self._tokenizer。 |
| [PaligemmaTokenizer.tokenize](../../../../code/src/openpi/models/tokenizer.py#L36) | 清理任务文本；有state时将其按归一化区间分成256箱并加入Task/State/Action格式，最后截断或补齐并生成token mask。 |
| [FASTTokenizer](../../../../code/src/openpi/models/tokenizer.py#L68) | π0/π0.5主线只需PaligemmaTokenizer：把指令及可选离散状态转成token和padding mask。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [FASTTokenizer.__init__](../../../../code/src/openpi/models/tokenizer.py#L72) | 初始化FASTTokenizer的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._max_len、self._paligemma_tokenizer、self._fast_tokenizer、self._fast_skip_tokens。 |
| [FASTTokenizer.tokenize](../../../../code/src/openpi/models/tokenizer.py#L88) | 本函数位于“文本与状态入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 prompt.lower().strip().replace → prompt.lower().strip → prompt.lower 追踪具体实现。 |
| [FASTTokenizer.extract_actions](../../../../code/src/openpi/models/tokenizer.py#L148) | 本函数位于“文本与状态入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self._paligemma_tokenizer.decode → tokens.tolist → np.zeros 追踪具体实现。 |
| [FASTTokenizer._act_tokens_to_paligemma_tokens](../../../../code/src/openpi/models/tokenizer.py#L169) | 本函数位于“文本与状态入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 isinstance → np.array → self._paligemma_tokenizer.vocab_size 追踪具体实现。 |
| [BinningTokenizer](../../../../code/src/openpi/models/tokenizer.py#L182) | π0/π0.5主线只需PaligemmaTokenizer：把指令及可选离散状态转成token和padding mask。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [BinningTokenizer.__init__](../../../../code/src/openpi/models/tokenizer.py#L190) | 初始化BinningTokenizer的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._max_len、self._n_bins、self._paligemma_tokenizer、self._fast_skip_tokens。 |
| [BinningTokenizer.tokenize](../../../../code/src/openpi/models/tokenizer.py#L205) | 本函数位于“文本与状态入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 prompt.lower().strip().replace → prompt.lower().strip → prompt.lower 追踪具体实现。 |
| [BinningTokenizer.extract_actions](../../../../code/src/openpi/models/tokenizer.py#L268) | 本函数位于“文本与状态入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self._paligemma_tokenizer.decode → tokens.tolist → np.zeros 追踪具体实现。 |
| [BinningTokenizer._act_tokens_to_paligemma_tokens](../../../../code/src/openpi/models/tokenizer.py#L290) | 本函数位于“文本与状态入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 isinstance → np.array → self._paligemma_tokenizer.vocab_size 追踪具体实现。 |
| [FSQTokenizer](../../../../code/src/openpi/models/tokenizer.py#L297) | π0/π0.5主线只需PaligemmaTokenizer：把指令及可选离散状态转成token和padding mask。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [FSQTokenizer.__init__](../../../../code/src/openpi/models/tokenizer.py#L305) | 初始化FSQTokenizer的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._max_len、self._params、self._fsq_tokenizer、self._tokenize_fn、self._detokenize_fn。 |
| [FSQTokenizer.tokenize](../../../../code/src/openpi/models/tokenizer.py#L358) | 本函数位于“文本与状态入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 prompt.lower().strip().replace → prompt.lower().strip → prompt.lower 追踪具体实现。 |
| [FSQTokenizer.extract_actions](../../../../code/src/openpi/models/tokenizer.py#L408) | 本函数位于“文本与状态入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self._paligemma_tokenizer.decode → tokens.tolist → np.zeros 追踪具体实现。 |
| [FSQTokenizer._act_tokens_to_paligemma_tokens](../../../../code/src/openpi/models/tokenizer.py#L435) | 本函数位于“文本与状态入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 isinstance → np.array → self._paligemma_tokenizer.vocab_size 追踪具体实现。 |
