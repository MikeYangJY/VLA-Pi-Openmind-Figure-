# `src/openpi/models/gemma_fast.py` 中文阅读说明

**定位：** 可跳过。

服务FAST分支的Gemma实现和自回归KV缓存。

**建议读法：** 当前π0/π0.5学习路线不进入本文件。

**易错点：** 不要把它的逐token缓存更新直接套到flow重复去噪。

[注释源码](../../../../code/src/openpi/models/gemma_fast.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/gemma_fast.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [get_config](../../../../code/src/openpi/models/gemma_fast.py#L42) | 此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。 |
| [Einsum](../../../../code/src/openpi/models/gemma_fast.py#L85) | 此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。 |
| [Einsum.__call__](../../../../code/src/openpi/models/gemma_fast.py#L90) | 此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。 |
| [RMSNorm](../../../../code/src/openpi/models/gemma_fast.py#L98) | 此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。 |
| [RMSNorm.__call__](../../../../code/src/openpi/models/gemma_fast.py#L101) | 此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。 |
| [Embedder](../../../../code/src/openpi/models/gemma_fast.py#L114) | 此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。 |
| [Embedder.setup](../../../../code/src/openpi/models/gemma_fast.py#L121) | 此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。 |
| [Embedder.encode](../../../../code/src/openpi/models/gemma_fast.py#L129) | 此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。 |
| [Embedder.decode](../../../../code/src/openpi/models/gemma_fast.py#L135) | 此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。 |
| [Attention](../../../../code/src/openpi/models/gemma_fast.py#L141) | 此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。 |
| [Attention.setup](../../../../code/src/openpi/models/gemma_fast.py#L154) | 此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。 |
| [Attention._init_cache](../../../../code/src/openpi/models/gemma_fast.py#L183) | 此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。 |
| [Attention._update_cache](../../../../code/src/openpi/models/gemma_fast.py#L194) | 此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。 |
| [Attention.__call__](../../../../code/src/openpi/models/gemma_fast.py#L206) | 此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。 |
| [Block](../../../../code/src/openpi/models/gemma_fast.py#L249) | 此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。 |
| [Block.setup](../../../../code/src/openpi/models/gemma_fast.py#L264) | 此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。 |
| [Block.__call__](../../../../code/src/openpi/models/gemma_fast.py#L284) | 此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。 |
| [Module](../../../../code/src/openpi/models/gemma_fast.py#L303) | 此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。 |
| [Module.__call__](../../../../code/src/openpi/models/gemma_fast.py#L328) | 此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。 |
| [Module.init](../../../../code/src/openpi/models/gemma_fast.py#L446) | 此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。 |
| [_apply_rope](../../../../code/src/openpi/models/gemma_fast.py#L452) | 此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。 |
