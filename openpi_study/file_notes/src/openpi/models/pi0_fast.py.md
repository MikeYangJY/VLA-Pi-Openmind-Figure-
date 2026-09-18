# `src/openpi/models/pi0_fast.py` 中文阅读说明

**定位：** 可跳过。

π0-FAST自回归动作分支；保留上游文件以维持完整依赖结构。

**建议读法：** 学习π0/π0.5连续动作主线时可跳过本文件。

**易错点：** 本文件的token交叉熵和AR采样不能当成π0/π0.5的flow采样。

[注释源码](../../../../code/src/openpi/models/pi0_fast.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0_fast.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [make_attn_mask](../../../../code/src/openpi/models/pi0_fast.py#L30) | 此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。 |
| [left_to_right_align](../../../../code/src/openpi/models/pi0_fast.py#L60) | 此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。 |
| [put_along_last_axis](../../../../code/src/openpi/models/pi0_fast.py#L76) | 此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。 |
| [Pi0FASTConfig](../../../../code/src/openpi/models/pi0_fast.py#L87) | 此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。 |
| [Pi0FASTConfig.model_type](../../../../code/src/openpi/models/pi0_fast.py#L104) | 此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。 |
| [Pi0FASTConfig.create](../../../../code/src/openpi/models/pi0_fast.py#L109) | 此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。 |
| [Pi0FASTConfig.inputs_spec](../../../../code/src/openpi/models/pi0_fast.py#L114) | 此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。 |
| [Pi0FASTConfig.get_freeze_filter](../../../../code/src/openpi/models/pi0_fast.py#L141) | 此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。 |
| [Pi0FAST](../../../../code/src/openpi/models/pi0_fast.py#L149) | 此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。 |
| [Pi0FAST.__init__](../../../../code/src/openpi/models/pi0_fast.py#L151) | 此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。 |
| [Pi0FAST.embed_inputs](../../../../code/src/openpi/models/pi0_fast.py#L177) | 此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。 |
| [Pi0FAST.compute_loss](../../../../code/src/openpi/models/pi0_fast.py#L216) | 此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。 |
| [Pi0FAST.sample_actions](../../../../code/src/openpi/models/pi0_fast.py#L255) | 此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。 |
| [Pi0FAST.sample_actions.step](../../../../code/src/openpi/models/pi0_fast.py#L293) | 此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。 |
| [Pi0FAST.sample_actions.cond](../../../../code/src/openpi/models/pi0_fast.py#L326) | 此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。 |
