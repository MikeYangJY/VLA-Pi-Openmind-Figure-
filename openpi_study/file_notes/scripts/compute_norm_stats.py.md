# `scripts/compute_norm_stats.py` 中文阅读说明

**定位：** 训练前准备。

遍历训练数据统计state/action均值、标准差与分位数，供训练和推理使用。

**建议读法：** 根据配置建立未归一化数据管线，分batch更新RunningStats，再写入assets。

**易错点：** 统计必须基于与训练相同的动作表示；先做delta再统计与先统计绝对动作不同。

[注释源码](../../code/scripts/compute_norm_stats.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/scripts/compute_norm_stats.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [RemoveStrings](../../code/scripts/compute_norm_stats.py#L26) | 遍历训练数据统计state/action均值、标准差与分位数，供训练和推理使用。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [RemoveStrings.__call__](../../code/scripts/compute_norm_stats.py#L31) | 执行遍历训练数据统计state/action均值、标准差与分位数，供训练和推理使用。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。 |
| [create_torch_dataloader](../../code/scripts/compute_norm_stats.py#L39) | 本函数位于“训练前准备”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 ValueError → _data_loader.create_torch_dataset → _data_loader.TransformedDataset 追踪具体实现。 |
| [create_rlds_dataloader](../../code/scripts/compute_norm_stats.py#L79) | 本函数位于“训练前准备”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 _data_loader.create_rlds_dataset → _data_loader.IterableTransformedDataset → RemoveStrings 追踪具体实现。 |
| [main](../../code/scripts/compute_norm_stats.py#L111) | 本脚本入口：遍历训练数据统计state/action均值、标准差与分位数，供训练和推理使用。 根据配置建立未归一化数据管线，分batch更新RunningStats，再写入assets。 |
