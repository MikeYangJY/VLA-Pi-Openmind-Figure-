# `examples/inference.ipynb` 中文阅读说明

**定位：** 交互式示例。

按单元格演示推理或输入输出记录读取。

**建议读法：** 从导入/配置→加载或读文件→计算→展示按顺序阅读；加载checkpoint的单元格可能下载大权重。

**易错点：** 格式敏感文件、文档和许可原样保存；中文说明放在本页，避免破坏原格式。

[注释源码](../../code/examples/inference.ipynb) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/inference.ipynb)

## 逐单元格说明

### 单元格 1（code）

导入依赖与别名；先认清config、policy和数据处理模块，暂时不需记住全部库。

入口片段：`import dataclasses`

### 单元格 2（markdown）

说明文字：先理解它声明的前提，再读下一段代码。

入口片段：`# Policy inference`

### 单元格 3（code）

调用策略，得到一整段actions；检查返回shape，环境执行属于后续步骤。

入口片段：`config = _config.get_config("pi0_fast_droid")`

### 单元格 4（markdown）

说明文字：先理解它声明的前提，再读下一段代码。

入口片段：`# Working with a live model`

### 单元格 5（code）

根据配置定位并加载checkpoint，可能触发网络与大文件下载；阅读不需要执行。

入口片段：`config = _config.get_config("pi0_aloha_sim")`

### 单元格 6（markdown）

说明文字：先理解它声明的前提，再读下一段代码。

入口片段：`Now, we are going to create a data loader and use a real batch of training data to compute the loss.`

### 单元格 7（code）

代码单元：按顺序准备变量、运行调用并查看结果。

入口片段：`# Reduce the batch size to reduce memory usage.`
