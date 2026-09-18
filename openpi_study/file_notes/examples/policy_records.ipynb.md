# `examples/policy_records.ipynb` 中文阅读说明

**定位：** 交互式示例。

按单元格演示推理或输入输出记录读取。

**建议读法：** 从导入/配置→加载或读文件→计算→展示按顺序阅读；加载checkpoint的单元格可能下载大权重。

**易错点：** 格式敏感文件、文档和许可原样保存；中文说明放在本页，避免破坏原格式。

[注释源码](../../code/examples/policy_records.ipynb) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/policy_records.ipynb)

## 逐单元格说明

### 单元格 1（code）

读取先前保存的policy输入输出；先确认记录路径和文件来源。

入口片段：`import pathlib`

### 单元格 2（code）

代码单元：按顺序准备变量、运行调用并查看结果。

入口片段：`print("length of records", len(records))`

### 单元格 3（code）

把已有观测或结果可视化，不会额外训练模型。

入口片段：`from PIL import Image`

### 单元格 4（code）

导入依赖与别名；先认清config、policy和数据处理模块，暂时不需记住全部库。

入口片段：`import pandas as pd`

### 单元格 5（code）

代码单元：按顺序准备变量、运行调用并查看结果。

入口片段：`for name in in_data.columns:`

### 单元格 6（code）

代码单元：按顺序准备变量、运行调用并查看结果。

入口片段：`空单元格`
