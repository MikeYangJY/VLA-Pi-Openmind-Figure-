# `pyproject.toml` 中文阅读说明

**定位：** 项目与依赖配置。

定义Python包、依赖、可选组、构建和开发工具配置。

**建议读法：** 从project.dependencies到dependency-groups；理解JAX/PyTorch与rlds分别需要什么。

**易错点：** 格式敏感文件、文档和许可原样保存；中文说明放在本页，避免破坏原格式。

[注释源码](../code/pyproject.toml) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/pyproject.toml)

## 配置怎么影响执行

配置只在被相应工具读取时生效。先定位调用它的入口，再核对路径、版本、资源与开关；不要把安装/容器参数当成模型学习出的参数。
