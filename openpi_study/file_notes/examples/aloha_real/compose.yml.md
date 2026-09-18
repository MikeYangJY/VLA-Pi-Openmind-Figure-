# `examples/aloha_real/compose.yml` 中文阅读说明

**定位：** 容器编排。

连接模型服务、客户端、挂载目录、端口和GPU资源。

**建议读法：** 先看services，再看volumes、environment、ports和启动命令；路径相对compose文件解析。

**易错点：** 格式敏感文件、文档和许可原样保存；中文说明放在本页，避免破坏原格式。

[注释源码](../../../code/examples/aloha_real/compose.yml) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/aloha_real/compose.yml)

## 配置怎么影响执行

配置只在被相应工具读取时生效。先定位调用它的入口，再核对路径、版本、资源与开关；不要把安装/容器参数当成模型学习出的参数。
