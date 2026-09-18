# `.github/workflows/test.yml` 中文阅读说明

**定位：** 上游CI。

在GitHub Actions运行上游测试。

**建议读法：** 看测试设备/依赖与测试范围；本次仅做注释静态一致性验证，不冒称已运行这套CI。

**易错点：** 格式敏感文件、文档和许可原样保存；中文说明放在本页，避免破坏原格式。

[注释源码](../../../code/.github/workflows/test.yml) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/.github/workflows/test.yml)

## 配置怎么影响执行

配置只在被相应工具读取时生效。先定位调用它的入口，再核对路径、版本、资源与开关；不要把安装/容器参数当成模型学习出的参数。
