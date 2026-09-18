# 验证说明

核对日期：2026-09-18。上游固定版本：`215abfb217dbac7d5f1273282331b9b1866c0479`。

| 检查 | 结果 |
|---|---|
| 上游源码身份 | 140 个原始文件的 Git blob SHA 与官方固定版本树一致 |
| Python 程序结构 | 91 个文件的 AST（不计源码行号）与原版一致 |
| Python 可执行文本 | 91 个文件的非注释 token 序列与原版一致，保留原英文 docstring |
| Python 语法 | 91 个文件均通过编译检查，无需导入模型或下载权重 |
| 其他上游文件 | 49 个配置、文档、notebook、许可等文件逐字节一致 |
| 逐文件说明 | 140 个仓库文件均有独立中文说明和源码链接 |
| 中文注释 | 新增 2,876 行；覆盖文件定位、函数入口、调用线索和关键计算 |

这些检查验证注释没有改变程序结构和可执行 token，不能证明所有文字解释绝无错误，也不能替代模型数值测试。注释会改变源码行号，错误栈位置与上游不同。

**未执行：** GPU 训练、模型权重推理、仿真评测或真机部署。本次没有改可执行代码，也没有宣称复现论文实验。

**范围：** 140 个文件指 openpi 仓库本身；第三方 ALOHA/LIBERO Git 子模块只保留来源和固定 commit 记录。FAST 与其他旁支保留但标记为可跳过。配置和非 Python 文件的说明放在 `file_notes/`，不向 JSON、锁文件或许可证里插入注释。

**文档链接：** 本库新增与维护的说明页已检查相对路径和源码行号。原样保留的上游 DROID/UR5 文档有 6 处相对链接问题，正确入口已补在 [DROID 文档说明](file_notes/examples/droid/README_train.md.md) 和 [UR5 文档说明](file_notes/examples/ur5/README.md.md)。

逐文件内容哈希、新增注释数、符号数见 [ANNOTATION_MANIFEST.json](ANNOTATION_MANIFEST.json)；来源及子模块信息见 [UPSTREAM.json](UPSTREAM.json)。
