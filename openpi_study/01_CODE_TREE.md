# 代码树：先分清五层

<!-- reading-nav-start -->
[首页](../README.md) · [代码学习入口](README.md) · [按问题查找](../related_work/FIND_BY_QUESTION.md)
<!-- reading-nav-end -->

| 层 | 位置 | 负责什么 | 先读程度 |
|---|---|---|---|
| 入口 | `scripts/`、`examples/` | 启动训练、服务、数据转换或评测 | 先看调用关系，不立即运行 |
| 数据与策略 | `src/openpi/policies/`、`transforms.py` | 将平台数据变成统一输入，把模型输出还原成动作 | 第一轮重点 |
| 模型 | `src/openpi/models/` | 图像/语言编码、动作专家、flow loss和采样 | 第二轮重点 |
| 训练工程 | `src/openpi/training/`、`shared/` | 数据迭代、优化、分片、权重与checkpoint | 理解模型后读 |
| 网络与执行 | `serving/`、`packages/openpi-client/` | 观测远程推理、动作块缓存、环境控制循环 | 最后连接完整闭环 |

`models_pytorch/`是π0/π0.5的另一种计算框架实现。先沿较短的JAX `models/pi0.py`理解公式，再用PyTorch版本对照；不要第一遍同时追两套后端。

## 完整文件树

```text
code/
├── .github/
│   ├── workflows/
│   │   ├── pre-commit.yml
│   │   └── test.yml
│   └── CODEOWNERS
├── .vscode/
│   └── settings.json
├── docs/
│   ├── docker.md
│   ├── norm_stats.md
│   └── remote_inference.md
├── examples/
│   ├── aloha_real/
│   │   ├── Dockerfile
│   │   ├── README.md
│   │   ├── compose.yml
│   │   ├── constants.py
│   │   ├── convert_aloha_data_to_lerobot.py
│   │   ├── env.py
│   │   ├── main.py
│   │   ├── real_env.py
│   │   ├── requirements.in
│   │   ├── requirements.txt
│   │   ├── robot_utils.py
│   │   └── video_display.py
│   ├── aloha_sim/
│   │   ├── Dockerfile
│   │   ├── README.md
│   │   ├── compose.yml
│   │   ├── env.py
│   │   ├── main.py
│   │   ├── requirements.in
│   │   ├── requirements.txt
│   │   └── saver.py
│   ├── droid/
│   │   ├── README.md
│   │   ├── README_train.md
│   │   ├── compute_droid_nonidle_ranges.py
│   │   ├── convert_droid_data_to_lerobot.py
│   │   └── main.py
│   ├── libero/
│   │   ├── Dockerfile
│   │   ├── README.md
│   │   ├── compose.yml
│   │   ├── convert_libero_data_to_lerobot.py
│   │   ├── main.py
│   │   ├── requirements.in
│   │   └── requirements.txt
│   ├── simple_client/
│   │   ├── Dockerfile
│   │   ├── README.md
│   │   ├── compose.yml
│   │   ├── main.py
│   │   ├── requirements.in
│   │   └── requirements.txt
│   ├── ur5/
│   │   └── README.md
│   ├── convert_jax_model_to_pytorch.py
│   ├── inference.ipynb
│   └── policy_records.ipynb
├── packages/
│   └── openpi-client/
│       ├── src/
│       │   └── openpi_client/
│       │       ├── runtime/
│       │       │   ├── agents/
│       │       │   │   └── policy_agent.py
│       │       │   ├── agent.py
│       │       │   ├── environment.py
│       │       │   ├── runtime.py
│       │       │   └── subscriber.py
│       │       ├── __init__.py
│       │       ├── action_chunk_broker.py
│       │       ├── base_policy.py
│       │       ├── image_tools.py
│       │       ├── image_tools_test.py
│       │       ├── msgpack_numpy.py
│       │       ├── msgpack_numpy_test.py
│       │       └── websocket_client_policy.py
│       └── pyproject.toml
├── scripts/
│   ├── docker/
│   │   ├── compose.yml
│   │   ├── install_docker_ubuntu22.sh
│   │   ├── install_nvidia_container_toolkit.sh
│   │   └── serve_policy.Dockerfile
│   ├── __init__.py
│   ├── compute_norm_stats.py
│   ├── serve_policy.py
│   ├── train.py
│   ├── train_pytorch.py
│   └── train_test.py
├── src/
│   └── openpi/
│       ├── models/
│       │   ├── utils/
│       │   │   └── fsq_tokenizer.py
│       │   ├── __init__.py
│       │   ├── gemma.py
│       │   ├── gemma_fast.py
│       │   ├── lora.py
│       │   ├── lora_test.py
│       │   ├── model.py
│       │   ├── model_test.py
│       │   ├── pi0.py
│       │   ├── pi0_config.py
│       │   ├── pi0_fast.py
│       │   ├── pi0_test.py
│       │   ├── siglip.py
│       │   ├── tokenizer.py
│       │   ├── tokenizer_test.py
│       │   └── vit.py
│       ├── models_pytorch/
│       │   ├── transformers_replace/
│       │   │   └── models/
│       │   │       ├── gemma/
│       │   │       │   ├── configuration_gemma.py
│       │   │       │   └── modeling_gemma.py
│       │   │       ├── paligemma/
│       │   │       │   └── modeling_paligemma.py
│       │   │       └── siglip/
│       │   │           ├── check.py
│       │   │           └── modeling_siglip.py
│       │   ├── gemma_pytorch.py
│       │   ├── pi0_pytorch.py
│       │   └── preprocessing_pytorch.py
│       ├── policies/
│       │   ├── aloha_policy.py
│       │   ├── droid_policy.py
│       │   ├── libero_policy.py
│       │   ├── policy.py
│       │   ├── policy_config.py
│       │   └── policy_test.py
│       ├── serving/
│       │   └── websocket_policy_server.py
│       ├── shared/
│       │   ├── __init__.py
│       │   ├── array_typing.py
│       │   ├── download.py
│       │   ├── download_test.py
│       │   ├── image_tools.py
│       │   ├── image_tools_test.py
│       │   ├── nnx_utils.py
│       │   ├── normalize.py
│       │   └── normalize_test.py
│       ├── training/
│       │   ├── misc/
│       │   │   ├── polaris_config.py
│       │   │   └── roboarena_config.py
│       │   ├── checkpoints.py
│       │   ├── config.py
│       │   ├── data_loader.py
│       │   ├── data_loader_test.py
│       │   ├── droid_rlds_dataset.py
│       │   ├── optimizer.py
│       │   ├── sharding.py
│       │   ├── utils.py
│       │   └── weight_loaders.py
│       ├── __init__.py
│       ├── conftest.py
│       ├── py.typed
│       ├── transforms.py
│       └── transforms_test.py
├── .dockerignore
├── .gitignore
├── .gitmodules
├── .pre-commit-config.yaml
├── .python-version
├── CONTRIBUTING.md
├── LICENSE
├── LICENSE_GEMMA.txt
├── README.md
├── pyproject.toml
└── uv.lock
```

`third_party/aloha`和`third_party/libero`是外部Git子模块指针，未展开在这个源码快照里；它们的commit记录在[UPSTREAM.json](UPSTREAM.json)。完整运行相应环境时，建议另行clone固定版本上游仓库并初始化其子模块，而非在这个嵌套学习快照中直接执行根仓库的submodule命令。

每个文件具体用途、函数定位和易错点见[逐文件目录](FILE_INDEX.md)。下一步：[小白阅读路线](02_START_HERE.md)。

<!-- reading-footer-start -->
[接着读：跟一次推理](02_START_HERE.md) · [返回代码学习入口](README.md)
<!-- reading-footer-end -->
