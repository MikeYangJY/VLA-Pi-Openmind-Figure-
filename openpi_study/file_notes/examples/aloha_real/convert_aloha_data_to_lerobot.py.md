# `examples/aloha_real/convert_aloha_data_to_lerobot.py` 中文阅读说明

**定位：** ALOHA数据转换。

读原始ALOHA episode，检查相机和状态字段，再建立LeRobot数据集。

**建议读法：** DatasetConfig→create_empty_dataset→load_raw_episode_data→populate_dataset→保存。

**易错点：** 速度和effort字段可能并非每个数据集都有，需通过存在性检查分支处理。

[注释源码](../../../code/examples/aloha_real/convert_aloha_data_to_lerobot.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/aloha_real/convert_aloha_data_to_lerobot.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [DatasetConfig](../../../code/examples/aloha_real/convert_aloha_data_to_lerobot.py#L30) | 读原始ALOHA episode，检查相机和状态字段，再建立LeRobot数据集。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [create_empty_dataset](../../../code/examples/aloha_real/convert_aloha_data_to_lerobot.py#L45) | 本函数位于“ALOHA数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 Path(LEROBOT_HOME / repo_id).exists → Path → shutil.rmtree 追踪具体实现。 |
| [get_cameras](../../../code/examples/aloha_real/convert_aloha_data_to_lerobot.py#L143) | 本函数位于“ALOHA数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 h5py.File → ep['/observations/images'].keys 追踪具体实现。 |
| [has_velocity](../../../code/examples/aloha_real/convert_aloha_data_to_lerobot.py#L152) | 本函数位于“ALOHA数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 h5py.File 追踪具体实现。 |
| [has_effort](../../../code/examples/aloha_real/convert_aloha_data_to_lerobot.py#L160) | 本函数位于“ALOHA数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 h5py.File 追踪具体实现。 |
| [load_raw_images_per_camera](../../../code/examples/aloha_real/convert_aloha_data_to_lerobot.py#L169) | 本函数位于“ALOHA数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 imgs_array.append → cv2.cvtColor → cv2.imdecode 追踪具体实现。 |
| [load_raw_episode_data](../../../code/examples/aloha_real/convert_aloha_data_to_lerobot.py#L194) | 本函数位于“ALOHA数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 h5py.File → torch.from_numpy → load_raw_images_per_camera 追踪具体实现。 |
| [populate_dataset](../../../code/examples/aloha_real/convert_aloha_data_to_lerobot.py#L226) | 本函数位于“ALOHA数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 tqdm.tqdm → load_raw_episode_data → imgs_per_cam.items 追踪具体实现。 |
| [port_aloha](../../../code/examples/aloha_real/convert_aloha_data_to_lerobot.py#L265) | 本函数位于“ALOHA数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 (LEROBOT_HOME / repo_id).exists → shutil.rmtree → raw_dir.exists 追踪具体实现。 |
