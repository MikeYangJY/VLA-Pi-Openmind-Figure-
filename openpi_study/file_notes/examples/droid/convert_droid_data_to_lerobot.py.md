# `examples/droid/convert_droid_data_to_lerobot.py` 中文阅读说明

**定位：** DROID数据转换。

读取HDF5轨迹和多相机视频，按时间步对齐后写入LeRobot。

**建议读法：** TrajectoryReader读状态；RecordedMultiCameraWrapper/MP4Reader读视频；main把同步样本写成episode。

**易错点：** 相机时间索引错位会让图像和动作监督不匹配，不能只看最终文件是否生成。

[注释源码](../../../code/examples/droid/convert_droid_data_to_lerobot.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/droid/convert_droid_data_to_lerobot.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [resize_image](../../../code/examples/droid/convert_droid_data_to_lerobot.py#L42) | 本函数位于“DROID数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 Image.fromarray → np.array → image.resize 追踪具体实现。 |
| [main](../../../code/examples/droid/convert_droid_data_to_lerobot.py#L50) | 本脚本入口：读取HDF5轨迹和多相机视频，按时间步对齐后写入LeRobot。 TrajectoryReader读状态；RecordedMultiCameraWrapper/MP4Reader读视频；main把同步样本写成episode。 |
| [get_camera_type](../../../code/examples/droid/convert_droid_data_to_lerobot.py#L196) | 本函数位于“DROID数据转换”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。 |
| [MP4Reader](../../../code/examples/droid/convert_droid_data_to_lerobot.py#L204) | 读取HDF5轨迹和多相机视频，按时间步对齐后写入LeRobot。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [MP4Reader.__init__](../../../code/examples/droid/convert_droid_data_to_lerobot.py#L208) | 初始化MP4Reader的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.serial_number、self._index、self._mp4_reader。 |
| [MP4Reader.set_reading_parameters](../../../code/examples/droid/convert_droid_data_to_lerobot.py#L220) | 本函数位于“DROID数据转换”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。 |
| [MP4Reader.get_frame_resolution](../../../code/examples/droid/convert_droid_data_to_lerobot.py#L238) | 本函数位于“DROID数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self._mp4_reader.get 追踪具体实现。 |
| [MP4Reader.get_frame_count](../../../code/examples/droid/convert_droid_data_to_lerobot.py#L245) | 本函数位于“DROID数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self._mp4_reader.get 追踪具体实现。 |
| [MP4Reader.set_frame_index](../../../code/examples/droid/convert_droid_data_to_lerobot.py#L253) | 本函数位于“DROID数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self._mp4_reader.set → self.read_camera 追踪具体实现。 |
| [MP4Reader._process_frame](../../../code/examples/droid/convert_droid_data_to_lerobot.py#L268) | 本函数位于“DROID数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 copy.deepcopy → self.resize_func 追踪具体实现。 |
| [MP4Reader.read_camera](../../../code/examples/droid/convert_droid_data_to_lerobot.py#L278) | 本函数位于“DROID数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self._mp4_reader.read → self._process_frame 追踪具体实现。 |
| [MP4Reader.disable_camera](../../../code/examples/droid/convert_droid_data_to_lerobot.py#L308) | 本函数位于“DROID数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 hasattr → self._mp4_reader.release 追踪具体实现。 |
| [RecordedMultiCameraWrapper](../../../code/examples/droid/convert_droid_data_to_lerobot.py#L314) | 读取HDF5轨迹和多相机视频，按时间步对齐后写入LeRobot。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [RecordedMultiCameraWrapper.__init__](../../../code/examples/droid/convert_droid_data_to_lerobot.py#L318) | 初始化RecordedMultiCameraWrapper的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.camera_kwargs、self.camera_dict、self.camera_dict[serial_number]。 |
| [RecordedMultiCameraWrapper.read_cameras](../../../code/examples/droid/convert_droid_data_to_lerobot.py#L343) | 本函数位于“DROID数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 defaultdict → self.camera_dict.keys → print 追踪具体实现。 |
| [get_hdf5_length](../../../code/examples/droid/convert_droid_data_to_lerobot.py#L380) | 本函数位于“DROID数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 isinstance → get_hdf5_length 追踪具体实现。 |
| [load_hdf5_to_dict](../../../code/examples/droid/convert_droid_data_to_lerobot.py#L406) | 本函数位于“DROID数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 isinstance → load_hdf5_to_dict 追踪具体实现。 |
| [TrajectoryReader](../../../code/examples/droid/convert_droid_data_to_lerobot.py#L425) | 读取HDF5轨迹和多相机视频，按时间步对齐后写入LeRobot。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [TrajectoryReader.__init__](../../../code/examples/droid/convert_droid_data_to_lerobot.py#L429) | 初始化TrajectoryReader的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._hdf5_file、self._read_images、self._length、self._video_readers、self._index。 |
| [TrajectoryReader.length](../../../code/examples/droid/convert_droid_data_to_lerobot.py#L439) | 本函数位于“DROID数据转换”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。 |
| [TrajectoryReader.read_timestep](../../../code/examples/droid/convert_droid_data_to_lerobot.py#L446) | 本函数位于“DROID数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 keys_to_ignore.copy → load_hdf5_to_dict 追踪具体实现。 |
| [TrajectoryReader.close](../../../code/examples/droid/convert_droid_data_to_lerobot.py#L466) | 本函数位于“DROID数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self._hdf5_file.close 追踪具体实现。 |
| [load_trajectory](../../../code/examples/droid/convert_droid_data_to_lerobot.py#L474) | 本函数位于“DROID数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 TrajectoryReader → RecordedMultiCameraWrapper → traj_reader.length 追踪具体实现。 |
