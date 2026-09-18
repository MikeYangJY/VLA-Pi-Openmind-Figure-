# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：DROID数据转换｜读取HDF5轨迹和多相机视频，按时间步对齐后写入LeRobot。
# 阅读顺序：TrajectoryReader读状态；RecordedMultiCameraWrapper/MP4Reader读视频；main把同步样本写成episode。
# 重点边界：相机时间索引错位会让图像和动作监督不匹配，不能只看最终文件是否生成。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
"""
Minimal example script for converting a dataset collected on the DROID platform to LeRobot format.

Usage:
uv run examples/droid/convert_droid_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/droid/convert_droid_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

The resulting dataset will get saved to the $LEROBOT_HOME directory.
"""

from collections import defaultdict
import copy
import glob
import json
from pathlib import Path
import shutil

import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
from tqdm import tqdm
import tyro

REPO_NAME = "your_hf_username/my_droid_dataset"  # Name of the output dataset, also used for the Hugging Face Hub


# 【resize_image】本函数位于“DROID数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 Image.fromarray → np.array → image.resize 追踪具体实现。
# 输入接口：image；size。
# 返回值可从这里追踪：np.array(image.resize(size, resample=Image.BICUBIC))。
# 内部调用线索：Image.fromarray → np.array → image.resize（含分支中的调用，实际路径由条件决定）。
def resize_image(image, size):
    image = Image.fromarray(image)
    return np.array(image.resize(size, resample=Image.BICUBIC))


# 【main】本脚本入口：读取HDF5轨迹和多相机视频，按时间步对齐后写入LeRobot。 TrajectoryReader读状态；RecordedMultiCameraWrapper/MP4Reader读视频；main把同步样本写成episode。
# 输入接口：data_dir:str；push_to_hub:bool。
# 内部调用线索：output_path.exists → shutil.rmtree → Path → LeRobotDataset.create → (data_dir / 'aggregated-annotations-030724.json').open（含分支中的调用，实际路径由条件决定）。
def main(data_dir: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)
    data_dir = Path(data_dir)

    # Create LeRobot dataset, define features to store
    # We will follow the DROID data naming conventions here.
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=15,  # DROID data is typically recorded at 15fps
        features={
            # We call this "left" since we will only use the left stereo camera (following DROID RLDS convention)
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (180, 320, 3),  # This is the resolution used in the DROID RLDS dataset
                "names": ["height", "width", "channel"],
            },
            "exterior_image_2_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_position"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),  # We will use joint *velocity* actions here (7D) + gripper position (1D)
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Load language annotations
    # Note: we load the DROID language annotations for this example, but you can manually define them for your own data
    with (data_dir / "aggregated-annotations-030724.json").open() as f:
        language_annotations = json.load(f)

    # Loop over raw DROID fine-tuning datasets and write episodes to the LeRobot dataset
    # We assume the following directory structure:
    # RAW_DROID_PATH/
    #   - <...>/
    #     - recordings/
    #        - MP4/
    #          - <camera_id>.mp4  # single-view video of left stereo pair camera
    #     - trajectory.hdf5
    #   - <...>/
    episode_paths = list(data_dir.glob("**/trajectory.h5"))
    print(f"Found {len(episode_paths)} episodes for conversion")

    # We will loop over each dataset_name and write episodes to the LeRobot dataset
    for episode_path in tqdm(episode_paths, desc="Converting episodes"):
        # Load raw data
        recording_folderpath = episode_path.parent / "recordings" / "MP4"
        trajectory = load_trajectory(str(episode_path), recording_folderpath=str(recording_folderpath))

        # To load the language instruction, we need to parse out the episode_id from the metadata file
        # Again, you can modify this step for your own data, to load your own language instructions
        metadata_filepath = next(iter(episode_path.parent.glob("metadata_*.json")))
        episode_id = metadata_filepath.name.split(".")[0].split("_")[-1]
        language_instruction = language_annotations.get(episode_id, {"language_instruction1": "Do something"})[
            "language_instruction1"
        ]
        print(f"Converting episode with language instruction: {language_instruction}")

        # Write to LeRobot dataset
        for step in trajectory:
            camera_type_dict = step["observation"]["camera_type"]
            wrist_ids = [k for k, v in camera_type_dict.items() if v == 0]
            exterior_ids = [k for k, v in camera_type_dict.items() if v != 0]
            dataset.add_frame(
                {
                    # Note: need to flip BGR --> RGB for loaded images
                    "exterior_image_1_left": resize_image(
                        step["observation"]["image"][exterior_ids[0]][..., ::-1], (320, 180)
                    ),
                    "exterior_image_2_left": resize_image(
                        step["observation"]["image"][exterior_ids[1]][..., ::-1], (320, 180)
                    ),
                    "wrist_image_left": resize_image(step["observation"]["image"][wrist_ids[0]][..., ::-1], (320, 180)),
                    "joint_position": np.asarray(
                        step["observation"]["robot_state"]["joint_positions"], dtype=np.float32
                    ),
                    "gripper_position": np.asarray(
                        step["observation"]["robot_state"]["gripper_position"][None], dtype=np.float32
                    ),
                    # Important: we use joint velocity actions here since pi05-droid was pre-trained on joint velocity actions
                    "actions": np.concatenate(
                        [step["action"]["joint_velocity"], step["action"]["gripper_position"][None]], dtype=np.float32
                    ),
                    "task": language_instruction,
                }
            )
        dataset.save_episode()

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


##########################################################################################################
################ The rest of this file are functions to parse the raw DROID data #########################
################ You don't need to worry about understanding this part           #########################
################ It was copied from here: https://github.com/JonathanYang0127/r2d2_rlds_dataset_builder/blob/parallel_convert/r2_d2/r2_d2.py
##########################################################################################################


camera_type_dict = {
    "hand_camera_id": 0,
    "varied_camera_1_id": 1,
    "varied_camera_2_id": 1,
}

camera_type_to_string_dict = {
    0: "hand_camera",
    1: "varied_camera",
    2: "fixed_camera",
}


# 【get_camera_type】本函数位于“DROID数据转换”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。
# 输入接口：cam_id。
# 返回值可从这里追踪：None / camera_type_to_string_dict[type_int]。
def get_camera_type(cam_id):
    if cam_id not in camera_type_dict:
        return None
    type_int = camera_type_dict[cam_id]
    return camera_type_to_string_dict[type_int]


# 【MP4Reader】读取HDF5轨迹和多相机视频，按时间步对齐后写入LeRobot。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
class MP4Reader:
    # 【MP4Reader.__init__】初始化MP4Reader的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.serial_number、self._index、self._mp4_reader。
    # 输入接口：filepath；serial_number。
    # 内部调用线索：cv2.VideoCapture → self._mp4_reader.isOpened → RuntimeError（含分支中的调用，实际路径由条件决定）。
    def __init__(self, filepath, serial_number):
        # Save Parameters #
        self.serial_number = serial_number
        self._index = 0

        # Open Video Reader #
        self._mp4_reader = cv2.VideoCapture(filepath)
        if not self._mp4_reader.isOpened():
            raise RuntimeError("Corrupted MP4 File")

    # 【MP4Reader.set_reading_parameters】本函数位于“DROID数据转换”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。
    # 输入接口：image；concatenate_images；resolution；resize_func。
    def set_reading_parameters(
        self,
        image=True,  # noqa: FBT002
        concatenate_images=False,  # noqa: FBT002
        resolution=(0, 0),
        resize_func=None,
    ):
        # Save Parameters #
        self.image = image
        self.concatenate_images = concatenate_images
        self.resolution = resolution
        self.resize_func = cv2.resize
        self.skip_reading = not image
        if self.skip_reading:
            return

    # 【MP4Reader.get_frame_resolution】本函数位于“DROID数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self._mp4_reader.get 追踪具体实现。
    # 返回值可从这里追踪：(width, height)。
    def get_frame_resolution(self):
        width = self._mp4_reader.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        height = self._mp4_reader.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        return (width, height)

    # 【MP4Reader.get_frame_count】本函数位于“DROID数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self._mp4_reader.get 追踪具体实现。
    # 返回值可从这里追踪：0 / int(self._mp4_reader.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))。
    def get_frame_count(self):
        if self.skip_reading:
            return 0
        return int(self._mp4_reader.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    # 【MP4Reader.set_frame_index】本函数位于“DROID数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self._mp4_reader.set → self.read_camera 追踪具体实现。
    # 输入接口：index。
    # 内部调用线索：self._mp4_reader.set → self.read_camera（含分支中的调用，实际路径由条件决定）。
    def set_frame_index(self, index):
        if self.skip_reading:
            return

        if index < self._index:
            self._mp4_reader.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
            self._index = index

        while self._index < index:
            self.read_camera(ignore_data=True)

    # 【MP4Reader._process_frame】本函数位于“DROID数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 copy.deepcopy → self.resize_func 追踪具体实现。
    # 输入接口：frame。
    # 返回值可从这里追踪：frame / self.resize_func(frame, self.resolution)。
    # 内部调用线索：copy.deepcopy → self.resize_func（含分支中的调用，实际路径由条件决定）。
    def _process_frame(self, frame):
        frame = copy.deepcopy(frame)
        if self.resolution == (0, 0):
            return frame
        return self.resize_func(frame, self.resolution)

    # 【MP4Reader.read_camera】本函数位于“DROID数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self._mp4_reader.read → self._process_frame 追踪具体实现。
    # 输入接口：ignore_data；correct_timestamp。
    # 返回值可从这里追踪：{} / None。
    # 内部调用线索：self._mp4_reader.read → self._process_frame（含分支中的调用，实际路径由条件决定）。
    def read_camera(self, ignore_data=False, correct_timestamp=None):  # noqa: FBT002
        # Skip if Read Unnecessary #
        if self.skip_reading:
            return {}

        # Read Camera #
        success, frame = self._mp4_reader.read()

        self._index += 1
        if not success:
            return None
        if ignore_data:
            return None

        # Return Data #
        data_dict = {}

        if self.concatenate_images or "stereo" not in self.serial_number:
            data_dict["image"] = {self.serial_number: self._process_frame(frame)}
        else:
            single_width = frame.shape[1] // 2
            data_dict["image"] = {
                self.serial_number + "_left": self._process_frame(frame[:, :single_width, :]),
                self.serial_number + "_right": self._process_frame(frame[:, single_width:, :]),
            }

        return data_dict

    # 【MP4Reader.disable_camera】本函数位于“DROID数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 hasattr → self._mp4_reader.release 追踪具体实现。
    # 内部调用线索：hasattr → self._mp4_reader.release（含分支中的调用，实际路径由条件决定）。
    def disable_camera(self):
        if hasattr(self, "_mp4_reader"):
            self._mp4_reader.release()


# 【RecordedMultiCameraWrapper】读取HDF5轨迹和多相机视频，按时间步对齐后写入LeRobot。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
class RecordedMultiCameraWrapper:
    # 【RecordedMultiCameraWrapper.__init__】初始化RecordedMultiCameraWrapper的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.camera_kwargs、self.camera_dict、self.camera_dict[serial_number]。
    # 输入接口：recording_folderpath；camera_kwargs。
    # 内部调用线索：glob.glob → f.split → get_camera_type → camera_kwargs.get → f.endswith（含分支中的调用，实际路径由条件决定）。
    def __init__(self, recording_folderpath, camera_kwargs={}):  # noqa: B006
        # Save Camera Info #
        self.camera_kwargs = camera_kwargs

        # Open Camera Readers #
        mp4_filepaths = glob.glob(recording_folderpath + "/*.mp4")
        all_filepaths = mp4_filepaths

        self.camera_dict = {}
        for f in all_filepaths:
            serial_number = f.split("/")[-1][:-4]
            cam_type = get_camera_type(serial_number)
            camera_kwargs.get(cam_type, {})

            if f.endswith(".mp4"):
                Reader = MP4Reader  # noqa: N806
            else:
                raise ValueError

            self.camera_dict[serial_number] = Reader(f, serial_number)

    # 【RecordedMultiCameraWrapper.read_cameras】本函数位于“DROID数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 defaultdict → self.camera_dict.keys → print 追踪具体实现。
    # 输入接口：index；camera_type_dict；timestamp_dict。
    # 返回值可从这里追踪：None / full_obs_dict。
    # 内部调用线索：defaultdict → self.camera_dict.keys → print → ValueError → self.camera_kwargs.get（含分支中的调用，实际路径由条件决定）。
    def read_cameras(self, index=None, camera_type_dict={}, timestamp_dict={}):  # noqa: B006
        full_obs_dict = defaultdict(dict)

        # Read Cameras In Randomized Order #
        all_cam_ids = list(self.camera_dict.keys())
        # random.shuffle(all_cam_ids)

        for cam_id in all_cam_ids:
            if "stereo" in cam_id:
                continue
            try:
                cam_type = camera_type_dict[cam_id]
            except KeyError:
                print(f"{self.camera_dict} -- {camera_type_dict}")
                raise ValueError(f"Camera type {cam_id} not found in camera_type_dict")  # noqa: B904
            curr_cam_kwargs = self.camera_kwargs.get(cam_type, {})
            self.camera_dict[cam_id].set_reading_parameters(**curr_cam_kwargs)

            timestamp = timestamp_dict.get(cam_id + "_frame_received", None)
            if index is not None:
                self.camera_dict[cam_id].set_frame_index(index)

            data_dict = self.camera_dict[cam_id].read_camera(correct_timestamp=timestamp)

            # Process Returned Data #
            if data_dict is None:
                return None
            for key in data_dict:
                full_obs_dict[key].update(data_dict[key])

        return full_obs_dict


# 【get_hdf5_length】本函数位于“DROID数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 isinstance → get_hdf5_length 追踪具体实现。
# 输入接口：hdf5_file；keys_to_ignore。
# 返回值可从这里追踪：length。
# 内部调用线索：isinstance → get_hdf5_length（含分支中的调用，实际路径由条件决定）。
def get_hdf5_length(hdf5_file, keys_to_ignore=[]):  # noqa: B006
    length = None

    for key in hdf5_file:
        if key in keys_to_ignore:
            continue

        curr_data = hdf5_file[key]
        if isinstance(curr_data, h5py.Group):
            curr_length = get_hdf5_length(curr_data, keys_to_ignore=keys_to_ignore)
        elif isinstance(curr_data, h5py.Dataset):
            curr_length = len(curr_data)
        else:
            raise ValueError

        if length is None:
            length = curr_length
        assert curr_length == length

    return length


# 【load_hdf5_to_dict】本函数位于“DROID数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 isinstance → load_hdf5_to_dict 追踪具体实现。
# 输入接口：hdf5_file；index；keys_to_ignore。
# 返回值可从这里追踪：data_dict。
# 内部调用线索：isinstance → load_hdf5_to_dict（含分支中的调用，实际路径由条件决定）。
def load_hdf5_to_dict(hdf5_file, index, keys_to_ignore=[]):  # noqa: B006
    data_dict = {}

    for key in hdf5_file:
        if key in keys_to_ignore:
            continue

        curr_data = hdf5_file[key]
        if isinstance(curr_data, h5py.Group):
            data_dict[key] = load_hdf5_to_dict(curr_data, index, keys_to_ignore=keys_to_ignore)
        elif isinstance(curr_data, h5py.Dataset):
            data_dict[key] = curr_data[index]
        else:
            raise ValueError

    return data_dict


# 【TrajectoryReader】读取HDF5轨迹和多相机视频，按时间步对齐后写入LeRobot。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
class TrajectoryReader:
    # 【TrajectoryReader.__init__】初始化TrajectoryReader的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._hdf5_file、self._read_images、self._length、self._video_readers、self._index。
    # 输入接口：filepath；read_images。
    # 内部调用线索：h5py.File → get_hdf5_length（含分支中的调用，实际路径由条件决定）。
    def __init__(self, filepath, read_images=True):  # noqa: FBT002
        self._hdf5_file = h5py.File(filepath, "r")
        is_video_folder = "observations/videos" in self._hdf5_file
        self._read_images = read_images and is_video_folder
        self._length = get_hdf5_length(self._hdf5_file)
        self._video_readers = {}
        self._index = 0

    # 【TrajectoryReader.length】本函数位于“DROID数据转换”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。
    # 返回值可从这里追踪：self._length。
    def length(self):
        return self._length

    # 【TrajectoryReader.read_timestep】本函数位于“DROID数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 keys_to_ignore.copy → load_hdf5_to_dict 追踪具体实现。
    # 输入接口：index；keys_to_ignore。
    # 返回值可从这里追踪：timestep。
    # 内部调用线索：keys_to_ignore.copy → load_hdf5_to_dict（含分支中的调用，实际路径由条件决定）。
    def read_timestep(self, index=None, keys_to_ignore=[]):  # noqa: B006
        # Make Sure We Read Within Range #
        if index is None:
            index = self._index
        else:
            assert not self._read_images
            self._index = index
        assert index < self._length

        # Load Low Dimensional Data #
        keys_to_ignore = [*keys_to_ignore.copy(), "videos"]
        timestep = load_hdf5_to_dict(self._hdf5_file, self._index, keys_to_ignore=keys_to_ignore)

        # Increment Read Index #
        self._index += 1

        # Return Timestep #
        return timestep

    # 【TrajectoryReader.close】本函数位于“DROID数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self._hdf5_file.close 追踪具体实现。
    def close(self):
        self._hdf5_file.close()


# 【load_trajectory】本函数位于“DROID数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 TrajectoryReader → RecordedMultiCameraWrapper → traj_reader.length 追踪具体实现。
# 输入接口：filepath；read_cameras；recording_folderpath；camera_kwargs；remove_skipped_steps；num_samples_per_traj；num_samples_per_traj_coeff。
# 返回值可从这里追踪：timestep_list。
# 内部调用线索：TrajectoryReader → RecordedMultiCameraWrapper → traj_reader.length → min → np.sort（含分支中的调用，实际路径由条件决定）。
def load_trajectory(
    filepath=None,
    read_cameras=True,  # noqa: FBT002
    recording_folderpath=None,
    camera_kwargs={},  # noqa: B006
    remove_skipped_steps=False,  # noqa: FBT002
    num_samples_per_traj=None,
    num_samples_per_traj_coeff=1.5,
):
    read_recording_folderpath = read_cameras and (recording_folderpath is not None)

    traj_reader = TrajectoryReader(filepath)
    if read_recording_folderpath:
        camera_reader = RecordedMultiCameraWrapper(recording_folderpath, camera_kwargs)

    horizon = traj_reader.length()
    timestep_list = []

    # Choose Timesteps To Save #
    if num_samples_per_traj:
        num_to_save = num_samples_per_traj
        if remove_skipped_steps:
            num_to_save = int(num_to_save * num_samples_per_traj_coeff)
        max_size = min(num_to_save, horizon)
        indices_to_save = np.sort(np.random.choice(horizon, size=max_size, replace=False))
    else:
        indices_to_save = np.arange(horizon)

    # Iterate Over Trajectory #
    for i in indices_to_save:
        # Get HDF5 Data #
        timestep = traj_reader.read_timestep(index=i)

        # If Applicable, Get Recorded Data #
        if read_recording_folderpath:
            timestamp_dict = timestep["observation"]["timestamp"]["cameras"]
            camera_type_dict = {
                k: camera_type_to_string_dict[v] for k, v in timestep["observation"]["camera_type"].items()
            }
            camera_obs = camera_reader.read_cameras(
                index=i, camera_type_dict=camera_type_dict, timestamp_dict=timestamp_dict
            )
            camera_failed = camera_obs is None

            # Add Data To Timestep If Successful #
            if camera_failed:
                break
            timestep["observation"].update(camera_obs)

        # Filter Steps #
        step_skipped = not timestep["observation"]["controller_info"].get("movement_enabled", True)
        delete_skipped_step = step_skipped and remove_skipped_steps

        # Save Filtered Timesteps #
        if delete_skipped_step:
            del timestep
        else:
            timestep_list.append(timestep)

    # Remove Extra Transitions #
    timestep_list = np.array(timestep_list)
    if (num_samples_per_traj is not None) and (len(timestep_list) > num_samples_per_traj):
        ind_to_keep = np.random.choice(len(timestep_list), size=num_samples_per_traj, replace=False)
        timestep_list = timestep_list[ind_to_keep]

    # Close Readers #
    traj_reader.close()

    # Return Data #
    return timestep_list


if __name__ == "__main__":
    tyro.cli(main)
