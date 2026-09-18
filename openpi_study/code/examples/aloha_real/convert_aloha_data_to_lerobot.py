# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：ALOHA数据转换｜读原始ALOHA episode，检查相机和状态字段，再建立LeRobot数据集。
# 阅读顺序：DatasetConfig→create_empty_dataset→load_raw_episode_data→populate_dataset→保存。
# 重点边界：速度和effort字段可能并非每个数据集都有，需通过存在性检查分支处理。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
import numpy as np
import torch
import tqdm
import tyro


# 【DatasetConfig】读原始ALOHA episode，检查相机和状态字段，再建立LeRobot数据集。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


# 【create_empty_dataset】本函数位于“ALOHA数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 Path(LEROBOT_HOME / repo_id).exists → Path → shutil.rmtree 追踪具体实现。
# 输入接口：repo_id:str；robot_type:str；mode:Literal['video', 'image']；has_velocity:bool；has_effort:bool；dataset_config:DatasetConfig。
# 返回类型：LeRobotDataset；类型/shape约定需与调用方配套。
# 内部调用线索：Path(LEROBOT_HOME / repo_id).exists → Path → shutil.rmtree → LeRobotDataset.create（含分支中的调用，实际路径由条件决定）。
def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
    ]
    cameras = [
        "cam_high",
        "cam_low",
        "cam_left_wrist",
        "cam_right_wrist",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


# 【get_cameras】本函数位于“ALOHA数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 h5py.File → ep['/observations/images'].keys 追踪具体实现。
# 输入接口：hdf5_files:list[Path]。
# 返回类型：list[str]；类型/shape约定需与调用方配套。
# 内部调用线索：h5py.File → ep['/observations/images'].keys（含分支中的调用，实际路径由条件决定）。
def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        # ignore depth channel, not currently handled
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118


# 【has_velocity】本函数位于“ALOHA数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 h5py.File 追踪具体实现。
# 输入接口：hdf5_files:list[Path]。
# 返回类型：bool；类型/shape约定需与调用方配套。
def has_velocity(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


# 【has_effort】本函数位于“ALOHA数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 h5py.File 追踪具体实现。
# 输入接口：hdf5_files:list[Path]。
# 返回类型：bool；类型/shape约定需与调用方配套。
def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


# 【load_raw_images_per_camera】本函数位于“ALOHA数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 imgs_array.append → cv2.cvtColor → cv2.imdecode 追踪具体实现。
# 输入接口：ep:h5py.File；cameras:list[str]。
# 返回类型：dict[str, np.ndarray]；类型/shape约定需与调用方配套。
# 内部调用线索：imgs_array.append → cv2.cvtColor → cv2.imdecode → np.array（含分支中的调用，实际路径由条件决定）。
def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4

        if uncompressed:
            # load all images in RAM
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            import cv2

            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                imgs_array.append(cv2.cvtColor(cv2.imdecode(data, 1), cv2.COLOR_BGR2RGB))
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


# 【load_raw_episode_data】本函数位于“ALOHA数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 h5py.File → torch.from_numpy → load_raw_images_per_camera 追踪具体实现。
# 输入接口：ep_path:Path。
# 返回类型：tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]；类型/shape约定需与调用方配套。
# 内部调用线索：h5py.File → torch.from_numpy → load_raw_images_per_camera（含分支中的调用，实际路径由条件决定）。
def load_raw_episode_data(
    ep_path: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["/observations/qpos"][:])
        action = torch.from_numpy(ep["/action"][:])

        velocity = None
        if "/observations/qvel" in ep:
            velocity = torch.from_numpy(ep["/observations/qvel"][:])

        effort = None
        if "/observations/effort" in ep:
            effort = torch.from_numpy(ep["/observations/effort"][:])

        imgs_per_cam = load_raw_images_per_camera(
            ep,
            [
                "cam_high",
                "cam_low",
                "cam_left_wrist",
                "cam_right_wrist",
            ],
        )

    return imgs_per_cam, state, action, velocity, effort


# 【populate_dataset】本函数位于“ALOHA数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 tqdm.tqdm → load_raw_episode_data → imgs_per_cam.items 追踪具体实现。
# 输入接口：dataset:LeRobotDataset；hdf5_files:list[Path]；task:str；episodes:list[int] | None。
# 返回类型：LeRobotDataset；类型/shape约定需与调用方配套。
# 内部调用线索：tqdm.tqdm → load_raw_episode_data → imgs_per_cam.items → dataset.add_frame → dataset.save_episode（含分支中的调用，实际路径由条件决定）。
def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]

        imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_path)
        num_frames = state.shape[0]

        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
            }

            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]

            dataset.add_frame(frame)

        dataset.save_episode(task=task)

    return dataset


# 【port_aloha】本函数位于“ALOHA数据转换”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 (LEROBOT_HOME / repo_id).exists → shutil.rmtree → raw_dir.exists 追踪具体实现。
# 输入接口：raw_dir:Path；repo_id:str；raw_repo_id:str | None；task:str；episodes:list[int] | None；push_to_hub:bool；is_mobile:bool；mode:Literal['video', 'image']；其余见函数签名。
# 内部调用线索：(LEROBOT_HOME / repo_id).exists → shutil.rmtree → raw_dir.exists → ValueError → download_raw（含分支中的调用，实际路径由条件决定）。
def port_aloha(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = True,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    if not raw_dir.exists():
        if raw_repo_id is None:
            raise ValueError("raw_repo_id must be provided if raw_dir does not exist")
        download_raw(raw_dir, repo_id=raw_repo_id)

    hdf5_files = sorted(raw_dir.glob("episode_*.hdf5"))

    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_aloha" if is_mobile else "aloha",
        mode=mode,
        has_effort=has_effort(hdf5_files),
        has_velocity=has_velocity(hdf5_files),
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        task=task,
        episodes=episodes,
    )
    dataset.consolidate()

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_aloha)
