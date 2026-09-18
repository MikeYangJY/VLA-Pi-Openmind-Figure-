# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：DROID真机入口｜连接DROID机器人环境与远程策略，管理重置、观测、动作执行及交互。
# 阅读顺序：先_extract_observation理解平台字典；再main看episode循环和action chunk处理。
# 重点边界：本脚本会实际驱动硬件；源码学习时不运行真机循环。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
# ruff: noqa

import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import time
from moviepy.editor import ImageSequenceClip
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import pandas as pd
from PIL import Image
from droid.robot_env import RobotEnv
import tqdm
import tyro

faulthandler.enable()

# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 15


# 【Args】连接DROID机器人环境与远程策略，管理重置、观测、动作执行及交互。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
@dataclasses.dataclass
class Args:
    # Hardware parameters
    left_camera_id: str = "<your_camera_id>"  # e.g., "24259877"
    right_camera_id: str = "<your_camera_id>"  # e.g., "24514023"
    wrist_camera_id: str = "<your_camera_id>"  # e.g., "13062452"

    # Policy parameters
    external_camera: str | None = (
        None  # which external camera should be fed to the policy, choose from ["left", "right"]
    )

    # Rollout parameters
    max_timesteps: int = 600
    # How many actions to execute from a predicted action chunk before querying policy server again
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 8

    # Remote server parameters
    remote_host: str = "0.0.0.0"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = (
        8000  # point this to the port of the policy server, default server port for openpi servers is 8000
    )


# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
# 【prevent_keyboard_interrupt】本函数位于“DROID真机入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 signal.getsignal → signal.signal 追踪具体实现。
# 内部调用线索：signal.getsignal → signal.signal（含分支中的调用，实际路径由条件决定）。
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    # 【prevent_keyboard_interrupt.handler】本函数位于“DROID真机入口”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。
    # 输入接口：signum；frame。
    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


# 【main】本脚本入口：连接DROID机器人环境与远程策略，管理重置、观测、动作执行及交互。 先_extract_observation理解平台字典；再main看episode循环和action chunk处理。
# 输入接口：args:Args。
# 内部调用线索：RobotEnv → print → websocket_client_policy.WebsocketClientPolicy → pd.DataFrame → input（含分支中的调用，实际路径由条件决定）。
def main(args: Args):
    # Make sure external camera is specified by user -- we only use one external camera for the policy
    assert (
        args.external_camera is not None and args.external_camera in ["left", "right"]
    ), f"Please specify an external camera to use for the policy, choose from ['left', 'right'], but got {args.external_camera}"

    # Initialize the Panda environment. Using joint velocity action space and gripper position action space is very important.
    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    print("Created the droid env!")

    # Connect to the policy server
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)

    df = pd.DataFrame(columns=["success", "duration", "video_filename"])

    while True:
        instruction = input("Enter instruction: ")

        # Rollout parameters
        actions_from_chunk_completed = 0
        pred_action_chunk = None

        # Prepare to save video of rollout
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        video = []
        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")
        for t_step in bar:
            start_time = time.time()
            try:
                # Get the current observation
                curr_obs = _extract_observation(
                    args,
                    env.get_observation(),
                    # Save the first observation to disk
                    save_to_disk=t_step == 0,
                )

                video.append(curr_obs[f"{args.external_camera}_image"])

                # Send websocket request to policy server if it's time to predict a new chunk
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                    actions_from_chunk_completed = 0

                    # We resize images on the robot laptop to minimize the amount of data sent to the policy server
                    # and improve latency.
                    request_data = {
                        "observation/exterior_image_1_left": image_tools.resize_with_pad(
                            curr_obs[f"{args.external_camera}_image"], 224, 224
                        ),
                        "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                        "observation/joint_position": curr_obs["joint_position"],
                        "observation/gripper_position": curr_obs["gripper_position"],
                        "prompt": instruction,
                    }

                    # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                    # Ctrl+C will be handled after the server call is complete
                    with prevent_keyboard_interrupt():
                        # this returns action chunk [10, 8] of 10 joint velocity actions (7) + gripper position (1)
                        pred_action_chunk = policy_client.infer(request_data)["actions"]
                    assert pred_action_chunk.shape == (10, 8)

                # Select current action to execute from chunk
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                # Binarize gripper action
                if action[-1].item() > 0.5:
                    # action[-1] = 1.0
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    # action[-1] = 0.0
                    action = np.concatenate([action[:-1], np.zeros((1,))])

                # clip all dimensions of action to [-1, 1]
                action = np.clip(action, -1, 1)

                env.step(action)

                # Sleep to match DROID data collection frequency
                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
            except KeyboardInterrupt:
                break

        video = np.stack(video)
        save_filename = "video_" + timestamp
        ImageSequenceClip(list(video), fps=10).write_videofile(save_filename + ".mp4", codec="libx264")

        success: str | float | None = None
        while not isinstance(success, float):
            success = input(
                "Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value 0-100 based on the evaluation spec"
            )
            if success == "y":
                success = 1.0
            elif success == "n":
                success = 0.0

            success = float(success) / 100
            if not (0 <= success <= 1):
                print(f"Success must be a number in [0, 100] but got: {success * 100}")

        df = df.append(
            {
                "success": success,
                "duration": t_step,
                "video_filename": save_filename,
            },
            ignore_index=True,
        )

        if input("Do one more eval? (enter y or n) ").lower() != "y":
            break
        env.reset()

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    csv_filename = os.path.join("results", f"eval_{timestamp}.csv")
    df.to_csv(csv_filename)
    print(f"Results saved to {csv_filename}")


# 【_extract_observation】本函数位于“DROID真机入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.array → np.concatenate → Image.fromarray 追踪具体实现。
# 输入接口：args:Args；obs_dict；save_to_disk。
# 返回值可从这里追踪：{'left_image': left_image, 'right_image': right_image, 'wrist_image': wrist_image, 'cartesian_position': carte…。
# 内部调用线索：np.array → np.concatenate → Image.fromarray → combined_image.save（含分支中的调用，实际路径由条件决定）。
def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None
    for key in image_observations:
        # Note the "left" below refers to the left camera in the stereo pair.
        # The model is only trained on left stereo cams, so we only feed those.
        if args.left_camera_id in key and "left" in key:
            left_image = image_observations[key]
        elif args.right_camera_id in key and "left" in key:
            right_image = image_observations[key]
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = image_observations[key]

    # Drop the alpha dimension
    left_image = left_image[..., :3]
    right_image = right_image[..., :3]
    wrist_image = wrist_image[..., :3]

    # Convert to RGB
    left_image = left_image[..., ::-1]
    right_image = right_image[..., ::-1]
    wrist_image = wrist_image[..., ::-1]

    # In addition to image observations, also capture the proprioceptive state
    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    # Save the images to disk so that they can be viewed live while the robot is running
    # Create one combined image to make live viewing easy
    if save_to_disk:
        combined_image = np.concatenate([left_image, wrist_image, right_image], axis=1)
        combined_image = Image.fromarray(combined_image)
        combined_image.save("robot_camera_views.png")

    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
