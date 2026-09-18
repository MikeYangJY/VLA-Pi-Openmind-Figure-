# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：ALOHA真机控制｜通过机器人驱动读取双臂关节/相机，并发送关节和夹爪命令。
# 阅读顺序：先get_observation，再step；reset调用复位流程，setup_robots建立硬件连接。
# 重点边界：动作维度、夹爪单位和关节顺序直接影响真实运动；本轮只阅读。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
# Ignore lint errors because this file is mostly copied from ACT (https://github.com/tonyzhaozh/act).
# ruff: noqa
import collections
import time
from typing import Optional, List
import dm_env
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
import numpy as np

from examples.aloha_real import constants
from examples.aloha_real import robot_utils

# This is the reset position that is used by the standard Aloha runtime.
DEFAULT_RESET_POSITION = [0, -0.96, 1.16, 0, -0.3, 0]


# 【RealEnv】通过机器人驱动读取双臂关节/相机，并发送关节和夹爪命令。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
class RealEnv:
    """
    Environment for real robot bi-manual manipulation
    Action space:      [left_arm_qpos (6),             # absolute joint position
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_qpos (6),            # absolute joint position
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),          # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"cam_high": (480x640x3),        # h, w, c, dtype='uint8'
                                   "cam_low": (480x640x3),         # h, w, c, dtype='uint8'
                                   "cam_left_wrist": (480x640x3),  # h, w, c, dtype='uint8'
                                   "cam_right_wrist": (480x640x3)} # h, w, c, dtype='uint8'
    """

    # 【RealEnv.__init__】初始化RealEnv的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._reset_position、self.puppet_bot_left、self.puppet_bot_right、self.recorder_left、self.recorder_right。
    # 输入接口：init_node；reset_position:Optional[List[float]]；setup_robots:bool。
    # 内部调用线索：InterbotixManipulatorXS → self.setup_robots → robot_utils.Recorder → robot_utils.ImageRecorder → JointSingleCommand（含分支中的调用，实际路径由条件决定）。
    def __init__(self, init_node, *, reset_position: Optional[List[float]] = None, setup_robots: bool = True):
        # reset_position = START_ARM_POSE[:6]
        self._reset_position = reset_position[:6] if reset_position else DEFAULT_RESET_POSITION

        self.puppet_bot_left = InterbotixManipulatorXS(
            robot_model="vx300s",
            group_name="arm",
            gripper_name="gripper",
            robot_name="puppet_left",
            init_node=init_node,
        )
        self.puppet_bot_right = InterbotixManipulatorXS(
            robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name="puppet_right", init_node=False
        )
        if setup_robots:
            self.setup_robots()

        self.recorder_left = robot_utils.Recorder("left", init_node=False)
        self.recorder_right = robot_utils.Recorder("right", init_node=False)
        self.image_recorder = robot_utils.ImageRecorder(init_node=False)
        self.gripper_command = JointSingleCommand(name="gripper")

    # 【RealEnv.setup_robots】本函数位于“ALOHA真机控制”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 robot_utils.setup_puppet_bot 追踪具体实现。
    def setup_robots(self):
        robot_utils.setup_puppet_bot(self.puppet_bot_left)
        robot_utils.setup_puppet_bot(self.puppet_bot_right)

    # 【RealEnv.get_qpos】本函数位于“ALOHA真机控制”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 constants.PUPPET_GRIPPER_POSITION_NORMALIZE_FN → np.concatenate 追踪具体实现。
    # 返回值可从这里追踪：np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])。
    # 内部调用线索：constants.PUPPET_GRIPPER_POSITION_NORMALIZE_FN → np.concatenate（含分支中的调用，实际路径由条件决定）。
    def get_qpos(self):
        left_qpos_raw = self.recorder_left.qpos
        right_qpos_raw = self.recorder_right.qpos
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [
            constants.PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[7])
        ]  # this is position not joint
        right_gripper_qpos = [
            constants.PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[7])
        ]  # this is position not joint
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    # 【RealEnv.get_qvel】本函数位于“ALOHA真机控制”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 constants.PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN → np.concatenate 追踪具体实现。
    # 返回值可从这里追踪：np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])。
    # 内部调用线索：constants.PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN → np.concatenate（含分支中的调用，实际路径由条件决定）。
    def get_qvel(self):
        left_qvel_raw = self.recorder_left.qvel
        right_qvel_raw = self.recorder_right.qvel
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [constants.PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[7])]
        right_gripper_qvel = [constants.PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[7])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    # 【RealEnv.get_effort】本函数位于“ALOHA真机控制”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.concatenate 追踪具体实现。
    # 返回值可从这里追踪：np.concatenate([left_robot_effort, right_robot_effort])。
    def get_effort(self):
        left_effort_raw = self.recorder_left.effort
        right_effort_raw = self.recorder_right.effort
        left_robot_effort = left_effort_raw[:7]
        right_robot_effort = right_effort_raw[:7]
        return np.concatenate([left_robot_effort, right_robot_effort])

    # 【RealEnv.get_images】本函数位于“ALOHA真机控制”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.image_recorder.get_images 追踪具体实现。
    # 返回值可从这里追踪：self.image_recorder.get_images()。
    def get_images(self):
        return self.image_recorder.get_images()

    # 【RealEnv.set_gripper_pose】本函数位于“ALOHA真机控制”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 constants.PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN → self.puppet_bot_left.gripper.core.pub_single.publish → self.puppet_bot_right.gripper.core.pub_single.publish 追踪具体实现。
    # 输入接口：left_gripper_desired_pos_normalized；right_gripper_desired_pos_normalized。
    # 内部调用线索：constants.PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN → self.puppet_bot_left.gripper.core.pub_single.publish → self.puppet_bot_right.gripper.core.pub_single.publish（含分支中的调用，实际路径由条件决定）。
    def set_gripper_pose(self, left_gripper_desired_pos_normalized, right_gripper_desired_pos_normalized):
        left_gripper_desired_joint = constants.PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(left_gripper_desired_pos_normalized)
        self.gripper_command.cmd = left_gripper_desired_joint
        self.puppet_bot_left.gripper.core.pub_single.publish(self.gripper_command)

        right_gripper_desired_joint = constants.PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(
            right_gripper_desired_pos_normalized
        )
        self.gripper_command.cmd = right_gripper_desired_joint
        self.puppet_bot_right.gripper.core.pub_single.publish(self.gripper_command)

    # 【RealEnv._reset_joints】本函数位于“ALOHA真机控制”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 robot_utils.move_arms 追踪具体实现。
    def _reset_joints(self):
        robot_utils.move_arms(
            [self.puppet_bot_left, self.puppet_bot_right], [self._reset_position, self._reset_position], move_time=1
        )

    # 【RealEnv._reset_gripper】本函数位于“ALOHA真机控制”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 robot_utils.move_grippers 追踪具体实现。
    def _reset_gripper(self):
        """Set to position mode and do position resets: first close then open. Then change back to PWM mode

        NOTE: This diverges from the original Aloha code which first opens then closes the gripper. Pi internal aloha data
        was collected with the gripper starting in the open position. Leaving the grippers fully closed was also found to
        increase the frequency of motor faults.
        """
        robot_utils.move_grippers(
            [self.puppet_bot_left, self.puppet_bot_right], [constants.PUPPET_GRIPPER_JOINT_CLOSE] * 2, move_time=1
        )
        robot_utils.move_grippers(
            [self.puppet_bot_left, self.puppet_bot_right], [constants.PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5
        )

    # 【RealEnv.get_observation】读取并组织当前环境的传感器/状态，形成策略下一次决策的输入。
    # 返回值可从这里追踪：obs。
    # 内部调用线索：collections.OrderedDict → self.get_qpos → self.get_qvel → self.get_effort → self.get_images（含分支中的调用，实际路径由条件决定）。
    def get_observation(self):
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos()
        obs["qvel"] = self.get_qvel()
        obs["effort"] = self.get_effort()
        obs["images"] = self.get_images()
        return obs

    # 【RealEnv.get_reward】本函数位于“ALOHA真机控制”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。
    # 返回值可从这里追踪：0。
    def get_reward(self):
        return 0

    # 【RealEnv.reset】清理上一episode/调用周期留下的状态，或调用底层环境进行重置。
    # 输入接口：fake。
    # 返回值可从这里追踪：dm_env.TimeStep(step_type=dm_env.StepType.FIRST, reward=self.get_reward(), discount=None, observation=self.get…。
    # 内部调用线索：self.puppet_bot_left.dxl.robot_reboot_motors → self.puppet_bot_right.dxl.robot_reboot_motors → self._reset_joints → self._reset_gripper → dm_env.TimeStep（含分支中的调用，实际路径由条件决定）。
    def reset(self, *, fake=False):
        if not fake:
            # Reboot puppet robot gripper motors
            self.puppet_bot_left.dxl.robot_reboot_motors("single", "gripper", True)
            self.puppet_bot_right.dxl.robot_reboot_motors("single", "gripper", True)
            self._reset_joints()
            self._reset_gripper()
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST, reward=self.get_reward(), discount=None, observation=self.get_observation()
        )

    # 【RealEnv.step】本函数位于“ALOHA真机控制”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.puppet_bot_left.arm.set_joint_positions → self.puppet_bot_right.arm.set_joint_positions → self.set_gripper_pose 追踪具体实现。
    # 输入接口：action。
    # 返回值可从这里追踪：dm_env.TimeStep(step_type=dm_env.StepType.MID, reward=self.get_reward(), discount=None, observation=self.get_o…。
    # 内部调用线索：self.puppet_bot_left.arm.set_joint_positions → self.puppet_bot_right.arm.set_joint_positions → self.set_gripper_pose → time.sleep → dm_env.TimeStep（含分支中的调用，实际路径由条件决定）。
    def step(self, action):
        state_len = int(len(action) / 2)
        left_action = action[:state_len]
        right_action = action[state_len:]
        self.puppet_bot_left.arm.set_joint_positions(left_action[:6], blocking=False)
        self.puppet_bot_right.arm.set_joint_positions(right_action[:6], blocking=False)
        self.set_gripper_pose(left_action[-1], right_action[-1])
        time.sleep(constants.DT)
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID, reward=self.get_reward(), discount=None, observation=self.get_observation()
        )


# 【get_action】本函数位于“ALOHA真机控制”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.zeros → constants.MASTER_GRIPPER_JOINT_NORMALIZE_FN 追踪具体实现。
# 输入接口：master_bot_left；master_bot_right。
# 返回值可从这里追踪：action。
# 内部调用线索：np.zeros → constants.MASTER_GRIPPER_JOINT_NORMALIZE_FN（含分支中的调用，实际路径由条件决定）。
def get_action(master_bot_left, master_bot_right):
    action = np.zeros(14)  # 6 joint + 1 gripper, for two arms
    # Arm actions
    action[:6] = master_bot_left.dxl.joint_states.position[:6]
    action[7 : 7 + 6] = master_bot_right.dxl.joint_states.position[:6]
    # Gripper actions
    action[6] = constants.MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_left.dxl.joint_states.position[6])
    action[7 + 6] = constants.MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_right.dxl.joint_states.position[6])

    return action


# 【make_real_env】本函数位于“ALOHA真机控制”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 RealEnv 追踪具体实现。
# 输入接口：init_node；reset_position:Optional[List[float]]；setup_robots:bool。
# 返回类型：RealEnv；类型/shape约定需与调用方配套。
def make_real_env(init_node, *, reset_position: Optional[List[float]] = None, setup_robots: bool = True) -> RealEnv:
    return RealEnv(init_node, reset_position=reset_position, setup_robots=setup_robots)
