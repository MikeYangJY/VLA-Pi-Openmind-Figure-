# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：ALOHA硬件工具｜提供ROS相机/关节回调、双臂同步移动、夹爪控制和电机配置。
# 阅读顺序：ImageRecorder/Recorder保存传感信息；move_arms/move_grippers下发插值动作；其余函数管理硬件状态。
# 重点边界：torque、PID和复位函数属于真实硬件操作，与模型训练无关。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
# Ignore lint errors because this file is mostly copied from ACT (https://github.com/tonyzhaozh/act).
# ruff: noqa
from collections import deque
import datetime
import json
import time

from aloha.msg import RGBGrayscaleImage
from cv_bridge import CvBridge
from interbotix_xs_msgs.msg import JointGroupCommand
from interbotix_xs_msgs.msg import JointSingleCommand
import numpy as np
import rospy
from sensor_msgs.msg import JointState

from examples.aloha_real import constants


# 【ImageRecorder】提供ROS相机/关节回调、双臂同步移动、夹爪控制和电机配置。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
class ImageRecorder:
    # 【ImageRecorder.__init__】初始化ImageRecorder的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.is_debug、self.bridge、self.camera_names、self.cam_last_timestamps。
    # 输入接口：init_node；is_debug。
    # 内部调用线索：CvBridge → rospy.init_node → setattr → rospy.Subscriber → deque（含分支中的调用，实际路径由条件决定）。
    def __init__(self, init_node=True, is_debug=False):
        self.is_debug = is_debug
        self.bridge = CvBridge()
        self.camera_names = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]

        if init_node:
            rospy.init_node("image_recorder", anonymous=True)
        for cam_name in self.camera_names:
            setattr(self, f"{cam_name}_rgb_image", None)
            setattr(self, f"{cam_name}_depth_image", None)
            setattr(self, f"{cam_name}_timestamp", 0.0)
            if cam_name == "cam_high":
                callback_func = self.image_cb_cam_high
            elif cam_name == "cam_low":
                callback_func = self.image_cb_cam_low
            elif cam_name == "cam_left_wrist":
                callback_func = self.image_cb_cam_left_wrist
            elif cam_name == "cam_right_wrist":
                callback_func = self.image_cb_cam_right_wrist
            else:
                raise NotImplementedError
            rospy.Subscriber(f"/{cam_name}", RGBGrayscaleImage, callback_func)
            if self.is_debug:
                setattr(self, f"{cam_name}_timestamps", deque(maxlen=50))

        self.cam_last_timestamps = {cam_name: 0.0 for cam_name in self.camera_names}
        time.sleep(0.5)

    # 【ImageRecorder.image_cb】本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 setattr → self.bridge.imgmsg_to_cv2 → getattr(self, f'{cam_name}_timestamps').append 追踪具体实现。
    # 输入接口：cam_name；data。
    # 内部调用线索：setattr → self.bridge.imgmsg_to_cv2 → getattr(self, f'{cam_name}_timestamps').append → getattr（含分支中的调用，实际路径由条件决定）。
    def image_cb(self, cam_name, data):
        setattr(
            self,
            f"{cam_name}_rgb_image",
            self.bridge.imgmsg_to_cv2(data.images[0], desired_encoding="bgr8"),
        )
        # setattr(
        #     self,
        #     f"{cam_name}_depth_image",
        #     self.bridge.imgmsg_to_cv2(data.images[1], desired_encoding="mono16"),
        # )
        setattr(
            self,
            f"{cam_name}_timestamp",
            data.header.stamp.secs + data.header.stamp.nsecs * 1e-9,
        )
        # setattr(self, f'{cam_name}_secs', data.images[0].header.stamp.secs)
        # setattr(self, f'{cam_name}_nsecs', data.images[0].header.stamp.nsecs)
        # cv2.imwrite('/home/lucyshi/Desktop/sample.jpg', cv_image)
        if self.is_debug:
            getattr(self, f"{cam_name}_timestamps").append(
                data.images[0].header.stamp.secs + data.images[0].header.stamp.nsecs * 1e-9
            )

    # 【ImageRecorder.image_cb_cam_high】本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.image_cb 追踪具体实现。
    # 输入接口：data。
    # 返回值可从这里追踪：self.image_cb(cam_name, data)。
    def image_cb_cam_high(self, data):
        cam_name = "cam_high"
        return self.image_cb(cam_name, data)

    # 【ImageRecorder.image_cb_cam_low】本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.image_cb 追踪具体实现。
    # 输入接口：data。
    # 返回值可从这里追踪：self.image_cb(cam_name, data)。
    def image_cb_cam_low(self, data):
        cam_name = "cam_low"
        return self.image_cb(cam_name, data)

    # 【ImageRecorder.image_cb_cam_left_wrist】本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.image_cb 追踪具体实现。
    # 输入接口：data。
    # 返回值可从这里追踪：self.image_cb(cam_name, data)。
    def image_cb_cam_left_wrist(self, data):
        cam_name = "cam_left_wrist"
        return self.image_cb(cam_name, data)

    # 【ImageRecorder.image_cb_cam_right_wrist】本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.image_cb 追踪具体实现。
    # 输入接口：data。
    # 返回值可从这里追踪：self.image_cb(cam_name, data)。
    def image_cb_cam_right_wrist(self, data):
        cam_name = "cam_right_wrist"
        return self.image_cb(cam_name, data)

    # 【ImageRecorder.get_images】本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 getattr → time.sleep 追踪具体实现。
    # 返回值可从这里追踪：image_dict。
    # 内部调用线索：getattr → time.sleep（含分支中的调用，实际路径由条件决定）。
    def get_images(self):
        image_dict = {}
        for cam_name in self.camera_names:
            while getattr(self, f"{cam_name}_timestamp") <= self.cam_last_timestamps[cam_name]:
                time.sleep(0.00001)
            rgb_image = getattr(self, f"{cam_name}_rgb_image")
            depth_image = getattr(self, f"{cam_name}_depth_image")
            self.cam_last_timestamps[cam_name] = getattr(self, f"{cam_name}_timestamp")
            image_dict[cam_name] = rgb_image
            image_dict[f"{cam_name}_depth"] = depth_image
        return image_dict

    # 【ImageRecorder.print_diagnostics】本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 dt_helper → getattr → print 追踪具体实现。
    # 内部调用线索：dt_helper → getattr → print（含分支中的调用，实际路径由条件决定）。
    def print_diagnostics(self):
        # 【ImageRecorder.print_diagnostics.dt_helper】本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.array → np.mean 追踪具体实现。
        # 输入接口：l。
        # 返回值可从这里追踪：np.mean(diff)。
        # 内部调用线索：np.array → np.mean（含分支中的调用，实际路径由条件决定）。
        def dt_helper(l):
            l = np.array(l)
            diff = l[1:] - l[:-1]
            return np.mean(diff)

        for cam_name in self.camera_names:
            image_freq = 1 / dt_helper(getattr(self, f"{cam_name}_timestamps"))
            print(f"{cam_name} {image_freq=:.2f}")
        print()


# 【Recorder】提供ROS相机/关节回调、双臂同步移动、夹爪控制和电机配置。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
class Recorder:
    # 【Recorder.__init__】初始化Recorder的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.secs、self.nsecs、self.qpos、self.effort、self.arm_command。
    # 输入接口：side；init_node；is_debug。
    # 内部调用线索：rospy.init_node → rospy.Subscriber → deque → time.sleep（含分支中的调用，实际路径由条件决定）。
    def __init__(self, side, init_node=True, is_debug=False):
        self.secs = None
        self.nsecs = None
        self.qpos = None
        self.effort = None
        self.arm_command = None
        self.gripper_command = None
        self.is_debug = is_debug

        if init_node:
            rospy.init_node("recorder", anonymous=True)
        rospy.Subscriber(f"/puppet_{side}/joint_states", JointState, self.puppet_state_cb)
        rospy.Subscriber(
            f"/puppet_{side}/commands/joint_group",
            JointGroupCommand,
            self.puppet_arm_commands_cb,
        )
        rospy.Subscriber(
            f"/puppet_{side}/commands/joint_single",
            JointSingleCommand,
            self.puppet_gripper_commands_cb,
        )
        if self.is_debug:
            self.joint_timestamps = deque(maxlen=50)
            self.arm_command_timestamps = deque(maxlen=50)
            self.gripper_command_timestamps = deque(maxlen=50)
        time.sleep(0.1)

    # 【Recorder.puppet_state_cb】本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.joint_timestamps.append → time.time 追踪具体实现。
    # 输入接口：data。
    # 内部调用线索：self.joint_timestamps.append → time.time（含分支中的调用，实际路径由条件决定）。
    def puppet_state_cb(self, data):
        self.qpos = data.position
        self.qvel = data.velocity
        self.effort = data.effort
        self.data = data
        if self.is_debug:
            self.joint_timestamps.append(time.time())

    # 【Recorder.puppet_arm_commands_cb】本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.arm_command_timestamps.append → time.time 追踪具体实现。
    # 输入接口：data。
    # 内部调用线索：self.arm_command_timestamps.append → time.time（含分支中的调用，实际路径由条件决定）。
    def puppet_arm_commands_cb(self, data):
        self.arm_command = data.cmd
        if self.is_debug:
            self.arm_command_timestamps.append(time.time())

    # 【Recorder.puppet_gripper_commands_cb】本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.gripper_command_timestamps.append → time.time 追踪具体实现。
    # 输入接口：data。
    # 内部调用线索：self.gripper_command_timestamps.append → time.time（含分支中的调用，实际路径由条件决定）。
    def puppet_gripper_commands_cb(self, data):
        self.gripper_command = data.cmd
        if self.is_debug:
            self.gripper_command_timestamps.append(time.time())

    # 【Recorder.print_diagnostics】本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 dt_helper → print 追踪具体实现。
    # 内部调用线索：dt_helper → print（含分支中的调用，实际路径由条件决定）。
    def print_diagnostics(self):
        # 【Recorder.print_diagnostics.dt_helper】本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.array → np.mean 追踪具体实现。
        # 输入接口：l。
        # 返回值可从这里追踪：np.mean(diff)。
        # 内部调用线索：np.array → np.mean（含分支中的调用，实际路径由条件决定）。
        def dt_helper(l):
            l = np.array(l)
            diff = l[1:] - l[:-1]
            return np.mean(diff)

        joint_freq = 1 / dt_helper(self.joint_timestamps)
        arm_command_freq = 1 / dt_helper(self.arm_command_timestamps)
        gripper_command_freq = 1 / dt_helper(self.gripper_command_timestamps)

        print(f"{joint_freq=:.2f}\n{arm_command_freq=:.2f}\n{gripper_command_freq=:.2f}\n")


# 【get_arm_joint_positions】本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。
# 输入接口：bot。
# 返回值可从这里追踪：bot.arm.core.joint_states.position[:6]。
def get_arm_joint_positions(bot):
    return bot.arm.core.joint_states.position[:6]


# 【get_arm_gripper_positions】本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。
# 输入接口：bot。
# 返回值可从这里追踪：bot.gripper.core.joint_states.position[6]。
def get_arm_gripper_positions(bot):
    return bot.gripper.core.joint_states.position[6]


# 【move_arms】本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 get_arm_joint_positions → np.linspace → bot.arm.set_joint_positions 追踪具体实现。
# 输入接口：bot_list；target_pose_list；move_time。
# 内部调用线索：get_arm_joint_positions → np.linspace → bot.arm.set_joint_positions → time.sleep（含分支中的调用，实际路径由条件决定）。
def move_arms(bot_list, target_pose_list, move_time=1):
    num_steps = int(move_time / constants.DT)
    curr_pose_list = [get_arm_joint_positions(bot) for bot in bot_list]
    traj_list = [
        np.linspace(curr_pose, target_pose, num_steps)
        for curr_pose, target_pose in zip(curr_pose_list, target_pose_list)
    ]
    for t in range(num_steps):
        for bot_id, bot in enumerate(bot_list):
            bot.arm.set_joint_positions(traj_list[bot_id][t], blocking=False)
        time.sleep(constants.DT)


# 【move_grippers】本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 print → JointSingleCommand → get_arm_gripper_positions 追踪具体实现。
# 输入接口：bot_list；target_pose_list；move_time。
# 内部调用线索：print → JointSingleCommand → get_arm_gripper_positions → np.linspace → open（含分支中的调用，实际路径由条件决定）。
def move_grippers(bot_list, target_pose_list, move_time):
    print(f"Moving grippers to {target_pose_list=}")
    gripper_command = JointSingleCommand(name="gripper")
    num_steps = int(move_time / constants.DT)
    curr_pose_list = [get_arm_gripper_positions(bot) for bot in bot_list]
    traj_list = [
        np.linspace(curr_pose, target_pose, num_steps)
        for curr_pose, target_pose in zip(curr_pose_list, target_pose_list)
    ]

    with open(f"/data/gripper_traj_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl", "a") as f:
        for t in range(num_steps):
            d = {}
            for bot_id, bot in enumerate(bot_list):
                gripper_command.cmd = traj_list[bot_id][t]
                bot.gripper.core.pub_single.publish(gripper_command)
                d[bot_id] = {"obs": get_arm_gripper_positions(bot), "act": traj_list[bot_id][t]}
            f.write(json.dumps(d) + "\n")
            time.sleep(constants.DT)


# 【setup_puppet_bot】本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 bot.dxl.robot_reboot_motors → bot.dxl.robot_set_operating_modes → torque_on 追踪具体实现。
# 输入接口：bot。
# 内部调用线索：bot.dxl.robot_reboot_motors → bot.dxl.robot_set_operating_modes → torque_on（含分支中的调用，实际路径由条件决定）。
def setup_puppet_bot(bot):
    bot.dxl.robot_reboot_motors("single", "gripper", True)
    bot.dxl.robot_set_operating_modes("group", "arm", "position")
    bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    torque_on(bot)


# 【setup_master_bot】本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 bot.dxl.robot_set_operating_modes → torque_off 追踪具体实现。
# 输入接口：bot。
# 内部调用线索：bot.dxl.robot_set_operating_modes → torque_off（含分支中的调用，实际路径由条件决定）。
def setup_master_bot(bot):
    bot.dxl.robot_set_operating_modes("group", "arm", "pwm")
    bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    torque_off(bot)


# 【set_standard_pid_gains】本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 bot.dxl.robot_set_motor_registers 追踪具体实现。
# 输入接口：bot。
def set_standard_pid_gains(bot):
    bot.dxl.robot_set_motor_registers("group", "arm", "Position_P_Gain", 800)
    bot.dxl.robot_set_motor_registers("group", "arm", "Position_I_Gain", 0)


# 【set_low_pid_gains】本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 bot.dxl.robot_set_motor_registers 追踪具体实现。
# 输入接口：bot。
def set_low_pid_gains(bot):
    bot.dxl.robot_set_motor_registers("group", "arm", "Position_P_Gain", 100)
    bot.dxl.robot_set_motor_registers("group", "arm", "Position_I_Gain", 0)


# 【torque_off】本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 bot.dxl.robot_torque_enable 追踪具体实现。
# 输入接口：bot。
def torque_off(bot):
    bot.dxl.robot_torque_enable("group", "arm", False)
    bot.dxl.robot_torque_enable("single", "gripper", False)


# 【torque_on】本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 bot.dxl.robot_torque_enable 追踪具体实现。
# 输入接口：bot。
def torque_on(bot):
    bot.dxl.robot_torque_enable("group", "arm", True)
    bot.dxl.robot_torque_enable("single", "gripper", True)


# for DAgger
# 【sync_puppet_to_master】本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 print → torque_on → get_arm_joint_positions 追踪具体实现。
# 输入接口：master_bot_left；master_bot_right；puppet_bot_left；puppet_bot_right。
# 内部调用线索：print → torque_on → get_arm_joint_positions → get_arm_gripper_positions → move_arms（含分支中的调用，实际路径由条件决定）。
def sync_puppet_to_master(master_bot_left, master_bot_right, puppet_bot_left, puppet_bot_right):
    print("\nSyncing!")

    # activate master arms
    torque_on(master_bot_left)
    torque_on(master_bot_right)

    # get puppet arm positions
    puppet_left_qpos = get_arm_joint_positions(puppet_bot_left)
    puppet_right_qpos = get_arm_joint_positions(puppet_bot_right)

    # get puppet gripper positions
    puppet_left_gripper = get_arm_gripper_positions(puppet_bot_left)
    puppet_right_gripper = get_arm_gripper_positions(puppet_bot_right)

    # move master arms to puppet positions
    move_arms(
        [master_bot_left, master_bot_right],
        [puppet_left_qpos, puppet_right_qpos],
        move_time=1,
    )

    # move master grippers to puppet positions
    move_grippers(
        [master_bot_left, master_bot_right],
        [puppet_left_gripper, puppet_right_gripper],
        move_time=1,
    )
