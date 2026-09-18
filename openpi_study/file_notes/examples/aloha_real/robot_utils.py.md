# `examples/aloha_real/robot_utils.py` 中文阅读说明

**定位：** ALOHA硬件工具。

提供ROS相机/关节回调、双臂同步移动、夹爪控制和电机配置。

**建议读法：** ImageRecorder/Recorder保存传感信息；move_arms/move_grippers下发插值动作；其余函数管理硬件状态。

**易错点：** torque、PID和复位函数属于真实硬件操作，与模型训练无关。

[注释源码](../../../code/examples/aloha_real/robot_utils.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/aloha_real/robot_utils.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [ImageRecorder](../../../code/examples/aloha_real/robot_utils.py#L26) | 提供ROS相机/关节回调、双臂同步移动、夹爪控制和电机配置。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [ImageRecorder.__init__](../../../code/examples/aloha_real/robot_utils.py#L30) | 初始化ImageRecorder的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.is_debug、self.bridge、self.camera_names、self.cam_last_timestamps。 |
| [ImageRecorder.image_cb](../../../code/examples/aloha_real/robot_utils.py#L61) | 本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 setattr → self.bridge.imgmsg_to_cv2 → getattr(self, f'{cam_name}_timestamps').append 追踪具体实现。 |
| [ImageRecorder.image_cb_cam_high](../../../code/examples/aloha_real/robot_utils.py#L88) | 本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.image_cb 追踪具体实现。 |
| [ImageRecorder.image_cb_cam_low](../../../code/examples/aloha_real/robot_utils.py#L95) | 本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.image_cb 追踪具体实现。 |
| [ImageRecorder.image_cb_cam_left_wrist](../../../code/examples/aloha_real/robot_utils.py#L102) | 本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.image_cb 追踪具体实现。 |
| [ImageRecorder.image_cb_cam_right_wrist](../../../code/examples/aloha_real/robot_utils.py#L109) | 本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.image_cb 追踪具体实现。 |
| [ImageRecorder.get_images](../../../code/examples/aloha_real/robot_utils.py#L116) | 本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 getattr → time.sleep 追踪具体实现。 |
| [ImageRecorder.print_diagnostics](../../../code/examples/aloha_real/robot_utils.py#L130) | 本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 dt_helper → getattr → print 追踪具体实现。 |
| [ImageRecorder.print_diagnostics.dt_helper](../../../code/examples/aloha_real/robot_utils.py#L135) | 本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.array → np.mean 追踪具体实现。 |
| [Recorder](../../../code/examples/aloha_real/robot_utils.py#L147) | 提供ROS相机/关节回调、双臂同步移动、夹爪控制和电机配置。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [Recorder.__init__](../../../code/examples/aloha_real/robot_utils.py#L151) | 初始化Recorder的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self.secs、self.nsecs、self.qpos、self.effort、self.arm_command。 |
| [Recorder.puppet_state_cb](../../../code/examples/aloha_real/robot_utils.py#L182) | 本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.joint_timestamps.append → time.time 追踪具体实现。 |
| [Recorder.puppet_arm_commands_cb](../../../code/examples/aloha_real/robot_utils.py#L193) | 本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.arm_command_timestamps.append → time.time 追踪具体实现。 |
| [Recorder.puppet_gripper_commands_cb](../../../code/examples/aloha_real/robot_utils.py#L201) | 本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.gripper_command_timestamps.append → time.time 追踪具体实现。 |
| [Recorder.print_diagnostics](../../../code/examples/aloha_real/robot_utils.py#L208) | 本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 dt_helper → print 追踪具体实现。 |
| [Recorder.print_diagnostics.dt_helper](../../../code/examples/aloha_real/robot_utils.py#L213) | 本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.array → np.mean 追踪具体实现。 |
| [get_arm_joint_positions](../../../code/examples/aloha_real/robot_utils.py#L228) | 本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。 |
| [get_arm_gripper_positions](../../../code/examples/aloha_real/robot_utils.py#L235) | 本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。 |
| [move_arms](../../../code/examples/aloha_real/robot_utils.py#L242) | 本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 get_arm_joint_positions → np.linspace → bot.arm.set_joint_positions 追踪具体实现。 |
| [move_grippers](../../../code/examples/aloha_real/robot_utils.py#L258) | 本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 print → JointSingleCommand → get_arm_gripper_positions 追踪具体实现。 |
| [setup_puppet_bot](../../../code/examples/aloha_real/robot_utils.py#L282) | 本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 bot.dxl.robot_reboot_motors → bot.dxl.robot_set_operating_modes → torque_on 追踪具体实现。 |
| [setup_master_bot](../../../code/examples/aloha_real/robot_utils.py#L292) | 本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 bot.dxl.robot_set_operating_modes → torque_off 追踪具体实现。 |
| [set_standard_pid_gains](../../../code/examples/aloha_real/robot_utils.py#L300) | 本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 bot.dxl.robot_set_motor_registers 追踪具体实现。 |
| [set_low_pid_gains](../../../code/examples/aloha_real/robot_utils.py#L307) | 本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 bot.dxl.robot_set_motor_registers 追踪具体实现。 |
| [torque_off](../../../code/examples/aloha_real/robot_utils.py#L314) | 本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 bot.dxl.robot_torque_enable 追踪具体实现。 |
| [torque_on](../../../code/examples/aloha_real/robot_utils.py#L321) | 本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 bot.dxl.robot_torque_enable 追踪具体实现。 |
| [sync_puppet_to_master](../../../code/examples/aloha_real/robot_utils.py#L330) | 本函数位于“ALOHA硬件工具”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 print → torque_on → get_arm_joint_positions 追踪具体实现。 |
