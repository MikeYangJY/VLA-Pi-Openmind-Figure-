# `examples/aloha_real/real_env.py` 中文阅读说明

**定位：** ALOHA真机控制。

通过机器人驱动读取双臂关节/相机，并发送关节和夹爪命令。

**建议读法：** 先get_observation，再step；reset调用复位流程，setup_robots建立硬件连接。

**易错点：** 动作维度、夹爪单位和关节顺序直接影响真实运动；本轮只阅读。

[注释源码](../../../code/examples/aloha_real/real_env.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/aloha_real/real_env.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [RealEnv](../../../code/examples/aloha_real/real_env.py#L25) | 通过机器人驱动读取双臂关节/相机，并发送关节和夹爪命令。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [RealEnv.__init__](../../../code/examples/aloha_real/real_env.py#L50) | 初始化RealEnv的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._reset_position、self.puppet_bot_left、self.puppet_bot_right、self.recorder_left、self.recorder_right。 |
| [RealEnv.setup_robots](../../../code/examples/aloha_real/real_env.py#L73) | 本函数位于“ALOHA真机控制”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 robot_utils.setup_puppet_bot 追踪具体实现。 |
| [RealEnv.get_qpos](../../../code/examples/aloha_real/real_env.py#L80) | 本函数位于“ALOHA真机控制”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 constants.PUPPET_GRIPPER_POSITION_NORMALIZE_FN → np.concatenate 追踪具体实现。 |
| [RealEnv.get_qvel](../../../code/examples/aloha_real/real_env.py#L96) | 本函数位于“ALOHA真机控制”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 constants.PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN → np.concatenate 追踪具体实现。 |
| [RealEnv.get_effort](../../../code/examples/aloha_real/real_env.py#L107) | 本函数位于“ALOHA真机控制”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.concatenate 追踪具体实现。 |
| [RealEnv.get_images](../../../code/examples/aloha_real/real_env.py#L116) | 本函数位于“ALOHA真机控制”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.image_recorder.get_images 追踪具体实现。 |
| [RealEnv.set_gripper_pose](../../../code/examples/aloha_real/real_env.py#L122) | 本函数位于“ALOHA真机控制”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 constants.PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN → self.puppet_bot_left.gripper.core.pub_single.publish → self.puppet_bot_right.gripper.core.pub_single.publish 追踪具体实现。 |
| [RealEnv._reset_joints](../../../code/examples/aloha_real/real_env.py#L134) | 本函数位于“ALOHA真机控制”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 robot_utils.move_arms 追踪具体实现。 |
| [RealEnv._reset_gripper](../../../code/examples/aloha_real/real_env.py#L140) | 本函数位于“ALOHA真机控制”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 robot_utils.move_grippers 追踪具体实现。 |
| [RealEnv.get_observation](../../../code/examples/aloha_real/real_env.py#L157) | 读取并组织当前环境的传感器/状态，形成策略下一次决策的输入。 |
| [RealEnv.get_reward](../../../code/examples/aloha_real/real_env.py#L167) | 本函数位于“ALOHA真机控制”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。 |
| [RealEnv.reset](../../../code/examples/aloha_real/real_env.py#L174) | 清理上一episode/调用周期留下的状态，或调用底层环境进行重置。 |
| [RealEnv.step](../../../code/examples/aloha_real/real_env.py#L189) | 本函数位于“ALOHA真机控制”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self.puppet_bot_left.arm.set_joint_positions → self.puppet_bot_right.arm.set_joint_positions → self.set_gripper_pose 追踪具体实现。 |
| [get_action](../../../code/examples/aloha_real/real_env.py#L206) | 本函数位于“ALOHA真机控制”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.zeros → constants.MASTER_GRIPPER_JOINT_NORMALIZE_FN 追踪具体实现。 |
| [make_real_env](../../../code/examples/aloha_real/real_env.py#L221) | 本函数位于“ALOHA真机控制”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 RealEnv 追踪具体实现。 |
