# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：控制循环｜按频率协调环境、策略代理和订阅者，运行多个episode。
# 阅读顺序：run→_run_episode→_step：读观测、要动作、执行动作、广播事件并计时。
# 重点边界：runtime的控制循环频率不等于每一步都完整推理大模型。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import logging
import threading
import time

from openpi_client.runtime import agent as _agent
from openpi_client.runtime import environment as _environment
from openpi_client.runtime import subscriber as _subscriber


# 【Runtime】按频率协调环境、策略代理和订阅者，运行多个episode。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
class Runtime:
    """The core module orchestrating interactions between key components of the system."""

    # 【Runtime.__init__】初始化Runtime的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._environment、self._agent、self._subscribers、self._max_hz、self._num_episodes。
    # 输入接口：environment:_environment.Environment；agent:_agent.Agent；subscribers:list[_subscriber.Subscriber]；max_hz:float；num_episodes:int；max_episode_steps:int。
    # 返回类型：None；类型/shape约定需与调用方配套。
    def __init__(
        self,
        environment: _environment.Environment,
        agent: _agent.Agent,
        subscribers: list[_subscriber.Subscriber],
        max_hz: float = 0,
        num_episodes: int = 1,
        max_episode_steps: int = 0,
    ) -> None:
        self._environment = environment
        self._agent = agent
        self._subscribers = subscribers
        self._max_hz = max_hz
        self._num_episodes = num_episodes
        self._max_episode_steps = max_episode_steps

        self._in_episode = False
        self._episode_steps = 0

    # 【Runtime.run】本函数位于“控制循环”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self._run_episode → self._environment.reset 追踪具体实现。
    # 返回类型：None；类型/shape约定需与调用方配套。
    # 内部调用线索：self._run_episode → self._environment.reset（含分支中的调用，实际路径由条件决定）。
    def run(self) -> None:
        """Runs the runtime loop continuously until stop() is called or the environment is done."""
        for _ in range(self._num_episodes):
            self._run_episode()

        # Final reset, this is important for real environments to move the robot to its home position.
        self._environment.reset()

    # 【Runtime.run_in_new_thread】本函数位于“控制循环”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 threading.Thread → thread.start 追踪具体实现。
    # 返回类型：threading.Thread；类型/shape约定需与调用方配套。
    # 内部调用线索：threading.Thread → thread.start（含分支中的调用，实际路径由条件决定）。
    def run_in_new_thread(self) -> threading.Thread:
        """Runs the runtime loop in a new thread."""
        thread = threading.Thread(target=self.run)
        thread.start()
        return thread

    # 【Runtime.mark_episode_complete】本函数位于“控制循环”路径，负责把调用方输入推进到下一处理阶段。它主要通过本地数组/字段运算完成处理。
    # 返回类型：None；类型/shape约定需与调用方配套。
    def mark_episode_complete(self) -> None:
        """Marks the end of an episode."""
        self._in_episode = False

    # 【Runtime._run_episode】组织一次任务的reset、逐步控制、频率调度与结束通知。
    # 返回类型：None；类型/shape约定需与调用方配套。
    # 内部调用线索：logging.info → self._environment.reset → self._agent.reset → subscriber.on_episode_start → time.time（含分支中的调用，实际路径由条件决定）。
    def _run_episode(self) -> None:
        """Runs a single episode."""
        logging.info("Starting episode...")
        self._environment.reset()
        self._agent.reset()
        for subscriber in self._subscribers:
            subscriber.on_episode_start()

        self._in_episode = True
        self._episode_steps = 0
        step_time = 1 / self._max_hz if self._max_hz > 0 else 0
        last_step_time = time.time()

        while self._in_episode:
            self._step()
            self._episode_steps += 1

            # Sleep to maintain the desired frame rate
            now = time.time()
            dt = now - last_step_time
            if dt < step_time:
                time.sleep(step_time - dt)
                last_step_time = time.time()
            else:
                last_step_time = now

        logging.info("Episode completed.")
        for subscriber in self._subscribers:
            subscriber.on_episode_end()

    # 【Runtime._step】从环境读取观测，向agent要动作，交给environment执行，再通知订阅者。
    # 返回类型：None；类型/shape约定需与调用方配套。
    # 内部调用线索：self._environment.get_observation → self._agent.get_action → self._environment.apply_action → subscriber.on_step → self._environment.is_episode_complete（含分支中的调用，实际路径由条件决定）。
    def _step(self) -> None:
        """A single step of the runtime loop."""
        observation = self._environment.get_observation()
        action = self._agent.get_action(observation)
        self._environment.apply_action(action)

        for subscriber in self._subscribers:
            subscriber.on_step(observation, action)

        if self._environment.is_episode_complete() or (
            self._max_episode_steps > 0 and self._episode_steps >= self._max_episode_steps
        ):
            self.mark_episode_complete()
