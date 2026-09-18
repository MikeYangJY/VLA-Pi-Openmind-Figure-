# `examples/libero/main.py` 中文阅读说明

**定位：** 仿真评测入口。

启动LIBERO任务，收集图像与状态，向服务端请求动作块并逐步执行，记录成功与视频。

**建议读法：** Args→eval_libero→环境reset→观测变换→client.infer→取动作→env.step→汇总。

**易错点：** 动作块执行长度由评测脚本设置；不能把模型预测50步理解成必须全部执行。

[注释源码](../../../code/examples/libero/main.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/main.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [Args](../../../code/examples/libero/main.py#L29) | 启动LIBERO任务，收集图像与状态，向服务端请求动作块并逐步执行，记录成功与视频。 本类封装其中的配置、状态或接口；具体入口见下面的方法。 |
| [eval_libero](../../../code/examples/libero/main.py#L60) | 本函数位于“仿真评测入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.random.seed → benchmark.get_benchmark_dict → benchmark_dict[args.task_suite_name] 追踪具体实现。 |
| [_get_libero_env](../../../code/examples/libero/main.py#L205) | 本函数位于“仿真评测入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 pathlib.Path → get_libero_path → OffScreenRenderEnv 追踪具体实现。 |
| [_quat2axisangle](../../../code/examples/libero/main.py#L219) | 本函数位于“仿真评测入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.sqrt → math.isclose → np.zeros 追踪具体实现。 |
