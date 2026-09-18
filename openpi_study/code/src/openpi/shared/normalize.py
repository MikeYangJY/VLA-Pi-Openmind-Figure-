# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：数据统计底层｜维护流式均值方差和直方图分位数，并保存/读取NormStats。
# 阅读顺序：RunningStats.update累计样本，get_statistics输出统计，serialize/save负责存盘。
# 重点边界：归一化统计来自数据，不是神经网络可训练权重。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import json
import pathlib

import numpy as np
import numpydantic
import pydantic


# 【NormStats】每个物理维度的mean/std/q01/q99等统计。
@pydantic.dataclasses.dataclass
class NormStats:
    mean: numpydantic.NDArray
    std: numpydantic.NDArray
    q01: numpydantic.NDArray | None = None  # 1st quantile
    q99: numpydantic.NDArray | None = None  # 99th quantile


# 【RunningStats】可以逐批更新的统计器，用于大数据集而非模型梯度学习。
class RunningStats:
    """Compute running statistics of a batch of vectors."""

    # 【RunningStats.__init__】初始化RunningStats的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._count、self._mean、self._mean_of_squares、self._min、self._max。
    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = 5000  # for computing quantiles on the fly

    # 【RunningStats.update】流式累积每一维的样本统计与直方图，无需一次把全部数据放入内存。
    # 输入接口：batch:np.ndarray。
    # 返回类型：None；类型/shape约定需与调用方配套。
    # 内部调用线索：batch.reshape → np.mean → np.min → np.max → np.zeros（含分支中的调用，实际路径由条件决定）。
    def update(self, batch: np.ndarray) -> None:
        """
        Update the running statistics with a batch of vectors.

        Args:
            vectors (np.ndarray): An array where all dimensions except the last are batch dimensions.
        """
        batch = batch.reshape(-1, batch.shape[-1])
        num_elements, vector_length = batch.shape
        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.size:
                raise ValueError("The length of new vectors does not match the initialized vector length.")
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)

        # Update running mean and mean of squares.
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (num_elements / self._count)

        self._update_histograms(batch)

    # 【RunningStats.get_statistics】从累计值输出mean/std和近似分位数，供Normalize/Unnormalize使用。
    # 返回类型：NormStats；类型/shape约定需与调用方配套。
    # 内部调用线索：ValueError → np.sqrt → np.maximum → self._compute_quantiles → NormStats（含分支中的调用，实际路径由条件决定）。
    def get_statistics(self) -> NormStats:
        """
        Compute and return the statistics of the vectors processed so far.

        Returns:
            dict: A dictionary containing the computed statistics.
        """
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0, variance))
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        return NormStats(mean=self._mean, std=stddev, q01=q01, q99=q99)

    # 【RunningStats._adjust_histograms】本函数位于“数据统计底层”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.linspace → np.histogram 追踪具体实现。
    # 内部调用线索：np.linspace → np.histogram（含分支中的调用，实际路径由条件决定）。
    def _adjust_histograms(self):
        """Adjust histograms when min or max changes."""
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1)

            # Redistribute the existing histogram counts to the new bins
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])

            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    # 【RunningStats._update_histograms】本函数位于“数据统计底层”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.histogram 追踪具体实现。
    # 输入接口：batch:np.ndarray。
    # 返回类型：None；类型/shape约定需与调用方配套。
    def _update_histograms(self, batch: np.ndarray) -> None:
        """Update histograms with new vectors."""
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    # 【RunningStats._compute_quantiles】本函数位于“数据统计底层”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 np.cumsum → np.searchsorted → q_values.append 追踪具体实现。
    # 输入接口：quantiles。
    # 返回值可从这里追踪：results。
    # 内部调用线索：np.cumsum → np.searchsorted → q_values.append → results.append → np.array（含分支中的调用，实际路径由条件决定）。
    def _compute_quantiles(self, quantiles):
        """Compute quantiles based on histograms."""
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                q_values.append(edges[idx])
            results.append(np.array(q_values))
        return results


# 【_NormStatsDict】维护流式均值方差和直方图分位数，并保存/读取NormStats。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
class _NormStatsDict(pydantic.BaseModel):
    # 字段含义：状态/动作的数值统计，需与训练变换对应。
    norm_stats: dict[str, NormStats]


# 【serialize_json】本函数位于“数据统计底层”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 _NormStatsDict(norm_stats=norm_stats).model_dump_json → _NormStatsDict 追踪具体实现。
# 输入接口：norm_stats:dict[str, NormStats]。
# 返回类型：str；类型/shape约定需与调用方配套。
# 内部调用线索：_NormStatsDict(norm_stats=norm_stats).model_dump_json → _NormStatsDict（含分支中的调用，实际路径由条件决定）。
def serialize_json(norm_stats: dict[str, NormStats]) -> str:
    """Serialize the running statistics to a JSON string."""
    return _NormStatsDict(norm_stats=norm_stats).model_dump_json(indent=2)


# 【deserialize_json】本函数位于“数据统计底层”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 _NormStatsDict → json.loads 追踪具体实现。
# 输入接口：data:str。
# 返回类型：dict[str, NormStats]；类型/shape约定需与调用方配套。
# 内部调用线索：_NormStatsDict → json.loads（含分支中的调用，实际路径由条件决定）。
def deserialize_json(data: str) -> dict[str, NormStats]:
    """Deserialize the running statistics from a JSON string."""
    return _NormStatsDict(**json.loads(data)).norm_stats


# 【save】把当前数据/状态写入指定存储；文件路径与覆盖行为由这里的实现决定。
# 输入接口：directory:pathlib.Path | str；norm_stats:dict[str, NormStats]。
# 返回类型：None；类型/shape约定需与调用方配套。
# 内部调用线索：pathlib.Path → path.parent.mkdir → path.write_text → serialize_json（含分支中的调用，实际路径由条件决定）。
def save(directory: pathlib.Path | str, norm_stats: dict[str, NormStats]) -> None:
    """Save the normalization stats to a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_json(norm_stats))


# 【load】读取或合并外部资源/参数；检查路径、结构与调用方期待是否一致。
# 输入接口：directory:pathlib.Path | str。
# 返回类型：dict[str, NormStats]；类型/shape约定需与调用方配套。
# 内部调用线索：pathlib.Path → path.exists → FileNotFoundError → deserialize_json → path.read_text（含分支中的调用，实际路径由条件决定）。
def load(directory: pathlib.Path | str) -> dict[str, NormStats]:
    """Load the normalization stats from a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {path}")
    return deserialize_json(path.read_text())
