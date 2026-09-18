# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：训练前准备｜遍历训练数据统计state/action均值、标准差与分位数，供训练和推理使用。
# 阅读顺序：根据配置建立未归一化数据管线，分batch更新RunningStats，再写入assets。
# 重点边界：统计必须基于与训练相同的动作表示；先做delta再统计与先统计绝对动作不同。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


# 【RemoveStrings】遍历训练数据统计state/action均值、标准差与分位数，供训练和推理使用。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
class RemoveStrings(transforms.DataTransformFn):
    # 【RemoveStrings.__call__】执行遍历训练数据统计state/action均值、标准差与分位数，供训练和推理使用。 本次调用的数据流见参数与返回值，内部调用顺序见下面代码。
    # 输入接口：x:dict。
    # 返回类型：dict；类型/shape约定需与调用方配套。
    # 内部调用线索：x.items → np.issubdtype → np.asarray（含分支中的调用，实际路径由条件决定）。
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


# 【create_torch_dataloader】本函数位于“训练前准备”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 ValueError → _data_loader.create_torch_dataset → _data_loader.TransformedDataset 追踪具体实现。
# 输入接口：data_config:_config.DataConfig；action_horizon:int；batch_size:int；model_config:_model.BaseModelConfig；num_workers:int；max_frames:int | None。
# 返回类型：tuple[_data_loader.Dataset, int]；类型/shape约定需与调用方配套。
# 内部调用线索：ValueError → _data_loader.create_torch_dataset → _data_loader.TransformedDataset → RemoveStrings → _data_loader.TorchDataLoader（含分支中的调用，实际路径由条件决定）。
def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


# 【create_rlds_dataloader】本函数位于“训练前准备”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 _data_loader.create_rlds_dataset → _data_loader.IterableTransformedDataset → RemoveStrings 追踪具体实现。
# 输入接口：data_config:_config.DataConfig；action_horizon:int；batch_size:int；max_frames:int | None。
# 返回类型：tuple[_data_loader.Dataset, int]；类型/shape约定需与调用方配套。
# 内部调用线索：_data_loader.create_rlds_dataset → _data_loader.IterableTransformedDataset → RemoveStrings → _data_loader.RLDSDataLoader（含分支中的调用，实际路径由条件决定）。
def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


# 【main】本脚本入口：遍历训练数据统计state/action均值、标准差与分位数，供训练和推理使用。 根据配置建立未归一化数据管线，分batch更新RunningStats，再写入assets。
# 输入接口：config_name:str；max_frames:int | None。
# 内部调用线索：_config.get_config → config.data.create → create_rlds_dataloader → create_torch_dataloader → normalize.RunningStats（含分支中的调用，实际路径由条件决定）。
def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
