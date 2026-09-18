# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：可跳过｜RoboArena相关实验和基线配置注册表。
# 阅读顺序：本轮不需要学习其中各类动作tokenizer基线。
# 重点边界：被config.py导入不代表当前选择的实验会运行这些模型。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
"""RoboArena baseline policy configs."""

from typing import TypeAlias

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.droid_policy as droid_policy
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType


# 【get_roboarena_configs】此符号属于π0/π0.5主线以外的上游分支。本轮可跳过；保留原实现和英文说明供依赖完整性及将来查询。
def get_roboarena_configs():
    # Import here to avoid circular imports.
    from openpi.training.config import AssetsConfig
    from openpi.training.config import DataConfig
    from openpi.training.config import SimpleDataConfig
    from openpi.training.config import TrainConfig

    return [
        #
        # RoboArena DROID baseline inference configs.
        #
        TrainConfig(
            # Trained from PaliGemma, using RT-2 / OpenVLA style binning tokenizer.
            name="paligemma_binning_droid",
            model=pi0_fast.Pi0FASTConfig(
                action_dim=8,
                action_horizon=15,
                max_token_len=400,
                fast_model_tokenizer=_tokenizer.BinningTokenizer,
            ),
            data=SimpleDataConfig(
                assets=AssetsConfig(asset_id="droid"),
                data_transforms=lambda model: _transforms.Group(
                    inputs=[droid_policy.DroidInputs(action_dim=model.action_dim, model_type=ModelType.PI0_FAST)],
                    outputs=[droid_policy.DroidOutputs()],
                ),
                base_config=DataConfig(
                    prompt_from_task=True,
                ),
            ),
        ),
        TrainConfig(
            # Trained from PaliGemma, using FAST tokenizer (using universal FAST+ tokenizer).
            name="paligemma_fast_droid",
            model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=15),
            data=SimpleDataConfig(
                assets=AssetsConfig(asset_id="droid"),
                data_transforms=lambda model: _transforms.Group(
                    inputs=[droid_policy.DroidInputs(action_dim=model.action_dim, model_type=ModelType.PI0_FAST)],
                    outputs=[droid_policy.DroidOutputs()],
                ),
                base_config=DataConfig(
                    prompt_from_task=True,
                ),
            ),
        ),
        TrainConfig(
            # Trained from PaliGemma, using FAST tokenizer (tokenizer trained on DROID dataset).
            name="paligemma_fast_specialist_droid",
            model=pi0_fast.Pi0FASTConfig(
                action_dim=8,
                action_horizon=15,
                fast_model_tokenizer=_tokenizer.FASTTokenizer,
                fast_model_tokenizer_kwargs={"fast_tokenizer_path": "KarlP/fast_droid_specialist"},
            ),
            data=SimpleDataConfig(
                assets=AssetsConfig(asset_id="droid"),
                data_transforms=lambda model: _transforms.Group(
                    inputs=[droid_policy.DroidInputs(action_dim=model.action_dim, model_type=ModelType.PI0_FAST)],
                    outputs=[droid_policy.DroidOutputs()],
                ),
                base_config=DataConfig(
                    prompt_from_task=True,
                ),
            ),
        ),
        TrainConfig(
            # Trained from PaliGemma, using FSQ tokenizer.
            name="paligemma_vq_droid",
            model=pi0_fast.Pi0FASTConfig(
                action_dim=8,
                action_horizon=15,
                fast_model_tokenizer=_tokenizer.FSQTokenizer,
                fast_model_tokenizer_kwargs={"fsq_tokenizer_path": "gs://openpi-assets/tokenizers/droid_fsq_tokenizer"},
            ),
            data=SimpleDataConfig(
                assets=AssetsConfig(asset_id="droid"),
                data_transforms=lambda model: _transforms.Group(
                    inputs=[droid_policy.DroidInputs(action_dim=model.action_dim, model_type=ModelType.PI0_FAST)],
                    outputs=[droid_policy.DroidOutputs()],
                ),
                base_config=DataConfig(
                    prompt_from_task=True,
                ),
            ),
        ),
        TrainConfig(
            # pi0-style diffusion / flow VLA, trained on DROID from PaliGemma.
            name="paligemma_diffusion_droid",
            model=pi0_config.Pi0Config(action_horizon=10, action_dim=8),
            data=SimpleDataConfig(
                assets=AssetsConfig(asset_id="droid"),
                data_transforms=lambda model: _transforms.Group(
                    inputs=[droid_policy.DroidInputs(action_dim=model.action_dim)],
                    outputs=[droid_policy.DroidOutputs()],
                ),
                base_config=DataConfig(
                    prompt_from_task=True,
                ),
            ),
        ),
    ]
