from pathlib import Path

import flax.nnx as nnx

from openpi.models import pi0_config
from openpi.policies import aloha_policy
from openpi.training import config
from openpi.training import weight_loaders
from openpi import transforms


def _created_data_config(train_config: config.TrainConfig) -> config.DataConfig:
    return train_config.data.create(
        Path("/tmp/restaurant-pi05-assets"), train_config.model
    )


def test_restaurant_full_config_uses_pi05_base_and_dual_franka_contract() -> None:
    train_config = config.get_config("pi05_restaurant_franka_full_finetune")
    data_config = _created_data_config(train_config)

    assert isinstance(train_config.model, pi0_config.Pi0Config)
    assert train_config.model.pi05 is True
    assert train_config.model.action_dim == 32
    assert train_config.model.action_horizon == 50
    assert train_config.batch_size == 256
    assert isinstance(train_config.freeze_filter, nnx.Nothing)
    assert isinstance(train_config.weight_loader, weight_loaders.CheckpointWeightLoader)
    assert (
        train_config.weight_loader.params_path
        == "gs://openpi-assets/checkpoints/pi05_base/params"
    )
    assert (
        data_config.repo_id == "openskillbench/restaurant_pass_counter_franka_dense50"
    )
    assert data_config.prompt_from_task is True
    assert data_config.use_quantile_norm is True

    assert isinstance(data_config.data_transforms.inputs[0], aloha_policy.AlohaInputs)
    assert data_config.data_transforms.inputs[0].adapt_to_pi is False
    delta = data_config.data_transforms.inputs[1]
    assert isinstance(delta, transforms.DeltaActions)
    assert tuple(delta.mask) == (True,) * 7 + (False,) + (True,) * 7 + (False,)
    output = data_config.data_transforms.outputs[-1]
    assert isinstance(output, aloha_policy.AlohaOutputs)
    assert output.action_dim == 16


def test_restaurant_smoke_config_is_one_step_lora_without_ema_or_wandb() -> None:
    train_config = config.get_config("pi05_restaurant_franka_lora_smoke")

    assert train_config.model.pi05 is True
    assert train_config.model.action_dim == 32
    assert "lora" in train_config.model.paligemma_variant
    assert "lora" in train_config.model.action_expert_variant
    assert not isinstance(train_config.freeze_filter, nnx.Nothing)
    assert train_config.batch_size == 1
    assert train_config.num_train_steps == 1
    assert train_config.fsdp_devices == 1
    assert train_config.ema_decay is None
    assert train_config.wandb_enabled is False
