"""Regression tests for the restaurant Pi0.5 fine-tuning modes."""

import dataclasses
from types import SimpleNamespace

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.training import checkpoints
from openpi.training import config
from openpi.training import finetune
from openpi.training import optimizer
from openpi.training import utils
from openpi.training import weight_loaders


def _selected(filter_, path: str) -> bool:
    return filter_(tuple(path.split("/")), nnx.Param(np.zeros(())))


def test_restaurant_modes_select_the_expected_parameter_families() -> None:
    full = config.get_config("pi05_restaurant_franka_full_finetune")
    lora = config.get_config("pi05_restaurant_franka_lora")
    action = config.get_config("pi05_restaurant_franka_action_head")
    assert all(candidate.weight_loader.strict for candidate in (full, lora, action))
    for candidate in (lora, action):
        assert candidate.data.repo_id == full.data.repo_id
        assert candidate.data.assets == full.data.assets
        assert candidate.policy_metadata == full.policy_metadata

    paths = {
        "vision": "PaliGemma/img/params/encoder/kernel",
        "language": "PaliGemma/llm/params/layers/attn/q_einsum/w",
        "embedding": "PaliGemma/llm/params/embedder/input_embedding",
        "expert": "PaliGemma/llm/params/layers/attn/q_einsum_1/w",
        "expert_norm": "PaliGemma/llm/params/layers/pre_attention_norm_1/Dense_0/kernel",
        "language_lora": "PaliGemma/llm/params/layers/attn/q_einsum/lora_a",
        "expert_lora": "PaliGemma/llm/params/layers/attn/q_einsum_1/lora_a",
        "action_input": "action_in_proj/kernel",
        "action_output": "action_out_proj/kernel",
        "time_input": "time_mlp_in/kernel",
        "time_output": "time_mlp_out/kernel",
    }
    assert isinstance(full.freeze_filter, nnx.Nothing)
    assert {_ for _, path in paths.items() if _selected(lora.trainable_filter, path)} == {
        "language_lora",
        "expert_lora",
        "action_input",
        "action_output",
        "time_input",
        "time_output",
    }
    assert {_ for _, path in paths.items() if _selected(action.trainable_filter, path)} == {
        "expert",
        "expert_norm",
        "action_input",
        "action_output",
        "time_input",
        "time_output",
    }
    assert lora.model.paligemma_variant == "gemma_2b_lora"
    assert lora.model.action_expert_variant == "gemma_300m_lora"
    assert (lora.model.paligemma_lora_rank, lora.model.paligemma_lora_alpha) == (16, 16.0)
    assert (lora.model.action_expert_lora_rank, lora.model.action_expert_lora_alpha) == (32, 32.0)


def test_lora_options_reach_attention_and_ffn_modules() -> None:
    model = config.get_config("pi05_restaurant_franka_lora").model
    paligemma, expert = model.gemma_configs()
    for target in ("attn", "ffn"):
        assert (paligemma.lora_configs[target].rank, paligemma.lora_configs[target].alpha) == (16, 16.0)
        assert (expert.lora_configs[target].rank, expert.lora_configs[target].alpha) == (32, 32.0)


def test_checkpoint_loader_rejects_unexpected_adapter_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(
        _model,
        "restore_params",
        lambda *_args, **_kwargs: {"a": np.ones((2,)), "adapter_lora_a": np.ones((2,))},
    )
    with pytest.raises(ValueError, match="unexpected.*adapter_lora_a"):
        weight_loaders.CheckpointWeightLoader("checkpoint/params", strict=True).load({"a": np.zeros((2,))})


def test_action_side_filter_rejects_language_and_vision() -> None:
    selector = finetune.action_side_filter()
    assert _selected(selector, "PaliGemma/llm/final_norm_1/Dense_0/kernel")
    assert _selected(selector, "PaliGemma/llm/params/layers/mlp_1/linear")
    assert not _selected(selector, "PaliGemma/llm/params/layers/mlp/linear")
    assert not _selected(selector, "PaliGemma/img/params/encoder/kernel")


def test_real_pi05_parameter_trees_have_expected_trainable_counts() -> None:
    expected = {
        "pi05_restaurant_franka_full_finetune": (3_353_433_872, 3_353_433_872),
        "pi05_restaurant_franka_lora": (3_403_421_456, 52_153_376),
        "pi05_restaurant_franka_action_head": (3_353_433_872, 430_098_464),
    }
    for name, (expected_total, expected_trainable) in expected.items():
        train_config = config.get_config(name)
        assert isinstance(train_config.model, pi0_config.Pi0Config)
        state = nnx.state(nnx.eval_shape(train_config.model.create, jax.random.key(0)))
        total = sum(param.value.size for param in state.flat_state().values())
        selected = state.filter(train_config.trainable_filter).flat_state()
        trainable = sum(param.value.size for param in selected.values())
        assert (total, trainable) == (expected_total, expected_trainable)
        if name != "pi05_restaurant_franka_full_finetune":
            assert not any(path[0:2] == ("PaliGemma", "img") for path in selected)
            assert not any("embedder" in path for path in selected)
        projection_paths = {
            path
            for path in state.flat_state()
            if path[0] in {"action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"}
        }
        if name == "pi05_restaurant_franka_lora":
            adapter_paths = {
                path for path in state.flat_state() if path[:2] == ("PaliGemma", "llm") and "lora" in str(path[-1])
            }
            assert selected.keys() == adapter_paths | projection_paths
        if name == "pi05_restaurant_franka_action_head":
            action_expert_paths = {
                path
                for path in state.flat_state()
                if path[:2] == ("PaliGemma", "llm") and any(str(segment).endswith("_1") for segment in path)
            }
            assert selected.keys() == action_expert_paths | projection_paths


class _TinyPi(_model.BaseModel):
    def __init__(self):
        super().__init__(1, 1, 1)
        self.PaliGemma = nnx.Dict(
            llm=nnx.Dict(
                params=nnx.Dict(
                    layers=nnx.Dict(
                        mlp=nnx.Dict(linear=nnx.Param(jnp.array(2.0))),
                        mlp_1=nnx.Dict(linear=nnx.Param(jnp.array(3.0))),
                    )
                )
            )
        )
        self.action_in_proj = nnx.Dict(kernel=nnx.Param(jnp.array(4.0)))

    def train(self):
        return self

    def compute_loss(self, rng, observation, actions, *, train=False):
        del rng, observation, train
        language = self.PaliGemma.llm.params.layers.mlp.linear.value
        expert = self.PaliGemma.llm.params.layers.mlp_1.linear.value
        action = self.action_in_proj.kernel.value
        return (language * expert * action - actions) ** 2

    def sample_actions(self, rng, observation, **kwargs):
        raise NotImplementedError


class _TinyPiLora(_TinyPi):
    def __init__(self):
        super().__init__()
        self.PaliGemma = nnx.Dict(
            llm=nnx.Dict(
                params=nnx.Dict(
                    layers=nnx.Dict(
                        mlp=nnx.Dict(
                            linear=nnx.Param(jnp.array(2.0)),
                            lora_a=nnx.Param(jnp.array(1.0)),
                            lora_b=nnx.Param(jnp.array(0.0)),
                        ),
                        mlp_1=nnx.Dict(
                            linear=nnx.Param(jnp.array(3.0)),
                            lora_a=nnx.Param(jnp.array(1.0)),
                            lora_b=nnx.Param(jnp.array(0.0)),
                        ),
                    )
                )
            )
        )

    def compute_loss(self, rng, observation, actions, *, train=False):
        del rng, observation, train
        layers = self.PaliGemma.llm.params.layers
        language = layers.mlp.linear.value + layers.mlp.lora_a.value * layers.mlp.lora_b.value
        expert = layers.mlp_1.linear.value + layers.mlp_1.lora_a.value * layers.mlp_1.lora_b.value
        return (language * expert * self.action_in_proj.kernel.value - actions) ** 2


@dataclasses.dataclass(frozen=True)
class _TinyConfig(_model.BaseModelConfig):
    @property
    def model_type(self):
        return _model.ModelType.PI05

    def create(self, rng):
        del rng
        return _TinyPi()

    def inputs_spec(self, *, batch_size=1):
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class _TinyLoraConfig(_TinyConfig):
    def create(self, rng):
        del rng
        return _TinyPiLora()


def test_action_head_real_optimizer_step_freezes_language_and_updates_action() -> None:
    from scripts import train as train_script

    model = _TinyPi()
    graphdef, params = nnx.split(model)
    train_config = dataclasses.replace(config.get_config("pi05_restaurant_franka_action_head"), ema_decay=None)
    tx = optimizer.create_optimizer(optimizer.AdamW(weight_decay=0), optimizer.CosineDecaySchedule(warmup_steps=0))
    state = utils.TrainState(
        step=0,
        params=params,
        model_def=graphdef,
        tx=tx,
        opt_state=tx.init(params.filter(train_config.trainable_filter)),
        ema_decay=None,
        ema_params=None,
    )
    observation = _model.Observation(images={}, image_masks={}, state=jnp.zeros((1, 1)))
    next_state, _ = train_script.train_step(train_config, jax.random.key(0), state, (observation, jnp.ones((1, 1, 1))))
    before = params.to_pure_dict()
    after = next_state.params.to_pure_dict()
    assert jnp.array_equal(
        before["PaliGemma"]["llm"]["params"]["layers"]["mlp"]["linear"],
        after["PaliGemma"]["llm"]["params"]["layers"]["mlp"]["linear"],
    )
    assert not jnp.array_equal(
        before["PaliGemma"]["llm"]["params"]["layers"]["mlp_1"]["linear"],
        after["PaliGemma"]["llm"]["params"]["layers"]["mlp_1"]["linear"],
    )
    assert not jnp.array_equal(before["action_in_proj"]["kernel"], after["action_in_proj"]["kernel"])


def test_lora_real_optimizer_step_updates_adapters_but_not_base_weights() -> None:
    from scripts import train as train_script

    model = _TinyPiLora()
    graphdef, params = nnx.split(model)
    train_config = config.get_config("pi05_restaurant_franka_lora")
    tx = optax.adamw(1e-3)
    state = utils.TrainState(
        step=0,
        params=params,
        model_def=graphdef,
        tx=tx,
        opt_state=tx.init(params.filter(train_config.trainable_filter)),
        ema_decay=None,
        ema_params=None,
    )
    observation = _model.Observation(images={}, image_masks={}, state=jnp.zeros((1, 1)))
    next_state, _ = train_script.train_step(train_config, jax.random.key(0), state, (observation, jnp.ones((1, 1, 1))))
    before = params.to_pure_dict()
    after = next_state.params.to_pure_dict()
    for expert in ("mlp", "mlp_1"):
        base_path = before["PaliGemma"]["llm"]["params"]["layers"][expert]
        updated_path = after["PaliGemma"]["llm"]["params"]["layers"][expert]
        assert jnp.array_equal(base_path["linear"], updated_path["linear"])
        assert not jnp.array_equal(base_path["lora_b"], updated_path["lora_b"])
    assert not jnp.array_equal(before["action_in_proj"]["kernel"], after["action_in_proj"]["kernel"])


def test_resume_contract_rejects_another_parameter_mask(tmp_path) -> None:
    params = nnx.state(_TinyPi())
    full = config.get_config("pi05_restaurant_franka_full_finetune")
    action = config.get_config("pi05_restaurant_franka_action_head")
    finetune.check_training_contract(tmp_path, full, params, resuming=False)
    with pytest.raises(ValueError, match="training contract"):
        finetune.check_training_contract(tmp_path, action, params, resuming=True)


def test_resume_contract_rejects_different_dataset_identity(tmp_path) -> None:
    params = nnx.state(_TinyPi())
    action = config.get_config("pi05_restaurant_franka_action_head")
    finetune.check_training_contract(tmp_path, action, params, resuming=False)
    other_data = dataclasses.replace(action.data, repo_id="openskillbench/different_dataset")
    with pytest.raises(ValueError, match="training contract"):
        finetune.check_training_contract(tmp_path, dataclasses.replace(action, data=other_data), params, resuming=True)


def test_checkpoint_roundtrip_keeps_model_and_optimizer_state(tmp_path) -> None:
    model = _TinyPiLora()
    graphdef, params = nnx.split(model)
    train_config = config.get_config("pi05_restaurant_franka_lora")
    tx = optax.adamw(1e-3)
    trainable = params.filter(train_config.trainable_filter)
    updates, opt_state = tx.update(jax.tree.map(jnp.ones_like, trainable), tx.init(trainable), trainable)
    updated = optax.apply_updates(trainable, updates)
    nnx.update(model, updated)
    state = utils.TrainState(
        step=1,
        params=nnx.state(model),
        model_def=graphdef,
        tx=tx,
        opt_state=opt_state,
        ema_decay=None,
        ema_params=None,
    )

    class _NoAssets:
        def data_config(self):
            return SimpleNamespace(norm_stats=None, asset_id=None)

    data_loader = _NoAssets()
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        tmp_path / "run", keep_period=1, overwrite=False, resume=False
    )
    assert not resuming
    finetune.check_training_contract(tmp_path / "run", train_config, state.params, resuming=False)
    checkpoints.save_state(manager, state, data_loader, 1)
    manager.wait_until_finished()
    finetune.check_training_contract(tmp_path / "run", train_config, state.params, resuming=True)
    restored = checkpoints.restore_state(manager, state, data_loader, step=1)
    assert int(restored.step) == 1
    for expected, actual in zip(jax.tree.leaves(state.params), jax.tree.leaves(restored.params), strict=True):
        np.testing.assert_array_equal(expected, actual)
    for expected, actual in zip(jax.tree.leaves(state.opt_state), jax.tree.leaves(restored.opt_state), strict=True):
        np.testing.assert_array_equal(expected, actual)
    inference_params = _model.restore_params(tmp_path / "run" / "1" / "params", restore_type=np.ndarray)
    assert inference_params.keys() == state.params.to_pure_dict().keys()
    inference_model = _TinyLoraConfig(1, 1, 1).load(inference_params, remove_extra_params=False)
    np.testing.assert_array_equal(
        nnx.state(inference_model).to_pure_dict()["action_in_proj"]["kernel"],
        state.params.to_pure_dict()["action_in_proj"]["kernel"],
    )
    with pytest.raises(ValueError, match="different structure"):
        _TinyLoraConfig(1, 1, 1).load({**inference_params, "unexpected": np.ones((1,))}, remove_extra_params=False)
    manager.close()
