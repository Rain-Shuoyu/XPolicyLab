"""Explicit Pi0.5 parameter selections and resume compatibility checks."""

import dataclasses
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import flax.nnx as nnx

from openpi.shared import nnx_utils

if TYPE_CHECKING:
    from openpi.training.config import TrainConfig


_ACTION_PROJECTIONS = nnx_utils.PathRegex(r"(?:action_in_proj|action_out_proj|time_mlp_in|time_mlp_out)/.+")
_ACTION_EXPERT = nnx_utils.PathRegex(
    r"PaliGemma/llm/(?:.*/)?(?:q_einsum_1|kv_einsum_1|qkv_einsum_1|attn_vec_einsum_1|"
    r"mlp_1|pre_attention_norm_1|pre_ffw_norm_1|final_norm_1)/.+"
)
_LLM_LORA = nnx_utils.PathRegex(r"PaliGemma/llm/.*/[^/]*lora[^/]*")


def action_side_filter() -> nnx.filterlib.Filter:
    """Select the action expert and Pi0.5 action/time projections, excluding PaliGemma's main expert."""
    return nnx.Any(nnx.All(_ACTION_EXPERT, nnx.Not(_LLM_LORA)), _ACTION_PROJECTIONS)


def lora_trainable_filter() -> nnx.filterlib.Filter:
    """Select Gemma adapters and the four non-LoRA action/time projections."""
    return nnx.Any(_LLM_LORA, _ACTION_PROJECTIONS)


def check_training_contract(checkpoint_dir: Path, config: "TrainConfig", params: nnx.State, *, resuming: bool) -> None:
    """Reject a resume when the optimizer's parameter tree or training topology changed."""
    path = checkpoint_dir / "training_contract.json"
    trainable = params.filter(config.trainable_filter).flat_state()
    contract = {
        "schema_version": 2,
        "config_name": config.name,
        "seed": config.seed,
        "batch_size": config.batch_size,
        "model": dataclasses.asdict(config.model),
        "data_repo_id": getattr(config.data, "repo_id", None),
        "data_asset_id": getattr(getattr(config.data, "assets", None), "asset_id", None),
        "policy_metadata": config.policy_metadata,
        "optimizer": dataclasses.asdict(config.optimizer),
        "lr_schedule": dataclasses.asdict(config.lr_schedule),
        "ema_decay": config.ema_decay,
        "trainable_shapes": {"/".join(map(str, key)): list(value.value.shape) for key, value in trainable.items()},
    }
    if resuming:
        if not path.exists():
            if config.name == "pi05_restaurant_franka_full_finetune":
                logging.warning(
                    "Legacy full-FT checkpoint has no training_contract.json; Orbax tree validation applies"
                )
                return
            raise ValueError(f"Missing training contract at {path}; cannot safely resume this fine-tuning mode")
        existing = json.loads(path.read_text())
        if existing != contract:
            raise ValueError(
                f"Incompatible training contract at {path}; use the original config for resume or initialize a new run"
            )
        return
    path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
