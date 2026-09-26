# Pi_05

**Contributor:** RoboDojo Team | **Paper:** Pi0.5 technical report | **arXiv:** TBD | **Original code:** https://github.com/Physical-Intelligence/openpi

`Pi_05` adapts Physical Intelligence's π0.5 policy to XPolicyLab/RoboDojo through the uv-managed OpenPI stack. Integration scripts live at this directory level; the vendored upstream implementation lives in `openpi/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

```bash
cd XPolicyLab/policy/Pi_05
bash install.sh
source openpi/.venv/bin/activate  # OpenPI is uv-managed; there is no policy conda env
```

`eval.sh` arg 9 is not a conda env: pass `uv` (uses `deploy.yml` `policy_uv_env_path`) or an explicit OpenPI project path.

## Data Processing

Converts RoboDojo demonstrations into the LeRobot repo consumed by training. The optional `expert_data_num` caps episodes for data conversion only (it is not part of checkpoint naming); the optional `raw_task_dirs` is a source task directory or comma-separated task list under `data/<bench_name>/` (defaults to `ckpt_name`). `raw_task_dirs` may also be passed directly as the 5th argument to write a differently named dataset from all of a task's demos, e.g. `bash process_data.sh RoboDojo stack_bowls_ablation arx_x5 joint stack_bowls`.

```bash
cd XPolicyLab/policy/Pi_05
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [raw_task_dirs]

# Example: convert stack_bowls demos for arx_x5 joint control
bash process_data.sh RoboDojo stack_bowls arx_x5 joint

# Example: create a 50-episode ablation while reading from the original task data
bash process_data.sh RoboDojo stack_bowls_50ep arx_x5 joint 50 stack_bowls
```

## Training

```bash
cd XPolicyLab/policy/Pi_05
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

# Example: train a cotrain run on GPU 0 (comma-separated gpu_id for multi-GPU)
bash train.sh RoboDojo cotrain arx_x5 joint 0 0
```

Checkpoints land in `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`; at eval time `ckpt_name` may be the short run name (auto-combined into that directory name), the full run-directory name, or a path to a checkpoint directory. By default training reads the LeRobot repo produced by `process_data.sh` (`<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>`); override with `OPENPI_LEROBOT_REPO_ID` when reusing an existing dataset. `train.sh` sets `fsdp_devices=1` for one visible GPU and `2` for multi-GPU by default (override with `OPENPI_FSDP_DEVICES`).

### Restaurant Pi0.5 fine-tuning modes

The restaurant launcher supports three named configs. They share the same 32-D action, 50-step horizon, dual-Franka input transforms, dataset and normalization asset contract. Set `OPENPI_TRAIN_CONFIG_NAME` before invoking the launcher; its default remains full fine-tuning.

| Config | Trainable parameters | Scope |
|---|---:|---|
| `pi05_restaurant_franka_full_finetune` | 3,353,433,872 / 3,353,433,872 | All model parameters; existing default. |
| `pi05_restaurant_franka_lora` | 52,153,376 / 3,403,421,456 | 27,869,184 language LoRA, 22,118,400 action-expert LoRA, and 2,165,792 non-LoRA action/time projections. Vision, embeddings, and both experts' base weights are frozen. |
| `pi05_restaurant_franka_action_head` | 430,098,464 / 3,353,433,872 | The complete action expert (427,932,672, including its adaRMSNorm conditioning and final norm) and 2,165,792 action/time projections. Vision, embeddings and the language expert are frozen. |

The action expert is the `_1` family under `PaliGemma/llm`: `q_einsum_1`, `kv_einsum_1`, `attn_vec_einsum_1`, `mlp_1`, `pre_attention_norm_1`, `pre_ffw_norm_1`, and `final_norm_1`. The four other trainable modules are `action_in_proj`, `action_out_proj`, `time_mlp_in`, and `time_mlp_out`. `PaliGemma/llm` also contains the frozen language expert, so freezing or enabling the whole `llm` tree is incorrect.

The LoRA config uses the vendored Gemma attention (`q`, `kv`, output) and FFN adapters in both experts. Defaults are rank/alpha 16/16 for the language expert and 32/32 for the action expert (scale `alpha/rank=1`). Each adapter's A matrix starts random and B starts at zero, so a base checkpoint produces the same initial output. Override ranks and alphas through OpenPI CLI flags such as `--model.paligemma-lora-rank=8` and `--model.action-expert-lora-alpha=16`; changing them requires a new run, not `--resume`.

```bash
# On the configured GPU host, with the existing dataset, assets and venv staged by this launcher:
OPENPI_TRAIN_CONFIG_NAME=pi05_restaurant_franka_full_finetune bash train_restaurant_franka.sh 1
OPENPI_TRAIN_CONFIG_NAME=pi05_restaurant_franka_lora bash train_restaurant_franka.sh 1
OPENPI_TRAIN_CONFIG_NAME=pi05_restaurant_franka_action_head bash train_restaurant_franka.sh 1

# Initialize a new run from a compatible full Pi0.5 checkpoint's params item:
OPENPI_TRAIN_CONFIG_NAME=pi05_restaurant_franka_lora \
  OPENPI_INIT_PARAMS_SOURCE=/path/to/full-run/<step>/params bash train_restaurant_franka.sh 1

# Resume the same run with its saved optimizer state, step and parameters:
OPENPI_TRAIN_CONFIG_NAME=pi05_restaurant_franka_lora OPENPI_TRAIN_RESUME=1 \
  bash train_restaurant_franka.sh 1
```

The full config keeps its historical run-directory name; LoRA and action-head runs append `-lora` and `-action_head`, respectively. The launcher stages `OPENPI_INIT_PARAMS_SOURCE` (default: the base Pi0.5 `params` item) onto node-local storage, and `OPENPI_TRAIN_RESUME=1` also stages the matching saved run. Each checkpoint step saves **full** inference parameters under `<run>/<step>/params`, training state (including optimizer state) under `train_state`, and normalization statistics under `assets/<repo_id>`. There is no adapter-only checkpoint export. Initializing LoRA from base or a full fine-tuned checkpoint is supported; initializing action-head-only from either is supported; initializing from an existing LoRA checkpoint requires the same LoRA topology. Extra checkpoint keys, partial LoRA adapters and shape mismatches fail instead of being silently ignored. Resume requires the same config, LoRA topology, optimizer, seed, global batch size and selected parameter tree; the run's `training_contract.json` enforces this for new modes. Inference must use the same `train_config_name` and the saved checkpoint step, for example `python openpi/scripts/serve_policy.py policy:checkpoint --policy.config=pi05_restaurant_franka_lora --policy.dir=/path/to/run/<step>`; `model.py` can also select that config through `train_config_name` and the checkpoint path.

If a LoRA or action-head run already exists at the persistent checkpoint path, the restaurant launcher requires `OPENPI_TRAIN_RESUME=1` or a different `OPENPI_CHECKPOINT_ROOT`, protecting the saved optimizer state from an accidental fresh run.

The existing `pi05_restaurant_franka_lora_smoke` remains a one-step diagnostic with its original broader trainable set (466,957,072 trainable parameters, including the vision tower); use `pi05_restaurant_franka_lora` for experiments. CPU tests validate parameter selection, optimizer updates, full-parameter checkpoint save/restore and strict inference loading. GPU memory, throughput and policy quality remain untested for the new modes.

These modes do not alter image conversion or policy input color order. Use a dataset and inference input with the matching color-order contract from the runtime integration; a separate official-runtime task owns the RGB/BGR correction. The normalization asset source and the `assets/<repo_id>` checkpoint layout are shared across all three modes.

## Evaluation

```bash
cd XPolicyLab/policy/Pi_05
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_uv_env> <eval_env_conda_env>

# Example: evaluate a trained cotrain checkpoint on stack_bowls
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 0 0 uv <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Configuration

`deploy.yml` keys to check before evaluation: `checkpoint_num`, `result_dir`, `obs_transform_pipeline`, `policy_uv_env_path`, `train_config_name` (must match the config used by `train.sh`), `repo_id`.

Environment variables used by the adapter scripts:

| Variable | Notes |
|---|---|
| `OPENPI_LEROBOT_REPO_ID` | Overrides the LeRobot repo id used by `train.sh`; defaults to `<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>`. |
| `OPENPI_FSDP_DEVICES` | Overrides the FSDP device count passed to OpenPI training. |
| `OPENPI_TRAIN_CONFIG_NAME` | Overrides the training config; defaults to `pi05_base_aloha_full_sim_arx-x5_seed_0`. |
| `OPENPI_INIT_PARAMS_SOURCE` | Restaurant launcher initialization `params` directory; defaults to the staged Pi0.5 base. |
| `OPENPI_TRAIN_RESUME` | Set to `1` to stage and resume the matching restaurant run instead of overwriting it. |
| `OPENPI_DATA_MODE` | Data-processing mode passed to `openpi/scripts/process_data.py`; defaults to `image`. |
| `OPENPI_LOCAL_CACHE_ROOT` | Per-host local cache root for the HF datasets / JAX compilation caches; defaults to `/tmp/openpi-cache-$(hostname)`. |

`OPENPI_ROOT` and `OPENPI_SRC` are additional overrides consumed by the local scripts.
