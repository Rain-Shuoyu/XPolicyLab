#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_ROOT="${OPENPI_SHARED_ROOT:-/mnt/afs/L202500576}"
TRAIN_ROOT="${OPENPI_TRAIN_ROOT:-${SHARED_ROOT}/training/pi05_restaurant}"
LOG_ROOT="${OPENPI_LOG_ROOT:-${TRAIN_ROOT}/logs}"

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${SHARED_ROOT}/datasets/lerobot}"
export OPENPI_LEROBOT_REPO_ID="${OPENPI_LEROBOT_REPO_ID:-openskillbench/restaurant_pass_counter_franka_dense50}"
export OPENPI_ASSETS_ROOT="${OPENPI_ASSETS_ROOT:-${TRAIN_ROOT}/assets}"
export OPENPI_CHECKPOINT_ROOT="${OPENPI_SMOKE_CHECKPOINT_ROOT:-${TRAIN_ROOT}/smoke_checkpoints}"
export OPENPI_TRAIN_CONFIG_NAME=pi05_restaurant_franka_lora_smoke
export OPENPI_FSDP_DEVICES=1
export OPENPI_VENV="${OPENPI_VENV:-${SHARED_ROOT}/venvs/pi05-openpi}"
export OPENPI_UV_BIN="${OPENPI_UV_BIN:-${SHARED_ROOT}/bin/uv}"

mkdir -p "${LOG_ROOT}"
exec > >(tee -a "${LOG_ROOT}/smoke_restaurant_franka.log") 2>&1
exec bash "${POLICY_DIR}/train.sh" restaurant_pass_counter franka_dense50 franka joint 0 0
