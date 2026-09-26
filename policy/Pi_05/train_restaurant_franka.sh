#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -ne 1 ]]; then
  echo "Expected GPU count to be one of: 1, 2, 4, 8" >&2
  exit 2
fi
GPU_COUNT=$1
case "${GPU_COUNT}" in
  1|2|4|8) ;;
  *)
    echo "Expected GPU count to be one of: 1, 2, 4, 8" >&2
    exit 2
    ;;
esac

VISIBLE_GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
if (( GPU_COUNT > VISIBLE_GPU_COUNT )); then
  echo "Requested ${GPU_COUNT} GPUs, but only ${VISIBLE_GPU_COUNT} are visible" >&2
  exit 2
fi

GPU_IDS=
for ((gpu_id = 0; gpu_id < GPU_COUNT; gpu_id++)); do
  if [[ -n "${GPU_IDS}" ]]; then
    GPU_IDS+=,
  fi
  GPU_IDS+="${gpu_id}"
done

SHARED_ROOT="${OPENPI_SHARED_ROOT:-/mnt/afs/L202500576}"
TRAIN_ROOT="${OPENPI_TRAIN_ROOT:-${SHARED_ROOT}/training/pi05_restaurant}"
PERSISTENT_LOG_ROOT="${OPENPI_LOG_ROOT:-${TRAIN_ROOT}/logs}"
PERSISTENT_CHECKPOINT_ROOT="${OPENPI_CHECKPOINT_ROOT:-${TRAIN_ROOT}/checkpoints}"
PERSISTENT_LEROBOT_HOME="${HF_LEROBOT_HOME:-${SHARED_ROOT}/datasets/lerobot}"
BASE_MODEL_SOURCE="${OPENPI_BASE_MODEL_SOURCE:-${SHARED_ROOT}/openpi-cache/huggingface/robotgeneralist-openpi_checkpoint_mirrors2/pi05_base}"
INIT_PARAMS_SOURCE="${OPENPI_INIT_PARAMS_SOURCE:-${BASE_MODEL_SOURCE}/params}"
ENV_ARCHIVE="${OPENPI_ENV_ARCHIVE:-${SHARED_ROOT}/environments/pi05-openpi.tar}"
LOCAL_ROOT="${OPENPI_NODE_LOCAL_ROOT:-$(mktemp -d "${TMPDIR:-/tmp}/pi05-restaurant-franka.XXXXXX")}"
LOCAL_LEROBOT_HOME="${LOCAL_ROOT}/lerobot"
LOCAL_BASE_MODEL="${LOCAL_ROOT}/base/pi05_base"
LOCAL_ENVIRONMENT_ROOT="${LOCAL_ROOT}/environment"
LOCAL_VENV="${LOCAL_ENVIRONMENT_ROOT}/pi05-openpi"
LOCAL_CHECKPOINT_ROOT="${LOCAL_ROOT}/checkpoints"
LOCAL_LOG_ROOT="${LOCAL_ROOT}/logs"
LOCAL_LOG="${LOCAL_LOG_ROOT}/train_restaurant_franka.log"

export OPENPI_LEROBOT_REPO_ID="${OPENPI_LEROBOT_REPO_ID:-openskillbench/restaurant_pass_counter_franka_atomic205_recovery5}"
export OPENPI_ASSETS_ROOT="${OPENPI_ASSETS_ROOT:-${TRAIN_ROOT}/assets}"
export OPENPI_TRAIN_CONFIG_NAME="${OPENPI_TRAIN_CONFIG_NAME:-pi05_restaurant_franka_full_finetune}"
case "${OPENPI_TRAIN_CONFIG_NAME}" in
  pi05_restaurant_franka_full_finetune|pi05_restaurant_franka_lora|pi05_restaurant_franka_action_head) ;;
  *) echo "Unsupported restaurant train config: ${OPENPI_TRAIN_CONFIG_NAME}" >&2; exit 2 ;;
esac
case "${OPENPI_TRAIN_RESUME:-0}" in
  0|1) ;;
  *) echo "OPENPI_TRAIN_RESUME must be 0 or 1" >&2; exit 2 ;;
esac
ckpt_setting="restaurant_pass_counter-franka_atomic205_recovery5-franka-joint-0"
case "${OPENPI_TRAIN_CONFIG_NAME}" in
  pi05_restaurant_franka_lora|pi05_restaurant_franka_action_head)
    ckpt_setting+="-${OPENPI_TRAIN_CONFIG_NAME#pi05_restaurant_franka_}"
    ;;
esac
if [[ "${OPENPI_TRAIN_RESUME:-0}" == "0" && "${OPENPI_TRAIN_CONFIG_NAME}" != "pi05_restaurant_franka_full_finetune" && -d "${PERSISTENT_CHECKPOINT_ROOT}/${ckpt_setting}" ]]; then
  echo "Checkpoint already exists; set OPENPI_TRAIN_RESUME=1 or choose another OPENPI_CHECKPOINT_ROOT" >&2
  exit 2
fi
export OPENPI_FSDP_DEVICES="${GPU_COUNT}"
mkdir -p \
  "$(dirname "${LOCAL_LEROBOT_HOME}/${OPENPI_LEROBOT_REPO_ID}")" \
  "${LOCAL_BASE_MODEL}" \
  "${LOCAL_ENVIRONMENT_ROOT}" \
  "${LOCAL_CHECKPOINT_ROOT}" \
  "${LOCAL_LOG_ROOT}"

stage_out() {
  local train_status=$?
  local checkpoint_status
  local log_status
  trap - EXIT
  set +e
  echo "[Pi_05] stage-out checkpoints=${PERSISTENT_CHECKPOINT_ROOT} logs=${PERSISTENT_LOG_ROOT}"
  mkdir -p "${PERSISTENT_CHECKPOINT_ROOT}" "${PERSISTENT_LOG_ROOT}"
  rsync -a "${LOCAL_CHECKPOINT_ROOT}/" "${PERSISTENT_CHECKPOINT_ROOT}/"
  checkpoint_status=$?
  echo "[Pi_05] train_status=${train_status} checkpoint_stage_out_status=${checkpoint_status}"
  rsync -a "${LOCAL_LOG_ROOT}/" "${PERSISTENT_LOG_ROOT}/"
  log_status=$?
  if [[ ${checkpoint_status} -ne 0 || ${log_status} -ne 0 ]]; then
    exit 1
  fi
  exit "${train_status}"
}
trap stage_out EXIT

echo "[Pi_05] stage-in dataset=${PERSISTENT_LEROBOT_HOME}/${OPENPI_LEROBOT_REPO_ID}"
rsync -a \
  "${PERSISTENT_LEROBOT_HOME}/${OPENPI_LEROBOT_REPO_ID}/" \
  "${LOCAL_LEROBOT_HOME}/${OPENPI_LEROBOT_REPO_ID}/"
echo "[Pi_05] stage-in init_params=${INIT_PARAMS_SOURCE}"
rsync -a "${INIT_PARAMS_SOURCE}/" "${LOCAL_BASE_MODEL}/params/"
if [[ "${OPENPI_TRAIN_RESUME:-0}" == "1" ]]; then
  echo "[Pi_05] stage-in resume_checkpoint=${PERSISTENT_CHECKPOINT_ROOT}/${ckpt_setting}"
  rsync -a "${PERSISTENT_CHECKPOINT_ROOT}/${ckpt_setting}/" "${LOCAL_CHECKPOINT_ROOT}/${ckpt_setting}/"
fi
echo "[Pi_05] stage-in environment=${ENV_ARCHIVE}"
tar -xf "${ENV_ARCHIVE}" -C "${LOCAL_ENVIRONMENT_ROOT}"

export HF_LEROBOT_HOME="${LOCAL_LEROBOT_HOME}"
export OPENPI_BASE_PARAMS="${LOCAL_BASE_MODEL}/params"
export OPENPI_DATA_HOME="${LOCAL_ROOT}/openpi_cache"
export OPENPI_CHECKPOINT_ROOT="${LOCAL_CHECKPOINT_ROOT}"
export OPENPI_LOCAL_CACHE_ROOT="${LOCAL_ROOT}/cache"
export OPENPI_VENV="${LOCAL_VENV}"
export WANDB_DIR="${WANDB_DIR:-${LOCAL_LOG_ROOT}/wandb}"

echo "[Pi_05] local_root=${LOCAL_ROOT}"
set +e
bash "${POLICY_DIR}/train.sh" restaurant_pass_counter franka_atomic205_recovery5 franka joint 0 "${GPU_IDS}" \
  2>&1 | tee -a "${LOCAL_LOG}"
train_status=${PIPESTATUS[0]}
set -e
exit "${train_status}"
