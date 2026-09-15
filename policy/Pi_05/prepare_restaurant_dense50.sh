#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPOLICYLAB_DIR="$(cd "${POLICY_DIR}/../.." && pwd)"
OPENPI_DIR="${POLICY_DIR}/openpi"

SHARED_ROOT="${OPENPI_SHARED_ROOT:-/mnt/afs/L202500576}"
COLLECTION_SUMMARY="${OPENPI_COLLECTION_SUMMARY:-${SHARED_ROOT}/projects/XPolicyLab/data/osb/restaurant_pass_counter/dense_augmentation_50/collection_summary.json}"
TRAIN_ROOT="${OPENPI_TRAIN_ROOT:-${SHARED_ROOT}/training/pi05_restaurant}"
STAGING_DIR="${OPENPI_STAGING_DIR:-${TRAIN_ROOT}/accepted_dense50}"
MANIFEST_PATH="${OPENPI_CONVERSION_MANIFEST:-${TRAIN_ROOT}/conversion_manifest.json}"

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${SHARED_ROOT}/datasets/lerobot}"
export OPENPI_LEROBOT_REPO_ID="${OPENPI_LEROBOT_REPO_ID:-openskillbench/restaurant_pass_counter_franka_dense50}"
export OPENPI_ASSETS_ROOT="${OPENPI_ASSETS_ROOT:-${TRAIN_ROOT}/assets}"
export OPENPI_CHECKPOINT_ROOT="${OPENPI_CHECKPOINT_ROOT:-${TRAIN_ROOT}/checkpoints}"
export OPENPI_VENV="${OPENPI_VENV:-${SHARED_ROOT}/venvs/pi05-openpi}"
export OPENPI_UV_BIN="${OPENPI_UV_BIN:-${SHARED_ROOT}/bin/uv}"

mkdir -p "${TRAIN_ROOT}" "${HF_LEROBOT_HOME}" "${OPENPI_ASSETS_ROOT}"
mkdir -p "${STAGING_DIR}"

export COLLECTION_SUMMARY STAGING_DIR MANIFEST_PATH
cd "${OPENPI_DIR}"
VIRTUAL_ENV="${OPENPI_VENV}" "${OPENPI_UV_BIN}" run --active --frozen --group lerobot python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

import h5py

summary_path = Path(os.environ["COLLECTION_SUMMARY"])
staging_dir = Path(os.environ["STAGING_DIR"])
manifest_path = Path(os.environ["MANIFEST_PATH"])
collection = json.loads(summary_path.read_text())
batch_summary_path = Path(collection["supervisor_summary_path"])
batch = json.loads(batch_summary_path.read_text())

for path in staging_dir.iterdir():
    if not path.is_symlink() or not path.name.startswith("episode_"):
        raise ValueError(f"Unexpected entry in accepted-only staging directory: {path}")
    path.unlink()

episode_roots = [Path(path) for path in collection["existing_episode_paths"]]
episode_roots.extend(
    Path(record["candidate_root_path"])
    for record in batch["records"]
    if record["status"] == "accepted"
)
if len(episode_roots) != 50:
    raise ValueError(f"Expected 50 accepted episodes, got {len(episode_roots)}")

sources = []
for index, episode_root in enumerate(episode_roots):
    source = episode_root / "public" / "episode.hdf5"
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    with h5py.File(source, "r") as episode:
        frame_count = int(episode["state/left_arm_joint_states"].shape[0])
    link = staging_dir / f"episode_{index:06d}.hdf5"
    link.symlink_to(source)
    sources.append(
        {
            "episode_index": index,
            "source_hdf5": str(source),
            "sha256": digest.hexdigest(),
            "frame_count": frame_count,
        }
    )

manifest = {
    "schema_version": "osb_pi05_lerobot_conversion_v1",
    "collection_summary": str(summary_path),
    "source_sampling_hz": "50/3",
    "lerobot_fps": 17,
    "resampled": False,
    "repo_id": os.environ["OPENPI_LEROBOT_REPO_ID"],
    "sources": sources,
    "expected_episode_count": 50,
    "expected_frame_count": sum(source["frame_count"] for source in sources),
}
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Staged {len(sources)} accepted episodes in {staging_dir}")
PY

HF_LEROBOT_HOME="${HF_LEROBOT_HOME}" \
VIRTUAL_ENV="${OPENPI_VENV}" \
"${OPENPI_UV_BIN}" run --active --frozen --group lerobot python \
  "${XPOLICYLAB_DIR}/scripts/transform_lerobot_v30_format.py" \
  osb.restaurant_pass_counter.franka \
  --input-dir "${STAGING_DIR}" \
  --fps 17 \
  --repo_id "${OPENPI_LEROBOT_REPO_ID}" \
  --max_episode 50

export MANIFEST_PATH
VIRTUAL_ENV="${OPENPI_VENV}" "${OPENPI_UV_BIN}" run --active --frozen --group lerobot python - <<'PY'
import json
import os
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

repo_id = os.environ["OPENPI_LEROBOT_REPO_ID"]
manifest_path = Path(os.environ["MANIFEST_PATH"])
manifest = json.loads(manifest_path.read_text())
metadata = LeRobotDatasetMetadata(repo_id)

camera_keys = {
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
}
if metadata.total_episodes != 50:
    raise ValueError(f"Expected 50 episodes, got {metadata.total_episodes}")
if metadata.total_frames != manifest["expected_frame_count"]:
    raise ValueError(
        f"Frame count mismatch: expected {manifest['expected_frame_count']}, got {metadata.total_frames}"
    )
if metadata.fps != 17:
    raise ValueError(f"Expected 17 FPS, got {metadata.fps}")
if metadata.shapes["observation.state"] != (16,) or metadata.shapes["action"] != (16,):
    raise ValueError("Expected 16-D observation.state and action")
if set(metadata.camera_keys) != camera_keys:
    raise ValueError(f"Unexpected camera keys: {metadata.camera_keys}")
if metadata.total_tasks < 1 or not metadata.tasks["task"].astype(str).str.strip().all():
    raise ValueError("Dataset task prompt is empty")

dataset = LeRobotDataset(repo_id, video_backend="pyav")
sample = dataset[0]
if tuple(sample["observation.state"].shape) != (16,) or tuple(sample["action"].shape) != (16,):
    raise ValueError("Decoded sample does not preserve 16-D state/action")
for camera_key in camera_keys:
    if camera_key not in sample:
        raise ValueError(f"Decoded sample is missing {camera_key}")

manifest["result"] = {
    "episode_count": metadata.total_episodes,
    "frame_count": metadata.total_frames,
    "fps": metadata.fps,
    "state_dim": 16,
    "action_dim": 16,
    "camera_keys": sorted(camera_keys),
}
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest["result"], indent=2))
PY

VIRTUAL_ENV="${OPENPI_VENV}" "${OPENPI_UV_BIN}" run --active --frozen --group lerobot \
  scripts/compute_norm_stats.py --config-name pi05_restaurant_franka_full_finetune

echo "Prepared ${OPENPI_LEROBOT_REPO_ID} and normalization stats under ${OPENPI_ASSETS_ROOT}"
