from pathlib import Path
import os
import subprocess


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "policy"
    / "Pi_05"
    / "train_restaurant_franka_8gpu.sh"
)


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(0o755)


def test_stages_training_inputs_and_copies_outputs_back_on_failure(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "nvidia-smi",
        "#!/bin/sh\nprintf '0\\n1\\n2\\n3\\n4\\n5\\n6\\n7\\n'\n",
    )
    _write_executable(
        fake_bin / "bash",
        """#!/bin/sh
{
  printf 'HF_LEROBOT_HOME=%s\\n' "$HF_LEROBOT_HOME"
  printf 'OPENPI_BASE_PARAMS=%s\\n' "$OPENPI_BASE_PARAMS"
  printf 'OPENPI_CHECKPOINT_ROOT=%s\\n' "$OPENPI_CHECKPOINT_ROOT"
  printf 'OPENPI_DATA_HOME=%s\\n' "$OPENPI_DATA_HOME"
} > "$CAPTURE_FILE"
mkdir -p "$OPENPI_CHECKPOINT_ROOT/fake-run"
printf 'checkpoint\\n' > "$OPENPI_CHECKPOINT_ROOT/fake-run/step.txt"
printf 'fake training failure\\n' >&2
exit 7
""",
    )

    shared_root = tmp_path / "shared"
    repo_id = "openskillbench/restaurant_pass_counter_franka_dense50"
    dataset = shared_root / "datasets" / "lerobot" / repo_id
    dataset.mkdir(parents=True)
    (dataset / "data.parquet").write_text("dataset")
    base_model = shared_root / "openpi-cache" / "pi05_base"
    (base_model / "params").mkdir(parents=True)
    (base_model / "params" / "weights").write_text("weights")

    train_root = shared_root / "training" / "pi05_restaurant"
    local_root = tmp_path / "node-local"
    capture = tmp_path / "training-env.txt"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "CAPTURE_FILE": str(capture),
            "OPENPI_SHARED_ROOT": str(shared_root),
            "OPENPI_TRAIN_ROOT": str(train_root),
            "OPENPI_NODE_LOCAL_ROOT": str(local_root),
            "OPENPI_BASE_MODEL_SOURCE": str(base_model),
            "OPENPI_LEROBOT_REPO_ID": repo_id,
        }
    )

    result = subprocess.run(
        ["/bin/bash", str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 7
    captured = dict(line.split("=", 1) for line in capture.read_text().splitlines())
    assert captured == {
        "HF_LEROBOT_HOME": str(local_root / "lerobot"),
        "OPENPI_BASE_PARAMS": str(local_root / "base" / "pi05_base" / "params"),
        "OPENPI_CHECKPOINT_ROOT": str(local_root / "checkpoints"),
        "OPENPI_DATA_HOME": str(local_root / "openpi_cache"),
    }
    assert (local_root / "lerobot" / repo_id / "data.parquet").read_text() == "dataset"
    assert (local_root / "base" / "pi05_base" / "params" / "weights").read_text() == "weights"
    assert (train_root / "checkpoints" / "fake-run" / "step.txt").read_text() == "checkpoint\n"
    assert "fake training failure" in (
        train_root / "logs" / "train_restaurant_franka_8gpu.log"
    ).read_text()
    assert "fake training failure" in result.stdout
    assert "sleep" not in SCRIPT.read_text()
