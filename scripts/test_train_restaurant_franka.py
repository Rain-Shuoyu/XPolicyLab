from pathlib import Path
import os
import re
import subprocess
import tarfile
import tomllib

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "policy"
    / "Pi_05"
    / "train_restaurant_franka.sh"
)
TRAIN_SCRIPT = SCRIPT.with_name("train.sh")
OPENPI_PROJECT = SCRIPT.parent / "openpi"


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(0o755)


def _prepare_environment(
    tmp_path: Path,
    visible_gpu_count: int,
    *,
    train_status: int = 0,
) -> tuple[dict[str, str], Path, Path, str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    visible_gpu_lines = "".join(f"{gpu}\n" for gpu in range(visible_gpu_count))
    _write_executable(
        fake_bin / "nvidia-smi",
        f"#!/bin/sh\nprintf '{visible_gpu_lines}'\n",
    )
    _write_executable(
        fake_bin / "bash",
        """#!/bin/sh
last_arg=
for arg in "$@"; do
  last_arg=$arg
done
{
  printf 'HF_LEROBOT_HOME=%s\n' "$HF_LEROBOT_HOME"
  printf 'OPENPI_BASE_PARAMS=%s\n' "$OPENPI_BASE_PARAMS"
  printf 'OPENPI_CHECKPOINT_ROOT=%s\n' "$OPENPI_CHECKPOINT_ROOT"
  printf 'OPENPI_DATA_HOME=%s\n' "$OPENPI_DATA_HOME"
  printf 'OPENPI_FSDP_DEVICES=%s\n' "$OPENPI_FSDP_DEVICES"
  printf 'OPENPI_VENV=%s\n' "$OPENPI_VENV"
  printf 'GPU_IDS=%s\n' "$last_arg"
} > "$CAPTURE_FILE"
if [ "$FAKE_TRAIN_STATUS" -ne 0 ]; then
  mkdir -p "$OPENPI_CHECKPOINT_ROOT/fake-run"
  printf 'checkpoint\n' > "$OPENPI_CHECKPOINT_ROOT/fake-run/step.txt"
  printf 'fake training failure\n' >&2
fi
exit "$FAKE_TRAIN_STATUS"
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
    environment = tmp_path / "environment"
    (environment / "bin").mkdir(parents=True)
    (environment / "bin" / "python").write_text("python")
    environment_archive = shared_root / "environments" / "pi05-openpi.tar"
    environment_archive.parent.mkdir(parents=True)
    with tarfile.open(environment_archive, "w") as archive:
        archive.add(environment, arcname="pi05-openpi")

    train_root = shared_root / "training" / "pi05_restaurant"
    local_root = tmp_path / "node-local"
    capture = tmp_path / "training-env.txt"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "CAPTURE_FILE": str(capture),
            "FAKE_TRAIN_STATUS": str(train_status),
            "OPENPI_SHARED_ROOT": str(shared_root),
            "OPENPI_TRAIN_ROOT": str(train_root),
            "OPENPI_NODE_LOCAL_ROOT": str(local_root),
            "OPENPI_BASE_MODEL_SOURCE": str(base_model),
            "OPENPI_ENV_ARCHIVE": str(environment_archive),
            "OPENPI_LEROBOT_REPO_ID": repo_id,
        }
    )
    return env, train_root, local_root, repo_id


@pytest.mark.parametrize(
    ("gpu_count", "expected_gpu_ids"),
    [(1, "0"), (2, "0,1"), (4, "0,1,2,3"), (8, "0,1,2,3,4,5,6,7")],
)
def test_selects_requested_gpu_count(
    tmp_path: Path,
    gpu_count: int,
    expected_gpu_ids: str,
) -> None:
    env, _, _, _ = _prepare_environment(tmp_path, visible_gpu_count=8)

    result = subprocess.run(
        ["/bin/bash", str(SCRIPT), str(gpu_count)],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    captured = dict(
        line.split("=", 1)
        for line in Path(env["CAPTURE_FILE"]).read_text().splitlines()
    )
    assert captured["OPENPI_FSDP_DEVICES"] == str(gpu_count)
    assert captured["GPU_IDS"] == expected_gpu_ids


@pytest.mark.parametrize("arguments", [[], ["3"]])
def test_rejects_missing_or_unsupported_gpu_count(
    tmp_path: Path,
    arguments: list[str],
) -> None:
    env, _, _, _ = _prepare_environment(tmp_path, visible_gpu_count=8)

    result = subprocess.run(
        ["/bin/bash", str(SCRIPT), *arguments],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert result.stderr == "Expected GPU count to be one of: 1, 2, 4, 8\n"
    assert not Path(env["CAPTURE_FILE"]).exists()
    assert "[Pi_05] stage-in" not in result.stdout


def test_rejects_request_larger_than_visible_gpu_count(tmp_path: Path) -> None:
    env, _, _, _ = _prepare_environment(tmp_path, visible_gpu_count=2)

    result = subprocess.run(
        ["/bin/bash", str(SCRIPT), "4"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert result.stderr == "Requested 4 GPUs, but only 2 are visible\n"
    assert not Path(env["CAPTURE_FILE"]).exists()
    assert "[Pi_05] stage-in" not in result.stdout


def test_stages_training_inputs_and_copies_outputs_back_on_failure(tmp_path: Path) -> None:
    env, train_root, local_root, repo_id = _prepare_environment(
        tmp_path,
        visible_gpu_count=2,
        train_status=7,
    )

    result = subprocess.run(
        ["/bin/bash", str(SCRIPT), "2"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 7
    captured = dict(
        line.split("=", 1)
        for line in Path(env["CAPTURE_FILE"]).read_text().splitlines()
    )
    assert captured == {
        "HF_LEROBOT_HOME": str(local_root / "lerobot"),
        "OPENPI_BASE_PARAMS": str(local_root / "base" / "pi05_base" / "params"),
        "OPENPI_CHECKPOINT_ROOT": str(local_root / "checkpoints"),
        "OPENPI_DATA_HOME": str(local_root / "openpi_cache"),
        "OPENPI_FSDP_DEVICES": "2",
        "OPENPI_VENV": str(local_root / "environment" / "pi05-openpi"),
        "GPU_IDS": "0,1",
    }
    assert (local_root / "lerobot" / repo_id / "data.parquet").read_text() == "dataset"
    assert (local_root / "base" / "pi05_base" / "params" / "weights").read_text() == "weights"
    assert (local_root / "environment" / "pi05-openpi" / "bin" / "python").read_text() == "python"
    assert (train_root / "checkpoints" / "fake-run" / "step.txt").read_text() == "checkpoint\n"
    assert "fake training failure" in (
        train_root / "logs" / "train_restaurant_franka.log"
    ).read_text()
    assert "fake training failure" in result.stdout


def test_has_no_hardware_model_branch_or_sleep() -> None:
    contents = SCRIPT.read_text()

    assert "sleep" not in contents
    assert "RTX" not in contents
    assert "H100" not in contents


def test_training_uses_staged_python_without_uv_sync() -> None:
    contents = TRAIN_SCRIPT.read_text()

    assert '"${openpi_venv}/bin/python"' in contents
    assert "uv_bin" not in contents
    assert "uv run" not in contents


def test_openpi_uses_only_headless_opencv() -> None:
    pyproject = tomllib.loads((OPENPI_PROJECT / "pyproject.toml").read_text())
    dependencies = pyproject["project"]["dependencies"]
    assert any(item.startswith("opencv-python-headless") for item in dependencies)
    assert not any(
        re.match(r"^opencv-python(?:[<>=!~]|$)", item) for item in dependencies
    )

    lock = tomllib.loads((OPENPI_PROJECT / "uv.lock").read_text())
    package_names = [package["name"] for package in lock["package"]]
    assert "opencv-python-headless" in package_names
    assert "opencv-python" not in package_names
