from pathlib import Path
import importlib.util
import os
import re
import subprocess
import tomllib

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "policy"
    / "Pi_05"
    / "train_restaurant_franka.sh"
)
TRAIN_SCRIPT = SCRIPT.with_name("train.sh")
PREFLIGHT_SCRIPT = SCRIPT.with_name("acp_runtime_preflight.py")
OPENPI_PROJECT = SCRIPT.parent / "openpi"


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(0o755)


def _prepare_environment(
    tmp_path: Path,
    visible_gpu_count: int,
    *,
    train_status: int = 0,
    preflight_status: int = 0,
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
  printf 'OPENPI_TRAIN_CONFIG_NAME=%s\n' "$OPENPI_TRAIN_CONFIG_NAME"
  printf 'OPENPI_VENV=%s\n' "$OPENPI_VENV"
  printf 'PTXAS=%s\n' "$(command -v ptxas)"
  printf 'PYTHONDONTWRITEBYTECODE=%s\n' "$PYTHONDONTWRITEBYTECODE"
  printf 'WANDB_MODE=%s\n' "$WANDB_MODE"
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
    environment = shared_root / "venvs" / "pi05-openpi"
    (environment / "bin").mkdir(parents=True)
    _write_executable(
        environment / "bin" / "python",
        """#!/bin/sh
printf '%s\n' "$*" > "$PREFLIGHT_CAPTURE"
exit "$FAKE_PREFLIGHT_STATUS"
        """,
    )
    ptxas = (
        environment
        / "lib"
        / "python3.11"
        / "site-packages"
        / "nvidia"
        / "cuda_nvcc"
        / "bin"
        / "ptxas"
    )
    ptxas.parent.mkdir(parents=True)
    _write_executable(ptxas, "#!/bin/sh\nprintf 'Cuda compilation tools, release 12.9'\n")

    train_root = shared_root / "training" / "pi05_restaurant"
    tokenizer = train_root / "openpi_cache" / "big_vision" / "paligemma_tokenizer.model"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_text("tokenizer")
    local_root = tmp_path / "node-local"
    capture = tmp_path / "training-env.txt"
    preflight_capture = tmp_path / "preflight-args.txt"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "CAPTURE_FILE": str(capture),
            "PREFLIGHT_CAPTURE": str(preflight_capture),
            "FAKE_TRAIN_STATUS": str(train_status),
            "FAKE_PREFLIGHT_STATUS": str(preflight_status),
            "OPENPI_SHARED_ROOT": str(shared_root),
            "OPENPI_TRAIN_ROOT": str(train_root),
            "OPENPI_NODE_LOCAL_ROOT": str(local_root),
            "OPENPI_BASE_MODEL_SOURCE": str(base_model),
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
        "OPENPI_DATA_HOME": str(train_root / "openpi_cache"),
        "OPENPI_FSDP_DEVICES": "2",
        "OPENPI_TRAIN_CONFIG_NAME": "pi05_restaurant_franka_full_finetune",
        "OPENPI_VENV": str(Path(env["OPENPI_SHARED_ROOT"]) / "venvs" / "pi05-openpi"),
        "PTXAS": str(
            Path(env["OPENPI_SHARED_ROOT"])
            / "venvs"
            / "pi05-openpi"
            / "lib"
            / "python3.11"
            / "site-packages"
            / "nvidia"
            / "cuda_nvcc"
            / "bin"
            / "ptxas"
        ),
        "PYTHONDONTWRITEBYTECODE": "1",
        "WANDB_MODE": "offline",
        "GPU_IDS": "0,1",
    }
    assert (local_root / "lerobot" / repo_id / "data.parquet").read_text() == "dataset"
    assert (local_root / "base" / "pi05_base" / "params" / "weights").read_text() == "weights"
    assert (train_root / "checkpoints" / "fake-run" / "step.txt").read_text() == "checkpoint\n"
    assert "fake training failure" in (
        train_root / "logs" / "train_restaurant_franka.log"
    ).read_text()
    assert "fake training failure" in result.stdout


def test_stage_out_does_not_preserve_owner_or_group() -> None:
    contents = SCRIPT.read_text()
    stage_out_body = contents.split("stage_out() {", 1)[1].split("trap stage_out EXIT", 1)[0]

    assert stage_out_body.count("rsync -a --no-owner --no-group") == 2


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


def test_preflight_failure_prevents_training(tmp_path: Path) -> None:
    env, _, _, _ = _prepare_environment(
        tmp_path,
        visible_gpu_count=8,
        preflight_status=9,
    )

    result = subprocess.run(
        ["/bin/bash", str(SCRIPT), "8"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 9
    assert Path(env["PREFLIGHT_CAPTURE"]).exists()
    assert not Path(env["CAPTURE_FILE"]).exists()


def test_allows_smoke_training_config_override(tmp_path: Path) -> None:
    env, _, _, _ = _prepare_environment(tmp_path, visible_gpu_count=1)
    env["OPENPI_TRAIN_CONFIG_NAME"] = "pi05_restaurant_franka_lora_smoke"

    result = subprocess.run(
        ["/bin/bash", str(SCRIPT), "1"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    captured = dict(
        line.split("=", 1)
        for line in Path(env["CAPTURE_FILE"]).read_text().splitlines()
    )
    assert captured["OPENPI_TRAIN_CONFIG_NAME"] == "pi05_restaurant_franka_lora_smoke"


def _load_preflight():
    spec = importlib.util.spec_from_file_location("acp_runtime_preflight", PREFLIGHT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_rejects_gui_opencv_distribution() -> None:
    preflight = _load_preflight()

    preflight.validate_opencv_distributions({"opencv-python-headless"})
    with pytest.raises(RuntimeError, match="opencv-python"):
        preflight.validate_opencv_distributions(
            {"opencv-python", "opencv-python-headless"}
        )


def test_preflight_rejects_missing_required_path(tmp_path: Path) -> None:
    preflight = _load_preflight()
    missing = tmp_path / "missing"

    with pytest.raises(RuntimeError, match=str(missing)):
        preflight.validate_required_paths([missing])


def test_preflight_rejects_insufficient_gpu_count() -> None:
    preflight = _load_preflight()

    with pytest.raises(RuntimeError, match="expected 8 GPUs, found 1"):
        preflight.validate_gpu_count(8, "0\n")


def test_preflight_requires_modern_ptxas() -> None:
    preflight = _load_preflight()

    preflight.validate_ptxas_version(
        "Cuda compilation tools, release 12.9, V12.9.41"
    )
    with pytest.raises(RuntimeError, match="too old for RTX 5090"):
        preflight.validate_ptxas_version(
            "Cuda compilation tools, release 12.4, V12.4.131"
        )


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


def test_openpi_uses_aliyun_as_default_package_index() -> None:
    pyproject = tomllib.loads((OPENPI_PROJECT / "pyproject.toml").read_text())

    assert pyproject["tool"]["uv"]["index"] == [
        {
            "name": "aliyun",
            "url": "https://mirrors.aliyun.com/pypi/simple/",
            "default": True,
        }
    ]
