#!/usr/bin/env python3
"""Fail fast when a reusable Pi 0.5 ACP runtime is incomplete."""

from __future__ import annotations

import argparse
import importlib
from importlib import metadata
from pathlib import Path
import re
import subprocess
import tempfile


REQUIRED_IMPORTS = ("cv2", "torch", "jax", "lerobot", "openpi")


def _normalized_distribution_name(name: str) -> str:
    return name.lower().replace("_", "-")


def installed_distribution_names() -> set[str]:
    return {
        _normalized_distribution_name(dist.metadata["Name"])
        for dist in metadata.distributions()
        if dist.metadata.get("Name")
    }


def validate_opencv_distributions(distributions: set[str]) -> None:
    normalized = {_normalized_distribution_name(name) for name in distributions}
    if "opencv-python" in normalized:
        raise RuntimeError(
            "forbidden GUI distribution opencv-python is installed; "
            "the ACP runtime must use opencv-python-headless only"
        )
    if "opencv-python-headless" not in normalized:
        raise RuntimeError("required distribution opencv-python-headless is missing")


def validate_required_paths(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError("required paths are missing: " + ", ".join(missing))


def validate_gpu_count(expected: int, nvidia_smi_output: str) -> None:
    visible = len([line for line in nvidia_smi_output.splitlines() if line.strip()])
    if visible < expected:
        raise RuntimeError(f"expected {expected} GPUs, found {visible}")


def validate_ptxas_version(output: str) -> None:
    match = re.search(r"release (\d+)\.(\d+)", output)
    if match is None:
        raise RuntimeError("could not determine ptxas version")
    version = tuple(int(part) for part in match.groups())
    if version < (12, 8):
        raise RuntimeError(
            f"ptxas {version[0]}.{version[1]} is too old for RTX 5090; "
            "CUDA 12.8 or newer is required"
        )


def validate_writable_directories(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".pi05-write-probe-", dir=path):
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-gpus", type=int, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-params", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--ptxas", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_opencv_distributions(installed_distribution_names())
    for module_name in REQUIRED_IMPORTS:
        importlib.import_module(module_name)
    validate_required_paths([args.dataset, args.base_params, args.tokenizer, args.ptxas])
    if not args.ptxas.is_file() or not args.ptxas.stat().st_mode & 0o111:
        raise RuntimeError(f"ptxas is not executable: {args.ptxas}")
    ptxas_result = subprocess.run(
        [str(args.ptxas), "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    validate_ptxas_version(ptxas_result.stdout + ptxas_result.stderr)
    validate_writable_directories(
        [args.assets_root, args.log_root, args.checkpoint_root]
    )
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    validate_gpu_count(args.expected_gpus, result.stdout)
    print(
        "[Pi_05 preflight] OK "
        f"gpus={args.expected_gpus} dataset={args.dataset} base_params={args.base_params}"
    )


if __name__ == "__main__":
    main()
