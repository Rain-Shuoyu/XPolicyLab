#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import h5py


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_composition(
    composition_path: Path,
    staging_dir: Path,
    instruction_file: Path,
    resolved_manifest_path: Path,
) -> dict:
    composition = json.loads(composition_path.read_text())
    episodes = composition["episodes"]
    expected_episode_count = int(composition["expected_episode_count"])
    if len(episodes) != expected_episode_count:
        raise ValueError(
            f"Expected {expected_episode_count} episodes, got {len(episodes)}"
        )

    staging_dir.mkdir(parents=True, exist_ok=True)
    for path in staging_dir.glob("episode_*.hdf5"):
        if path.is_symlink():
            path.unlink()

    sources = []
    instructions = []
    for episode_index, episode in enumerate(episodes):
        source = Path(episode["source_hdf5"]).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        instruction = str(episode["instruction"]).strip()
        if not instruction:
            raise ValueError(f"Episode {episode_index} has an empty instruction")
        with h5py.File(source, "r") as handle:
            frame_count = int(handle["state/left_arm_joint_states"].shape[0])
        (staging_dir / f"episode_{episode_index:06d}.hdf5").symlink_to(source)
        instructions.append(instruction)
        sources.append(
            {
                "episode_index": episode_index,
                "source_hdf5": str(source),
                "instruction": instruction,
                "provenance": episode.get("provenance", {}),
                "sha256": _sha256(source),
                "frame_count": frame_count,
            }
        )

    frame_count = sum(source["frame_count"] for source in sources)
    expected_frame_count = int(composition["expected_frame_count"])
    if frame_count != expected_frame_count:
        raise ValueError(f"Expected {expected_frame_count} frames, got {frame_count}")

    instruction_file.parent.mkdir(parents=True, exist_ok=True)
    instruction_file.write_text("\n".join(instructions) + "\n")
    result = {
        "schema_version": composition["schema_version"],
        "repo_id": composition["repo_id"],
        "fps": int(composition["fps"]),
        "expected_episode_count": expected_episode_count,
        "expected_frame_count": expected_frame_count,
        "composition_manifest": str(composition_path.resolve()),
        "instruction_file": str(instruction_file.resolve()),
        "staging_dir": str(staging_dir.resolve()),
        "sources": sources,
    }
    resolved_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_manifest_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--composition", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--instruction-file", type=Path, required=True)
    parser.add_argument("--resolved-manifest", type=Path, required=True)
    args = parser.parse_args()
    result = stage_composition(
        args.composition,
        args.staging_dir,
        args.instruction_file,
        args.resolved_manifest,
    )
    print(
        f"Staged {result['expected_episode_count']} episodes and "
        f"{result['expected_frame_count']} frames in {result['staging_dir']}"
    )


if __name__ == "__main__":
    main()
