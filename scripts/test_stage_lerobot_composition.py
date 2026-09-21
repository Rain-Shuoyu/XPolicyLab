import json
from pathlib import Path

import h5py

from scripts.stage_lerobot_composition import stage_composition


def _write_episode(path: Path, frame_count: int) -> None:
    with h5py.File(path, "w") as episode:
        episode.create_dataset(
            "state/left_arm_joint_states",
            shape=(frame_count, 7),
            dtype="f4",
        )


def test_stages_ordered_composition(tmp_path: Path) -> None:
    first = tmp_path / "first.hdf5"
    second = tmp_path / "second.hdf5"
    _write_episode(first, 3)
    _write_episode(second, 5)
    composition = tmp_path / "composition.json"
    composition.write_text(
        json.dumps(
            {
                "schema_version": "xpolicylab_lerobot_composition_v1",
                "repo_id": "test/combined",
                "fps": 17,
                "expected_episode_count": 2,
                "expected_frame_count": 8,
                "episodes": [
                    {
                        "source_hdf5": str(first),
                        "instruction": "First instruction.",
                        "provenance": {"source": "base"},
                    },
                    {
                        "source_hdf5": str(second),
                        "instruction": "Second instruction.",
                        "provenance": {"source": "recovery", "seed": 3},
                    },
                ],
            }
        )
    )
    staging = tmp_path / "staging"
    instructions = tmp_path / "instructions.txt"
    resolved = tmp_path / "resolved.json"

    result = stage_composition(composition, staging, instructions, resolved)

    assert (staging / "episode_000000.hdf5").resolve() == first
    assert (staging / "episode_000001.hdf5").resolve() == second
    assert instructions.read_text().splitlines() == [
        "First instruction.",
        "Second instruction.",
    ]
    assert result["expected_episode_count"] == 2
    assert result["expected_frame_count"] == 8
    assert [source["frame_count"] for source in result["sources"]] == [3, 5]
    assert all(len(source["sha256"]) == 64 for source in result["sources"])
    assert json.loads(resolved.read_text()) == result
