from pathlib import Path
import dataclasses

import pytest

from scripts import transform_lerobot_v30_format as converter


def test_collects_only_the_explicit_single_target_input_dir(tmp_path: Path) -> None:
    selected = tmp_path / "accepted"
    selected.mkdir()
    episode = selected / "episode.hdf5"
    episode.touch()

    target = [("osb", "restaurant_pass_counter", "franka")]
    collected = converter._collect_target_input_files(target, input_dir=selected)

    assert collected[0][3] == selected
    assert collected[0][4] == [episode]

    with pytest.raises(ValueError, match="exactly one target"):
        converter._collect_target_input_files(target * 2, input_dir=selected)


def test_explicit_input_dir_resolves_exact_target_without_data_tree() -> None:
    assert converter._resolve_targets(
        ["osb.restaurant_pass_counter.franka"],
        input_dir=Path("/shared/accepted"),
    ) == [("osb", "restaurant_pass_counter", "franka")]

    with pytest.raises(ValueError, match="wildcards"):
        converter._resolve_targets(["osb.*.franka"], input_dir=Path("/shared/accepted"))


def test_explicit_integer_fps_overrides_environment_metadata() -> None:
    assert converter._resolve_fps(None, metadata_fps=25) == 25
    assert converter._resolve_fps(17, metadata_fps=25) == 17

    with pytest.raises(ValueError, match="positive integer"):
        converter._resolve_fps(0, metadata_fps=25)


def test_reads_instruction_from_episode_metadata() -> None:
    data = {
        "metadata": {
            "instruction": (
                "Place the hamburger, french fries, and coke can on the tray in order, "
                "press the service bell, then return both arms home."
            )
        }
    }

    assert converter._find_instructions(data) == [data["metadata"]["instruction"]]


def test_instruction_pool_is_assigned_deterministically() -> None:
    instructions = ["first", "second", "third"]

    assert converter._choose_instruction({}, instructions, episode_index=0) == "first"
    assert converter._choose_instruction({}, instructions, episode_index=4) == "second"


def test_streaming_dataset_forwards_h264_codec_and_queue(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    class Dataset:
        @staticmethod
        def create(**kwargs):
            captured.update(kwargs)
            return object()

    monkeypatch.setattr(converter, "HF_LEROBOT_HOME", tmp_path)
    monkeypatch.setattr(converter, "LeRobotDataset", Dataset)
    config = dataclasses.replace(
        converter.DEFAULT_DATASET_CONFIG,
        encoder_queue_maxsize=512,
    )

    converter.create_empty_dataset(
        repo_id="test/dataset",
        robot_type="franka",
        motors=["joint"],
        fps=17,
        dataset_config=config,
    )

    assert captured["vcodec"] == "h264"
    assert captured["encoder_queue_maxsize"] == 512


def test_finalize_dataset_flushes_the_dataset_parquet_writer() -> None:
    class Dataset:
        finalized = False

        def finalize(self) -> None:
            self.finalized = True

    dataset = Dataset()

    converter.finalize_dataset(dataset)

    assert dataset.finalized is True


def test_prepare_validates_task_prompts_from_lerobot_tasks_index() -> None:
    prepare_script = (
        Path(__file__).resolve().parents[1]
        / "policy"
        / "Pi_05"
        / "prepare_restaurant_dense50.sh"
    ).read_text()

    assert "metadata.tasks.index" in prepare_script
    assert 'metadata.tasks["task"]' not in prepare_script
