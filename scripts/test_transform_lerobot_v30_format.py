from pathlib import Path
import dataclasses
import io
import importlib.util
import sys

import h5py
import numpy as np
import pytest
from PIL import Image

_MODULE = Path(__file__).with_name('transform_lerobot_v30_format.py')
_SPEC = importlib.util.spec_from_file_location('pi05_lerobot_v30_converter_test', _MODULE)
converter = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = converter
_SPEC.loader.exec_module(converter)


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


def test_pi05_pil_rgb_jpeg_conversion_and_already_decoded_rgb():
    original = np.zeros((8, 8, 3), dtype=np.uint8)
    original[..., 0], original[..., 2] = 230, 14
    buffer = io.BytesIO()
    Image.fromarray(original, mode='RGB').save(buffer, format='JPEG', quality=95)
    raw = buffer.getvalue()
    expected = np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'))
    from_jpeg = converter._decode_images_if_needed(
        [raw], 8, 8, pil_rgb_jpeg_source=True
    )
    from_rgb = converter._decode_images_if_needed(
        expected[None], 8, 8, pil_rgb_jpeg_source=True
    )
    assert np.array_equal(from_jpeg[0], expected)
    assert np.array_equal(from_rgb[0], expected)


def test_pi05_real_hdf5_loader_preserves_jpeg_identity_until_rgb_conversion(tmp_path):
    original = np.zeros((8, 8, 3), dtype=np.uint8)
    original[..., 0], original[..., 2] = 230, 14
    buffer = io.BytesIO()
    Image.fromarray(original, mode='RGB').save(buffer, format='JPEG', quality=95)
    raw = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    expected = np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert('RGB'))
    source = tmp_path / 'episode.hdf5'
    with h5py.File(source, 'w') as handle:
        for prefix in ('state', 'action'):
            for key, width in (('left_arm_joint_states', 7), ('left_ee_joint_states', 1),
                               ('right_arm_joint_states', 7), ('right_ee_joint_states', 1)):
                handle.create_dataset(f'{prefix}/{key}', data=np.zeros((1, width), dtype=np.float32))
        for camera in ('cam_head', 'cam_left_wrist', 'cam_right_wrist'):
            image = handle.create_dataset(
                f'vision/{camera}/colors', shape=(1,), dtype=h5py.vlen_dtype(np.dtype('uint8'))
            )
            image[0] = raw

    class RecordingDataset:
        def __init__(self):
            self.frames = []

        def add_frame(self, frame):
            self.frames.append(frame)

        def save_episode(self):
            pass

    dataset = RecordingDataset()
    converter.convert_one(
        source, dataset, 'xspark', 'v1.0', (8, 8), (8, 8), 8, 8,
        instruction_pool=['Serve the order.'], pil_rgb_jpeg_source=True,
    )
    assert len(dataset.frames) == 1
    for camera in ('cam_high', 'cam_left_wrist', 'cam_right_wrist'):
        assert np.array_equal(dataset.frames[0][f'observation.images.{camera}'], expected)

    legacy = RecordingDataset()
    converter.convert_one(
        source, legacy, 'xspark', 'v1.0', (8, 8), (8, 8), 8, 8,
        instruction_pool=['Serve the order.'], pil_rgb_jpeg_source=False,
    )
    for camera in ('cam_high', 'cam_left_wrist', 'cam_right_wrist'):
        assert np.array_equal(legacy.frames[0][f'observation.images.{camera}'], expected[..., ::-1])
