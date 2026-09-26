"""Pi-0.5 color contracts at the real converter and model input boundary."""

import io
import json

import cv2
import numpy as np
import pytest
from PIL import Image

from XPolicyLab.policy.Pi_05.color_order import (
    adapt_encoded_observation, decoded_frames_for_lerobot,
    is_encoded_image_source, require_new_dataset_path,
    resolve_input_color_order, write_rgb_dataset_contract,
)


def test_pil_jpeg_becomes_rgb_and_decoded_rgb_stays_rgb():
    original = np.zeros((8, 8, 3), dtype=np.uint8)
    original[..., 0], original[..., 1], original[..., 2] = 240, 35, 8
    buffer = io.BytesIO()
    Image.fromarray(original, mode='RGB').save(buffer, format='JPEG', quality=95)
    raw = buffer.getvalue()
    pil_rgb = np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'))
    cv2_bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert is_encoded_image_source([raw])
    assert np.array_equal(decoded_frames_for_lerobot(
        cv2_bgr[None], pil_rgb_jpeg_source=True, encoded=True
    )[0], pil_rgb)
    assert not is_encoded_image_source(pil_rgb[None])
    assert np.array_equal(decoded_frames_for_lerobot(
        pil_rgb[None], pil_rgb_jpeg_source=True, encoded=False
    ), pil_rgb[None])


def test_legacy_checkpoint_bgr_adapts_only_model_chw_images(tmp_path):
    root = tmp_path / 'checkpoint'
    root.mkdir()
    image = np.zeros((3, 2, 2), dtype=np.uint8)
    image[0], image[2] = 7, 240
    obs = {'images': {name: image for name in ('cam_high', 'cam_left_wrist', 'cam_right_wrist')},
           'state': np.arange(16, dtype=np.float32), 'prompt': 'burger'}
    assert resolve_input_color_order({}, root) == 'RGB'
    assert adapt_encoded_observation(obs, 'RGB') is obs
    assert resolve_input_color_order({'input_color_order': 'BGR'}, root) == 'BGR'
    changed = adapt_encoded_observation(obs, 'BGR')
    assert changed['state'] is obs['state'] and changed['prompt'] == obs['prompt']
    assert all(np.array_equal(value[0], image[2]) and np.array_equal(value[2], image[0])
               for value in changed['images'].values())
    assert np.array_equal(image[0], np.full((2, 2), 7))


def test_sidecar_conflicts_fail_and_existing_dataset_is_preserved(tmp_path):
    root = tmp_path / 'checkpoint'
    root.mkdir()
    (root / 'input-color-order.json').write_text(json.dumps({
        'schema_version': 'xpolicylab_pi05_input_color_order_v1',
        'input_color_order': 'BGR',
    }))
    assert resolve_input_color_order({}, root) == 'BGR'
    with pytest.raises(ValueError, match='conflict'):
        resolve_input_color_order({'input_color_order': 'RGB'}, root)
    with pytest.raises(ValueError, match='RGB or BGR'):
        resolve_input_color_order({'input_color_order': 'GRB'}, root)
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    with pytest.raises(FileExistsError):
        require_new_dataset_path(dataset)
    sidecar = write_rgb_dataset_contract(dataset, source='pil_rgb_jpeg')
    assert json.loads(sidecar.read_text())['camera_color_order'] == 'RGB'
    with pytest.raises(FileExistsError):
        write_rgb_dataset_contract(dataset, source='pil_rgb_jpeg')
