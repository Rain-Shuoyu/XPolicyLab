"""Pi-0.5 camera color contracts at dataset and model boundaries."""

import json
from pathlib import Path

import numpy as np


MODEL_SIDECAR = 'input-color-order.json'
DATASET_SIDECAR = 'camera-color-order.json'
MODEL_SCHEMA = 'xpolicylab_pi05_input_color_order_v1'
DATASET_SCHEMA = 'xpolicylab_pi05_camera_color_order_v1'


def is_encoded_image_source(value) -> bool:
    """Distinguish JPEG buffers from already decoded HWC/THWC arrays."""
    if isinstance(value, (bytes, bytearray, memoryview, str)):
        return True
    array = np.asarray(value)
    return array.dtype.kind in {'S', 'U', 'O'} or array.ndim <= 2


def decoded_frames_for_lerobot(frames, *, pil_rgb_jpeg_source: bool, encoded: bool):
    """Convert OpenCV-decoded PIL RGB JPEGs to RGB before LeRobot writes them."""
    frames = np.asarray(frames)
    if frames.dtype != np.uint8 or frames.ndim not in (3, 4) or frames.shape[-1] != 3:
        raise ValueError('expected decoded uint8 HWC or THWC color frames')
    if pil_rgb_jpeg_source and encoded:
        return np.ascontiguousarray(frames[..., ::-1])
    return frames


def require_new_dataset_path(path) -> None:
    if Path(path).exists():
        raise FileExistsError(f'LeRobot dataset already exists: {path}')


def write_rgb_dataset_contract(dataset_root, *, source: str) -> Path:
    """Record the new dataset contract without changing prior datasets."""
    if source != 'pil_rgb_jpeg':
        raise ValueError('Pi-0.5 RGB contract requires PIL RGB JPEG source')
    path = Path(dataset_root) / DATASET_SIDECAR
    with path.open('x') as stream:
        json.dump({'schema_version': DATASET_SCHEMA,
                   'source_image_encoding': source,
                   'camera_color_order': 'RGB',
                   'model_input_color_order': 'RGB'}, stream, indent=2)
        stream.write('\n')
    return path


def resolve_input_color_order(model_cfg: dict, model_root) -> str:
    """Legacy BGR weights opt in via config or sidecar; unknown weights use RGB."""
    explicit = model_cfg.get('input_color_order')
    model_root = Path(model_root)
    sidecar = model_root / MODEL_SIDECAR
    if not sidecar.exists() and model_root.name.isdigit():
        sidecar = model_root.parent / MODEL_SIDECAR
    recorded = None
    if sidecar.exists():
        metadata = json.loads(sidecar.read_text())
        if metadata.get('schema_version') != MODEL_SCHEMA:
            raise ValueError(f'unsupported Pi-0.5 color metadata: {sidecar}')
        recorded = metadata.get('input_color_order')
    for value in (explicit, recorded):
        if value is not None and value not in ('RGB', 'BGR'):
            raise ValueError('input_color_order must be RGB or BGR')
    if explicit is not None and recorded is not None and explicit != recorded:
        raise ValueError('input_color_order conflict between config and checkpoint metadata')
    return explicit or recorded or 'RGB'


def adapt_encoded_observation(observation: dict, input_color_order: str) -> dict:
    """Adapt only model CHW images; public RGB observations remain untouched."""
    if input_color_order == 'RGB':
        return observation
    if input_color_order != 'BGR':
        raise ValueError('input_color_order must be RGB or BGR')
    return {**observation, 'images': {
        name: np.ascontiguousarray(image[::-1])
        for name, image in observation['images'].items()
    }}
