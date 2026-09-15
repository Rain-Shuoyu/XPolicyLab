import numpy as np

from openpi.policies import aloha_policy


def test_aloha_adapter_preserves_dual_franka_16d_inputs() -> None:
    state = np.arange(16, dtype=np.float32)
    actions = np.arange(32, dtype=np.float32).reshape(2, 16)
    images = {
        "cam_high": np.zeros((3, 8, 8), dtype=np.uint8),
        "cam_left_wrist": np.ones((3, 8, 8), dtype=np.uint8),
        "cam_right_wrist": np.full((3, 8, 8), 2, dtype=np.uint8),
    }

    result = aloha_policy.AlohaInputs(adapt_to_pi=False)(
        {
            "state": state,
            "actions": actions,
            "images": images,
            "prompt": "serve the meal",
        }
    )

    np.testing.assert_array_equal(result["state"], state)
    np.testing.assert_array_equal(result["actions"], actions)
    assert set(result["image"]) == {
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    }
    np.testing.assert_array_equal(
        result["image"]["base_0_rgb"], np.zeros((8, 8, 3), dtype=np.uint8)
    )
    np.testing.assert_array_equal(
        result["image"]["left_wrist_0_rgb"], np.ones((8, 8, 3), dtype=np.uint8)
    )
    np.testing.assert_array_equal(
        result["image"]["right_wrist_0_rgb"], np.full((8, 8, 3), 2, dtype=np.uint8)
    )


def test_aloha_outputs_can_crop_dual_franka_actions_to_16d() -> None:
    model_actions = np.arange(64, dtype=np.float32).reshape(2, 32)

    result = aloha_policy.AlohaOutputs(adapt_to_pi=False, action_dim=16)(
        {"actions": model_actions}
    )

    np.testing.assert_array_equal(result["actions"], model_actions[:, :16])
