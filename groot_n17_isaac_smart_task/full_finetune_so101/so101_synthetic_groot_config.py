# SPDX-License-Identifier: Apache-2.0
"""GR00T N1.7 modality config for LeIsaac SO101 synthetic datasets.

The current LeIsaac synthetic datasets store SO101 follower proprioception and
actions as 6D LeRobot vectors:

    [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]

This config exposes the first five dimensions as a joint-space arm action and
the last dimension as an absolute gripper target, matching the SO100 example
shipped by Isaac-GR00T while keeping the dataset-specific camera names in
``meta/modality.json``.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


so101_synthetic_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "top",
            "wrist",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "single_arm",
            "gripper",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=[
            "single_arm",
            "gripper",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}


register_modality_config(
    so101_synthetic_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
