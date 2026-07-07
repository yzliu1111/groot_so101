# SPDX-License-Identifier: Apache-2.0
"""Wrist-only GR00T N1.7 modality config for LeIsaac SO101 datasets.

Use this when the prepared dataset has only one wrist camera. The custom robot
still uses GR00T's ``NEW_EMBODIMENT`` slot, so the same config must be loaded
again when deploying a checkpoint trained with this file.
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


so101_synthetic_wrist_only_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
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
    so101_synthetic_wrist_only_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
