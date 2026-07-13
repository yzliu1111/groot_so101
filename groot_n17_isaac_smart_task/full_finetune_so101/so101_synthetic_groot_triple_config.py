# SPDX-License-Identifier: Apache-2.0
"""Three-camera GR00T N1.7 modality config for LeIsaac SO101 datasets.

Use this when the prepared dataset provides a front/top view, a left-side view,
and a wrist view. The dataset-specific feature names remain in
``meta/modality.json``; GR00T sees the stable semantic keys ``top``, ``left``,
and ``wrist``.
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


so101_synthetic_triple_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "top",
            "left",
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
    so101_synthetic_triple_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
