"""Task-level deployment choices shared by the Isaac runner and its tests."""

from __future__ import annotations

from typing import Any, Iterable


DEFAULT_SO101_TASK = "LeIsaac-SO101-SmartTask-v0"
DEFAULT_FRANKA_TASK = "Groot-Franka-SmartTask-v0"

# The base deployment task has an incomplete source LEGO USD, so the experiments
# runner keeps a process-local cuboid fallback for this task alone.
LEGACY_SMART_TASKS = frozenset({DEFAULT_SO101_TASK})
LEGACY_DEFAULT_INSTRUCTION = "Pick up the red 2x4 lego brick."


def default_task_for_robot(robot: str) -> str:
    """Return the runner's backwards-compatible default Gym task ID."""

    if robot == "so101":
        return DEFAULT_SO101_TASK
    if robot == "franka":
        return DEFAULT_FRANKA_TASK
    raise ValueError(f"unsupported robot {robot!r}")


def validate_robot_task(robot: str, task: str) -> None:
    """Reject known task families that would use the wrong robot action path."""

    if task.startswith("LeIsaac-SO101-") and robot != "so101":
        raise ValueError(f"task {task!r} requires --robot so101")
    if task.startswith("Groot-Franka-") and robot != "franka":
        raise ValueError(f"task {task!r} requires --robot franka")
    if task.endswith("-Mimic-v0"):
        raise ValueError(
            f"task {task!r} is a data-generation Mimic environment; select its normal deployment task"
        )


def is_leisaac_smart_task(task: str) -> bool:
    """Return whether a Gym task belongs to the LeIsaac SO101 SmartTask family."""

    return task.startswith("LeIsaac-SO101-SmartTask")


def uses_legacy_smart_task_asset_patch(task: str) -> bool:
    """Return whether the old red 2x4 portability patch is valid for this task."""

    return task in LEGACY_SMART_TASKS


def resolve_smart_target_asset(task: str, requested_asset: str) -> str:
    """Resolve ``auto`` without applying the red fallback to another task."""

    if requested_asset != "auto":
        return requested_asset
    return "cuboid" if uses_legacy_smart_task_asset_patch(task) else "scene"


def target_name_from_env_cfg(env_cfg: Any) -> str | None:
    """Read the target object from a task's success termination when available."""

    terminations = getattr(env_cfg, "terminations", None)
    success = getattr(terminations, "success", None)
    params = getattr(success, "params", None)
    if not isinstance(params, dict):
        return None

    for key in ("object_cfg", "target_object_cfg", "target_cfg"):
        entity_cfg = params.get(key)
        name = getattr(entity_cfg, "name", None)
        if isinstance(name, str) and name:
            return name
    return None


def resolve_task_instruction(requested_instruction: str | None, env_cfg: Any, task: str) -> str:
    """Use an explicit checkpoint instruction, otherwise the selected task description."""

    if requested_instruction is not None:
        instruction = requested_instruction.strip()
        if not instruction:
            raise ValueError("--instruction cannot be empty")
        return instruction

    if task == DEFAULT_FRANKA_TASK or task in LEGACY_SMART_TASKS:
        return LEGACY_DEFAULT_INSTRUCTION
    task_description = getattr(env_cfg, "task_description", None)
    if isinstance(task_description, str) and task_description.strip():
        return task_description.strip()
    raise ValueError(
        f"task {task!r} does not expose task_description; pass the checkpoint instruction with --instruction"
    )


def resolve_target_name(
    available_names: Iterable[str],
    requested_name: str,
    configured_name: str | None = None,
) -> str:
    """Resolve a metric/debug target without guessing in a multi-object scene."""

    names = sorted(set(available_names))
    if requested_name != "auto":
        if requested_name in names:
            return requested_name
        raise KeyError(
            f"target object {requested_name!r} is not in scene rigid objects; "
            f"available rigid objects are {names}"
        )

    if configured_name is not None:
        if configured_name in names:
            return configured_name
        raise KeyError(
            f"task config target {configured_name!r} is not in scene rigid objects; "
            f"available rigid objects are {names}"
        )

    lego_matches = [
        name
        for name in names
        if "lego" in name.lower() and ("brick" in name.lower() or "block" in name.lower())
    ]
    if len(lego_matches) == 1:
        return lego_matches[0]
    if len(lego_matches) > 1:
        raise KeyError(
            "could not auto-detect one target in a multi-LEGO scene; "
            f"pass --target-object-key explicitly, candidates are {lego_matches}"
        )

    raise KeyError(
        "could not auto-detect SmartTask target object; "
        f"available rigid objects are {names}"
    )
