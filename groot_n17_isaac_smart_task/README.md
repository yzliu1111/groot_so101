# GR00T N1.7 SmartTask Native Probe

This experiment keeps the GR00T N1.7 path out of LeIsaac's LeRobot policy stack.
It reuses the existing LeIsaac IsaacLab task and USD assets, but talks to a tiny
GR00T-side bridge over a standard-library socket.

## Why This Shape

- IsaacLab env: `conda isaaclab`
- GR00T model: `~/Isaac-GR00T/.venv`
- No GR00T imports inside IsaacLab.
- No LeIsaac policy client changes.
- No dependencies installed into either environment.
- The Isaac runner prepends the copied project dependencies under
  `leisaac/dependencies/IsaacLab/source/*` to `sys.path`, so this experiment
  prefers the company workspace copy instead of silently using another IsaacLab
  source checkout on the machine.

## Task Variants

This folder now contains two isolated Isaac task routes:

- `--robot so101` uses the original LeIsaac task
  `LeIsaac-SO101-SmartTask-v0`.
- `--robot franka` imports the local `franka_smart_task` package and registers
  `Groot-Franka-SmartTask-v0`.

The Franka task keeps the same SmartTask scene/lego/camera observation contract,
but replaces the SO101 robot with IsaacLab's copied `FRANKA_PANDA_HIGH_PD_CFG`.
This is meant to reduce the mismatch between GR00T N1.7's OXE/DROID-style
outputs and SO101's much smaller morphology.

## Terminal 1: GR00T Bridge

```bash
cd /home/yzliu/smart_project
/home/yzliu/Isaac-GR00T/.venv/bin/python \
  experiments/groot_n17_isaac_smart_task/groot_bridge_server.py \
  --model-path nvidia/GR00T-N1.7-3B \
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
  --device cuda
```

The bridge listens on `127.0.0.1:5577`.

Note: avoid `--offline` for now. The local GR00T snapshot still references
`nvidia/Cosmos-Reason2-2B` in `config.json`, and the Transformers processor path
may call HuggingFace metadata helpers even when model weights are cached. This is
the same loading path that passed the official standalone smoke test.

## Terminal 2: Isaac SmartTask Probe

Dry-run first. This sends one observation to GR00T and prints action keys, but
does not step the robot:

```bash
cd /home/yzliu/smart_project
conda run -n isaaclab env \
  PYTHONPATH=/home/yzliu/smart_project/leisaac/source/leisaac \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  python experiments/groot_n17_isaac_smart_task/run_smart_task_closed_loop.py \
    --dry-run \
    --max-policy-calls 1 \
    --instruction "Pick up the red 2x4 lego brick."
```

Then try one short closed-loop chunk:

```bash
cd /home/yzliu/smart_project
conda run -n isaaclab env \
  PYTHONPATH=/home/yzliu/smart_project/leisaac/source/leisaac \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  python experiments/groot_n17_isaac_smart_task/run_smart_task_closed_loop.py \
    --max-policy-calls 1 \
    --instruction "Pick up the red 2x4 lego brick."
```

## Viewport Mode

To watch the run in the Isaac Sim viewport, use `--no-headless`. The
`--keep-open-s` option keeps the window open after the short run so you can
inspect the final state:

```bash
cd /home/yzliu/smart_project
conda run -n isaaclab env \
  PYTHONPATH=/home/yzliu/smart_project/leisaac/source/leisaac \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  python experiments/groot_n17_isaac_smart_task/run_smart_task_closed_loop.py \
    --no-headless \
    --keep-open-s 60 \
    --max-policy-calls 1 \
    --instruction "Pick up the red 2x4 lego brick."
```

The runner disables SmartTask's built-in success termination by default. That
termination is currently too loose: it only compares lego height against robot
base height, so it can fire before any real grasp happens. The runner instead
prints diagnostic metrics:

- `jaw_to_lego`: distance from gripper jaw frame to lego root
- `gripper`: gripper joint position
- `lego_z_minus_base`: lego height relative to robot base

Use a longer visible probe to inspect actual motion:

```bash
cd /home/yzliu/smart_project
conda run -n isaaclab env \
  PYTHONPATH=/home/yzliu/smart_project/leisaac/source/leisaac \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  python experiments/groot_n17_isaac_smart_task/run_smart_task_closed_loop.py \
    --no-headless \
    --render-sleep-s 0.08 \
    --keep-open-s 60 \
    --max-policy-calls 8 \
    --instruction "Pick up the red 2x4 lego brick."
```

`--keep-open-s` only keeps the viewport open after active control finishes. It
does not keep asking GR00T for actions. For visible motion, increase
`--max-policy-calls`. By default `--action-horizon 0` executes the full returned
GR00T action chunk. Set `--action-horizon N` only when you intentionally want to
truncate each chunk.

To test the more embodiment-neutral EEF route, add `--control-mode eef`. This
uses GR00T's `eef_9d` and `gripper_position` output through LeIsaac's
`mimic_so101leader` Differential IK action mode:

```bash
cd /home/yzliu/smart_project
conda run -n isaaclab env \
  PYTHONPATH=/home/yzliu/smart_project/leisaac/source/leisaac \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  python experiments/groot_n17_isaac_smart_task/run_smart_task_closed_loop.py \
    --control-mode eef \
    --no-headless \
    --render-sleep-s 0.08 \
    --keep-open-s 60 \
    --max-policy-calls 1 \
    --instruction "Pick up the red 2x4 lego brick."
```

## Franka SmartTask Variant

The Franka route is the cleaner zero-shot probe for GR00T N1.7 because Panda has
7 arm joints and a parallel gripper, matching the OXE/DROID action schema more
closely than SO101. Start the same GR00T bridge in terminal 1, then run:

```bash
cd /home/yzliu/smart_project
conda run -n isaaclab env \
  PYTHONPATH=/home/yzliu/smart_project/experiments/groot_n17_isaac_smart_task:/home/yzliu/smart_project/leisaac/source/leisaac \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  python experiments/groot_n17_isaac_smart_task/run_smart_task_closed_loop.py \
    --robot franka \
    --control-mode eef \
    --no-headless \
    --render-sleep-s 0.08 \
    --keep-open-s 60 \
    --max-policy-calls 1 \
    --instruction "Pick up the red 2x4 lego brick."
```

For a joint-space baseline on the same Franka task:

```bash
cd /home/yzliu/smart_project
conda run -n isaaclab env \
  PYTHONPATH=/home/yzliu/smart_project/experiments/groot_n17_isaac_smart_task:/home/yzliu/smart_project/leisaac/source/leisaac \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  python experiments/groot_n17_isaac_smart_task/run_smart_task_closed_loop.py \
    --robot franka \
    --control-mode joint \
    --no-headless \
    --render-sleep-s 0.08 \
    --keep-open-s 60 \
    --max-policy-calls 1 \
    --instruction "Pick up the red 2x4 lego brick."
```

## Current Mapping Hypothesis

The first probe uses the GR00T N1.7 base-model OXE/DROID schema:

- `camera1` -> `video.exterior_image_1_left`
- `camera3` -> `video.wrist_image_left`
- `ee_frame_state` -> `state.eef_9d`
- SO101 6D joint state padded to 7D -> `state.joint_position`
- gripper joint -> `state.gripper_position`
- default `--control-mode joint`: returned `action.joint_position[..., :6]` -> LeIsaac SO101 joint command
- optional `--control-mode eef`: returned relative `action.eef_9d` plus `action.gripper_position` -> LeIsaac `mimic_so101leader` IK command

This is intentionally a probe, not a claim that SO101 equals the DROID
embodiment. If this fails physically but the action stream is nontrivial, the
next experiment should compare against a custom `new_embodiment` modality.
