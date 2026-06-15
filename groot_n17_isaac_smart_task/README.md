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

The Franka task keeps the same SmartTask scene/lego/policy-observation key
contract, but replaces the SO101 robot with IsaacLab's copied
`FRANKA_PANDA_HIGH_PD_CFG`. This is meant to reduce the mismatch between GR00T
N1.7's OXE/DROID-style outputs and SO101's much smaller morphology.

The Franka route reuses the SmartTask scene assets, but keeps Franka-specific
robot, EEF, wrist camera, gripper, and action semantics. In particular,
`camera1` keeps the same policy key and GR00T bridge role, but Franka overrides
the physical sensor with a higher `Scene/franka_overview_camera` view. `camera3`
is the Franka wrist view and is not forced to see the lego in the initial pose.
The Franka root is initialized with a counterclockwise 90 degree yaw so the
official stack-task ready pose faces the SmartScene target direction more
naturally.

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

Then try a closed-loop probe:

```bash
cd /home/yzliu/smart_project
conda run -n isaaclab env \
  PYTHONPATH=/home/yzliu/smart_project/leisaac/source/leisaac \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  python experiments/groot_n17_isaac_smart_task/run_smart_task_closed_loop.py \
    --max-policy-calls 8 \
    --instruction "Pick up the red 2x4 lego brick."
```

For actual motion inspection or recorded videos, use more than one GR00T action
chunk. The examples below use `--max-policy-calls 8`; `--max-policy-calls 1` is
only a smoke test and usually produces a very short active-control video.

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
    --max-policy-calls 8 \
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
    --max-policy-calls 8 \
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
    --max-policy-calls 8 \
    --instruction "Pick up the red 2x4 lego brick."
```

To record the active Isaac viewport directly to a compact mp4, add
`--capture-video`. The defaults are intentionally small:
`960x540`, `15 fps`, `2 Mbps`.

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
    --max-policy-calls 8 \
    --capture-video \
    --capture-name franka_eef_probe \
    --instruction "Pick up the red 2x4 lego brick."
```

Captured videos are written under
`experiments/groot_n17_isaac_smart_task/runs/captures/` by default. To reduce
file size further, lower `--capture-bitrate-mbps`, lower
`--capture-width/--capture-height`, or set `--capture-every-nth-frames 2`.

## Camera Diagnostics

Before interpreting a policy run, inspect the actual policy images:

```bash
cd /home/yzliu/smart_project
conda run -n isaaclab env \
  PYTHONPATH=/home/yzliu/smart_project/experiments/groot_n17_isaac_smart_task:/home/yzliu/smart_project/leisaac/source/leisaac \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  python experiments/groot_n17_isaac_smart_task/run_smart_task_closed_loop.py \
    --robot franka \
    --control-mode eef \
    --headless \
    --debug-cameras-only
```

This does not connect to the GR00T bridge. It resets Isaac once, prints policy
camera tensor statistics, lists scene camera sensors and USD camera prims, and
saves `camera1.png`, `camera2.png`, and `camera3.png` under
`runs/camera_debug/`.

Current image semantics:

- `camera1`: top/global exterior view, sent to
  `video.exterior_image_1_left` in the zero-shot OXE/DROID probe. For SO101
  synthetic fine-tuning, the same source camera is named `top` in the custom
  GR00T modality config.
- `camera2`: fixed SmartScene left camera, retained in policy observations but
  not sent to GR00T in this OXE/DROID probe.
- `camera3`: robot wrist camera, sent to `video.wrist_image_left`.

For Franka, `camera3` should keep the wrist/gripper-view meaning. It should not
be rotated just to see the lego at the initial pose; the initial global target
view comes from `camera1`. If target-facing alignment is needed, rotate the
Franka root pose instead of changing the wrist camera's hand-relative offset.

## Four Video Probes

For reports, record the full 2x2 comparison:

```text
SO101  + joint
SO101  + EEF
Franka + joint
Franka + EEF
```

## SO101 Synthetic Fine-tuning

The current synthetic datasets under `dataset/` target the real deployment
embodiment: SOARM101/SO101. The workstation keeps LeRobot and Isaac-GR00T in
separate Python environments, so use a two-stage flow.

Stage 1 prepares GR00T-flavored LeRobot v2.1 dataset copies from the LeRobot
conda environment:

```bash
cd /home/yzliu/smart_project
conda run -n lerobot python \
  experiments/groot_n17_isaac_smart_task/train_so101_synthetic_groot.py \
    --instruction "Pick up the red 2x4 lego brick." \
    --force-prepare \
    --skip-stats \
    --prepare-only
```

Stage 2 generates GR00T stats and launches the official GR00T N1.7 fine-tune
entry point from the Isaac-GR00T virtualenv:

```bash
cd /home/yzliu/smart_project
/home/yzliu/Isaac-GR00T/.venv/bin/python \
  experiments/groot_n17_isaac_smart_task/train_so101_synthetic_groot.py \
    --skip-prepare \
    --max-steps 2000 \
    --save-steps 500 \
    --global-batch-size 32
```

The script keeps the original `dataset/` folders untouched. Prepared datasets
are written to `outputs/groot_so101_synthetic_datasets/`, and checkpoints are
written to `outputs/groot_so101_synthetic_finetune/`.

Training camera mapping:

- `observation.images.camera1` -> GR00T custom video key `top`
- `observation.images.camera3` -> GR00T custom video key `wrist`
- `observation.images.camera2` is left out of this GR00T fine-tuning run

For a quick metadata/video smoke test without launching training:

```bash
conda run -n lerobot python \
  experiments/groot_n17_isaac_smart_task/train_so101_synthetic_groot.py \
    --max-episodes 1 \
    --force-prepare \
    --skip-stats \
    --prepare-only \
    --instruction "Pick up the red 2x4 lego brick."
```

The helper attempts to reuse
`/home/yzliu/Isaac-GR00T/scripts/lerobot_conversion/convert_v3_to_v2.py`.
That official converter expects a specific LeRobot API; if the active
environment has a compatible mismatch, the helper falls back to a local
non-destructive converter with the same output layout.

All four commands share the same GR00T bridge. The mp4 files are written to
`experiments/groot_n17_isaac_smart_task/runs/captures/`.

SO101 joint-space baseline:

```bash
cd /home/yzliu/smart_project
conda run -n isaaclab env \
  PYTHONPATH=/home/yzliu/smart_project/experiments/groot_n17_isaac_smart_task:/home/yzliu/smart_project/leisaac/source/leisaac \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  python experiments/groot_n17_isaac_smart_task/run_smart_task_closed_loop.py \
    --robot so101 \
    --control-mode joint \
    --no-headless \
    --render-sleep-s 0.08 \
    --keep-open-s 60 \
    --max-policy-calls 8 \
    --capture-video \
    --capture-name so101_joint_probe \
    --instruction "Pick up the red 2x4 lego brick."
```

SO101 EEF/IK route:

```bash
cd /home/yzliu/smart_project
conda run -n isaaclab env \
  PYTHONPATH=/home/yzliu/smart_project/experiments/groot_n17_isaac_smart_task:/home/yzliu/smart_project/leisaac/source/leisaac \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  python experiments/groot_n17_isaac_smart_task/run_smart_task_closed_loop.py \
    --robot so101 \
    --control-mode eef \
    --no-headless \
    --render-sleep-s 0.08 \
    --keep-open-s 60 \
    --max-policy-calls 8 \
    --capture-video \
    --capture-name so101_eef_probe \
    --instruction "Pick up the red 2x4 lego brick."
```

Franka joint-space baseline:

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
    --max-policy-calls 8 \
    --capture-video \
    --capture-name franka_joint_probe \
    --instruction "Pick up the red 2x4 lego brick."
```

Franka EEF/IK route:

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
    --max-policy-calls 8 \
    --capture-video \
    --capture-name franka_eef_probe \
    --instruction "Pick up the red 2x4 lego brick."
```

## Current Mapping Hypothesis

The first probe uses the GR00T N1.7 base-model OXE/DROID schema:

- `camera1` top/global view -> `video.exterior_image_1_left`
- `camera3` -> `video.wrist_image_left`
- `camera2` remains available in the Isaac policy observation but is not sent to
  GR00T for the current OXE/DROID embodiment.
- `ee_frame_state` -> `state.eef_9d`
- SO101 6D joint state padded to 7D -> `state.joint_position`
- Franka 7D arm joint state -> `state.joint_position`
- gripper joint -> `state.gripper_position`
- SO101 `--control-mode joint`: returned `action.joint_position[..., :6]`
  -> LeIsaac SO101 joint command
- SO101 `--control-mode eef`: returned relative `action.eef_9d` plus
  `action.gripper_position` -> LeIsaac `mimic_so101leader` IK command
- Franka `--control-mode joint`: returned relative `action.joint_position`
  -> Franka 7D joint target plus binary gripper
- Franka `--control-mode eef`: returned relative `action.eef_9d` plus
  `action.gripper_position` -> Franka Differential IK target plus binary gripper

This is intentionally a probe, not a claim that SO101 equals the DROID
embodiment. If this fails physically but the action stream is nontrivial, the
next experiment should compare against a custom `new_embodiment` modality.
