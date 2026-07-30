# SmartTask 公司检验操作手册

这份文件只负责一件事：在公司机器上快速复现已经运行过的五类结果。

原理、代码构成和诊断方法不在这里展开，见
[代码地图与诊断说明](CODE_MAP_AND_DIAGNOSTICS_ZH.md) 和
[技术契约](../TECHNICAL_CONTRACTS_ZH.md)。

## 0. 已核对的五张运行卡

| case | 当前状态 | 本手册复现的现象 |
|---|---|---|
| sim002 | 已完成 | 抓取、抬起并稳定保持 |
| sim004 | 已完成 | 横放场景抓取、抬起并稳定保持 |
| real003 | 正在继续诊断 | 已出现一次抓取并保持；不能据此计算真机模型成功率 |
| real007 | 已完成阶段性定位 | 能抓取和抬起，随后掉落，place 失败 |
| real001 | 已完成阶段性定位 | 接近目标，但夹爪保持打开，不抓取、不抬起 |

这里的“复现”是固定 checkpoint、场景、reset、相机、材质、控制参数和双进程 seed，
复现对应的成功路径或失败特征。CUDA diffusion 和闭环物理不保证逐位一致，因此同一条命令
仍应多跑几次，并保存每次 trace 和图像。

这些卡固定的是已验证的历史行为，不追求逐字符复制旧 shell。后来加入 CLI 的旧默认值
（例如 material、settle=0、gripper step=0）会在卡中显式钉住，避免代码默认值以后漂移；
任何会改变历史行为的新机制都只出现在它实际验证过的 case 中。

## 1. 运行纪律

1. 一次只运行一张卡；开始前关闭其他 Isaac Sim 和旧 bridge。
2. 每个 trial 都启动 fresh bridge；bridge 与 runner 分别在两个新终端运行。
3. seed 必须直接出现在启动参数中。本手册不使用 shell seed 变量或代码默认值。
4. 每张卡默认写入 `trial01`。重复运行时，同时修改两个终端中的 trial 编号；不要覆盖旧证据。
5. 公司机器使用每段中的 `target` 路径。本机调试时，只把第一行替换为：

   ```bash
   source /home/yzliu/physical_ai/company_project/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh local
   ```

6. 五张运行卡统一使用 Isaac GUI viewport MP4。远程桌面不稳定时见第
   9 节切换为不录像的 headless 运行；不要同时启动第二个 Isaac 实例。
7. bridge 出现 `listening` 后才运行终端 2。runner 结束后回到终端 1 按 `Ctrl+C` 关闭 bridge。

## 2. sim002：抓取、抬起、稳定保持

### 终端 1：fresh bridge

```bash
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target
conda deactivate 2>/dev/null || true
unset PYTHONPATH PYTHONHOME PYTHONNOUSERSITE PYTHONDONTWRITEBYTECODE ISAAC_PATH ISAACLAB_PATH

cd "$SMART_PROJECT"
CHECKPOINT="$SMART_PROJECT/outputs/aws-0715/groot_so101_synthetic_finetune/aws_sim_run_002_wrist_notray_pick/aws_sim_so101_run_002_wrist_notray_pick/checkpoint-10000"
RUN_DIR="$SMART_PROJECT/experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/runs/company_eval/sim002_seed42_trial01"
if [[ ! -d "$CHECKPOINT" ]]; then printf '[FAIL] missing checkpoint: %s\n' "$CHECKPOINT" >&2; exit 1; fi
mkdir -p "$SMART_PROJECT/experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/runs/company_eval"
if [[ -e "$RUN_DIR" ]]; then printf '[FAIL] change trial number; run dir exists: %s\n' "$RUN_DIR" >&2; exit 1; fi
mkdir "$RUN_DIR"
set -o pipefail

"$GROOT_ROOT/.venv/bin/python" \
  experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/groot_bridge_server.py \
    --deployment-mode so101-finetuned \
    --camera-layout wrist-only \
    --modality-config-path "$EXPERIMENT_ROOT/full_finetune_so101/so101_synthetic_groot_wrist_only_config.py" \
    --model-path "$CHECKPOINT" \
    --host 127.0.0.1 \
    --port 5577 \
    --device cuda \
    --seed 42 \
  2>&1 | tee "$RUN_DIR/bridge.log"
```

通过标记：

```text
[bridge] inference seed: 42
[bridge] camera layout: wrist-only
[bridge] listening on 127.0.0.1:5577
```

### 终端 2：Isaac runner

```bash
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LEISAAC_ENV"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$SMART_PROJECT/experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task:$LEISAAC_ROOT/source/leisaac:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

cd "$SMART_PROJECT"
RUN_DIR="$SMART_PROJECT/experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/runs/company_eval/sim002_seed42_trial01"
if [[ ! -d "$RUN_DIR" ]]; then printf '[FAIL] terminal 1 did not create: %s\n' "$RUN_DIR" >&2; exit 1; fi
set -o pipefail

"$CONDA_PREFIX/bin/python" \
  experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/run_smart_task_closed_loop.py \
    --robot so101 \
    --deployment-mode so101-finetuned \
    --control-mode joint \
    --camera-layout wrist-only \
    --isaac-wrist-camera-key camera1 \
    --so101-checkpoint-joint-units lerobot_motor_units \
    --so101-initial-pose dataset-start \
    --so101-initial-dataset-state \
      -2.288494 -6.753181 10.013950 90.934647 -53.411201 1.186470 \
    --task LeIsaac-SO101-SmartTask-v0 \
    --scene-profile table-red24 \
    --red24-material asset \
    --so101-robot-material asset \
    --target-object-key red_2x4_lego_brick \
    --instruction "pick up block" \
    --bridge-host 127.0.0.1 \
    --bridge-port 5577 \
    --so101-arm-target-scale 1 \
    --so101-max-arm-step-rad 0.08 \
    --so101-max-gripper-step-rad 0 \
    --dynamic-reset-gripper-effort-limit \
    --max-policy-calls 20 \
    --action-horizon 0 \
    --policy-action-hz 30 \
    --warmup-frames 16 \
    --settle-env-steps 0 \
    --seed 42 \
    --ignore-terminations \
    --no-headless \
    --render-sleep-s 0.01 \
    --capture-video \
    --capture-dir "$RUN_DIR" \
    --capture-name sim002_seed42_fullchunk_20calls \
    --capture-width 960 \
    --capture-height 540 \
    --capture-fps 15 \
    --capture-bitrate-mbps 2 \
    --capture-every-nth-frames 1 \
    --capture-wait-timeout-s 90 \
    --action-trace-jsonl "$RUN_DIR/action_trace.jsonl" \
  2>&1 | tee "$RUN_DIR/rollout.log"
```

预期结果：完整 20 calls 后仍夹持红色 2x4。已验证基线中 LEGO 最终高度约
`0.11965 m`，没有在末段掉落。这里不能加入 real003 的 gripper `0.04/0.1` 配置。

## 3. sim004：横放场景抓取、抬起、稳定保持

注意：canonical case 名是 sim004，但 checkpoint 的历史父目录名包含 `sim_run_003`；
不要因此换错 checkpoint。成功场景必须是 `tray-red24-d4-horizontal`，旧 vertical 场景不是
这个正例。

### 终端 1：fresh bridge

```bash
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target
conda deactivate 2>/dev/null || true
unset PYTHONPATH PYTHONHOME PYTHONNOUSERSITE PYTHONDONTWRITEBYTECODE ISAAC_PATH ISAACLAB_PATH

cd "$SMART_PROJECT"
CHECKPOINT="$SMART_PROJECT/outputs/aws-0715/groot_so101_synthetic_finetune/aws_sim_run_003_pick_tri_notray/aws_sim_so101_run_003_pick_tri_notray/checkpoint-7500"
RUN_DIR="$SMART_PROJECT/experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/runs/company_eval/sim004_seed42_trial01"
if [[ ! -d "$CHECKPOINT" ]]; then printf '[FAIL] missing checkpoint: %s\n' "$CHECKPOINT" >&2; exit 1; fi
mkdir -p "$SMART_PROJECT/experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/runs/company_eval"
if [[ -e "$RUN_DIR" ]]; then printf '[FAIL] change trial number; run dir exists: %s\n' "$RUN_DIR" >&2; exit 1; fi
mkdir "$RUN_DIR"
set -o pipefail

"$GROOT_ROOT/.venv/bin/python" \
  experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/groot_bridge_server.py \
    --deployment-mode so101-finetuned \
    --camera-layout triple \
    --modality-config-path "$EXPERIMENT_ROOT/full_finetune_so101/so101_synthetic_groot_triple_config.py" \
    --model-path "$CHECKPOINT" \
    --host 127.0.0.1 \
    --port 5577 \
    --device cuda \
    --seed 42 \
  2>&1 | tee "$RUN_DIR/bridge.log"
```

通过标记：

```text
[bridge] inference seed: 42
[bridge] camera layout: triple
[bridge] listening on 127.0.0.1:5577
```

### 终端 2：Isaac runner

```bash
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LEISAAC_ENV"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$SMART_PROJECT/experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task:$LEISAAC_ROOT/source/leisaac:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

cd "$SMART_PROJECT"
RUN_DIR="$SMART_PROJECT/experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/runs/company_eval/sim004_seed42_trial01"
if [[ ! -d "$RUN_DIR" ]]; then printf '[FAIL] terminal 1 did not create: %s\n' "$RUN_DIR" >&2; exit 1; fi
set -o pipefail

"$CONDA_PREFIX/bin/python" \
  experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/run_smart_task_closed_loop.py \
    --robot so101 \
    --deployment-mode so101-finetuned \
    --control-mode joint \
    --camera-layout triple \
    --inject-so101-eval-cameras \
    --isaac-front-camera-key camera3 \
    --isaac-left-camera-key camera2 \
    --isaac-wrist-camera-key camera1 \
    --so101-checkpoint-joint-units lerobot_motor_units \
    --so101-initial-pose dataset-start \
    --so101-initial-dataset-state \
      -8.875954 -66.846802 87.866806 36.950424 -1.436111 21.493200 \
    --task LeIsaac-SO101-SmartTask-v0 \
    --scene-profile tray-red24-d4-horizontal \
    --red24-material asset \
    --so101-robot-material asset \
    --target-object-key red_2x4_lego_brick \
    --instruction "pick up block" \
    --bridge-host 127.0.0.1 \
    --bridge-port 5577 \
    --so101-arm-target-scale 1 \
    --so101-max-arm-step-rad 0.08 \
    --so101-max-gripper-step-rad 0 \
    --dynamic-reset-gripper-effort-limit \
    --max-policy-calls 20 \
    --action-horizon 0 \
    --policy-action-hz 30 \
    --warmup-frames 16 \
    --settle-env-steps 0 \
    --seed 42 \
    --ignore-terminations \
    --no-headless \
    --render-sleep-s 0.01 \
    --capture-video \
    --capture-dir "$RUN_DIR" \
    --capture-name sim004_d4_horizontal_seed42_fullchunk_20calls \
    --capture-width 960 \
    --capture-height 540 \
    --capture-camera-path /World/envs/env_0/Scene/camera_front_xform/camera_front \
    --capture-fps 15 \
    --capture-bitrate-mbps 2 \
    --capture-every-nth-frames 1 \
    --capture-wait-timeout-s 90 \
    --debug-cameras \
    --debug-camera-frame-dir "$RUN_DIR/camera_preflight" \
    --action-trace-jsonl "$RUN_DIR/action_trace.jsonl" \
  2>&1 | tee "$RUN_DIR/rollout.log"
```

预期结果：抓取横放红色 2x4，峰值净抬高约 `5.18 cm`，最终净抬高约 `4.19 cm`，
结束时仍保持。这里同样不能加入 real003 的 gripper 限幅或固定 effort。

## 4. real003：当前抓取成功诊断基线

这是真机数据 checkpoint 在 Isaac 中的部署诊断，不是真机模型质量验收。当前只确认 seed
`43/42` 的一次 live inference 出现抓取并保持；仍需多次复现并继续定位侧面接触侵入。

### 终端 1：fresh bridge

```bash
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target
conda deactivate 2>/dev/null || true
unset PYTHONPATH PYTHONHOME PYTHONNOUSERSITE PYTHONDONTWRITEBYTECODE ISAAC_PATH ISAACLAB_PATH

cd "$SMART_PROJECT"
CHECKPOINT="$SMART_PROJECT/outputs/aws-0715/groot_so101_synthetic_finetune/aws_real_run_003_pick_tri_notray/aws_real_so101_run_003_pick_tri_notray/checkpoint-10000"
RUN_DIR="$SMART_PROJECT/experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/runs/company_eval/real003_bridge43_isaac42_trial01"
if [[ ! -d "$CHECKPOINT" ]]; then printf '[FAIL] missing checkpoint: %s\n' "$CHECKPOINT" >&2; exit 1; fi
mkdir -p "$SMART_PROJECT/experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/runs/company_eval"
if [[ -e "$RUN_DIR" ]]; then printf '[FAIL] change trial number; run dir exists: %s\n' "$RUN_DIR" >&2; exit 1; fi
mkdir "$RUN_DIR"
set -o pipefail

"$GROOT_ROOT/.venv/bin/python" \
  experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/groot_bridge_server.py \
    --deployment-mode so101-finetuned \
    --camera-layout triple \
    --modality-config-path "$EXPERIMENT_ROOT/full_finetune_so101/so101_synthetic_groot_triple_config.py" \
    --model-path "$CHECKPOINT" \
    --host 127.0.0.1 \
    --port 5577 \
    --device cuda \
    --seed 43 \
  2>&1 | tee "$RUN_DIR/bridge.log"
```

通过标记：

```text
[bridge] inference seed: 43
[bridge] camera layout: triple
[bridge] listening on 127.0.0.1:5577
```

### 终端 2：Isaac runner

```bash
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LEISAAC_ENV"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$SMART_PROJECT/experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task:$LEISAAC_ROOT/source/leisaac:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

cd "$SMART_PROJECT"
RUN_DIR="$SMART_PROJECT/experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/runs/company_eval/real003_bridge43_isaac42_trial01"
if [[ ! -d "$RUN_DIR" ]]; then printf '[FAIL] terminal 1 did not create: %s\n' "$RUN_DIR" >&2; exit 1; fi
set -o pipefail

"$CONDA_PREFIX/bin/python" \
  experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/run_smart_task_closed_loop.py \
    --robot so101 \
    --deployment-mode so101-finetuned \
    --control-mode joint \
    --camera-layout triple \
    --inject-so101-eval-cameras \
    --isaac-front-camera-key camera2 \
    --isaac-left-camera-key camera1 \
    --isaac-wrist-camera-key camera3 \
    --so101-checkpoint-joint-units degrees \
    --so101-initial-pose dataset-start \
    --so101-initial-dataset-state \
      2.505495 -52.747253 55.472527 94.945053 -96.659340 1.371742 \
    --task LeIsaac-SO101-SmartTask-v0 \
    --scene-profile table-red24-d3-horizontal-right10mm-front5mm \
    --red24-material real-red \
    --so101-robot-material real-white \
    --target-object-key red_2x4_lego_brick \
    --instruction "pick up block" \
    --bridge-host 127.0.0.1 \
    --bridge-port 5577 \
    --so101-arm-target-scale 1 \
    --so101-max-arm-step-rad 0.08 \
    --so101-max-gripper-step-rad 0.04 \
    --so101-gripper-effort-limit-sim 0.1 \
    --no-dynamic-reset-gripper-effort-limit \
    --max-policy-calls 30 \
    --action-horizon 0 \
    --policy-action-hz 30 \
    --warmup-frames 16 \
    --settle-env-steps 30 \
    --seed 42 \
    --ignore-terminations \
    --no-headless \
    --render-sleep-s 0.01 \
    --capture-video \
    --capture-dir "$RUN_DIR" \
    --capture-name real003_bridge43_isaac42_gripper_guard_30calls \
    --capture-width 960 \
    --capture-height 540 \
    --capture-camera-path /World/envs/env_0/Scene/camera_left_xform/camera_left \
    --capture-fps 15 \
    --capture-bitrate-mbps 2 \
    --capture-every-nth-frames 1 \
    --capture-wait-timeout-s 90 \
    --debug-cameras \
    --debug-camera-frame-dir "$RUN_DIR/camera_preflight" \
    --contact-probe \
    --action-trace-jsonl "$RUN_DIR/action_trace.jsonl" \
  2>&1 | tee "$RUN_DIR/rollout.log"
```

预期结果：已验证 trial 的 LEGO 最终净抬高为 `37.592 mm`，并在结束时保持。本卡现在和
另外四张卡一样生成 GUI viewport MP4；录像视角固定为物理 left camera，只用于证据，
不改变 policy camera mapping。历史命令的 material 请求是 `auto/auto`，解析结果为
`real-red/real-white`；本卡直接固定解析后的值，避免默认逻辑漂移。

## 5. real007：pick 成功，place 失败

必须使用 `multi-lego-tray-real007-frame5`。后来的 `forward2cm` trace 不完整且没有复现
成功 pick，不能替代这张卡。

### 终端 1：fresh bridge

```bash
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target
conda deactivate 2>/dev/null || true
unset PYTHONPATH PYTHONHOME PYTHONNOUSERSITE PYTHONDONTWRITEBYTECODE ISAAC_PATH ISAACLAB_PATH

cd "$SMART_PROJECT"
CHECKPOINT="$SMART_PROJECT/outputs/aws-0715/groot_so101_synthetic_finetune/aws_real_run_007_3block/aws_real_so101_run_007_3block/checkpoint-9000"
RUN_DIR="$SMART_PROJECT/experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/runs/company_eval/real007_seed42_trial01"
if [[ ! -d "$CHECKPOINT" ]]; then printf '[FAIL] missing checkpoint: %s\n' "$CHECKPOINT" >&2; exit 1; fi
mkdir -p "$SMART_PROJECT/experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/runs/company_eval"
if [[ -e "$RUN_DIR" ]]; then printf '[FAIL] change trial number; run dir exists: %s\n' "$RUN_DIR" >&2; exit 1; fi
mkdir "$RUN_DIR"
set -o pipefail

"$GROOT_ROOT/.venv/bin/python" \
  experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/groot_bridge_server.py \
    --deployment-mode so101-finetuned \
    --camera-layout triple \
    --modality-config-path "$EXPERIMENT_ROOT/full_finetune_so101/so101_synthetic_groot_triple_config.py" \
    --model-path "$CHECKPOINT" \
    --host 127.0.0.1 \
    --port 5577 \
    --device cuda \
    --seed 42 \
  2>&1 | tee "$RUN_DIR/bridge.log"
```

通过标记：

```text
[bridge] inference seed: 42
[bridge] camera layout: triple
[bridge] listening on 127.0.0.1:5577
```

### 终端 2：Isaac runner

```bash
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LEISAAC_ENV"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$SMART_PROJECT/experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task:$LEISAAC_ROOT/source/leisaac:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

cd "$SMART_PROJECT"
RUN_DIR="$SMART_PROJECT/experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/runs/company_eval/real007_seed42_trial01"
if [[ ! -d "$RUN_DIR" ]]; then printf '[FAIL] terminal 1 did not create: %s\n' "$RUN_DIR" >&2; exit 1; fi
set -o pipefail

"$CONDA_PREFIX/bin/python" \
  experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/run_smart_task_closed_loop.py \
    --robot so101 \
    --deployment-mode so101-finetuned \
    --control-mode joint \
    --camera-layout triple \
    --inject-so101-eval-cameras \
    --isaac-front-camera-key camera3 \
    --isaac-left-camera-key camera2 \
    --isaac-wrist-camera-key camera1 \
    --so101-checkpoint-joint-units degrees \
    --so101-initial-pose dataset-start \
    --so101-initial-dataset-state \
      -2.7692308 -2.0219781 12.219780 92.571426 -95.252747 1.3031551 \
    --task LeIsaac-SO101-SmartTask-v0 \
    --scene-profile multi-lego-tray-real007-frame5 \
    --red24-material asset \
    --so101-robot-material asset \
    --target-object-key blue_2x4_lego_brick \
    --instruction "pick up the blue block and place it on the tray" \
    --bridge-host 127.0.0.1 \
    --bridge-port 5577 \
    --so101-arm-target-scale 1 \
    --so101-max-arm-step-rad 0.08 \
    --so101-max-gripper-step-rad 0 \
    --dynamic-reset-gripper-effort-limit \
    --max-policy-calls 40 \
    --action-horizon 0 \
    --policy-action-hz 30 \
    --warmup-frames 16 \
    --settle-env-steps 0 \
    --seed 42 \
    --ignore-terminations \
    --no-headless \
    --render-sleep-s 0.01 \
    --capture-video \
    --capture-dir "$RUN_DIR" \
    --capture-name real007_ep67_seed42_fullchunk_40calls \
    --capture-width 960 \
    --capture-height 540 \
    --capture-camera-path /World/envs/env_0/Scene/camera_front_xform/camera_front \
    --capture-fps 15 \
    --capture-bitrate-mbps 2 \
    --capture-every-nth-frames 1 \
    --capture-wait-timeout-s 90 \
    --debug-cameras \
    --debug-camera-frame-dir "$RUN_DIR/camera_preflight" \
    --action-trace-jsonl "$RUN_DIR/action_trace.jsonl" \
  2>&1 | tee "$RUN_DIR/rollout.log"
```

预期结果：约 call 26 闭爪，call 27–34 夹持并抬起；峰值净抬高约 `5.25 cm`。随后下降、
开爪并掉落，最终没有 place。相机 slot 必须是
`top<-camera3, left<-camera2, wrist<-camera1`，不能套用 real003 的排列。

## 6. real001：夹爪保持打开

这张卡固定的是 7 月 29 日修正后的 ep46 reset、真红 LEGO 和白色 SO101 外观。预期结果
本身是失败特征：机械臂接近目标，但夹爪不闭合。

### 终端 1：fresh bridge

```bash
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target
conda deactivate 2>/dev/null || true
unset PYTHONPATH PYTHONHOME PYTHONNOUSERSITE PYTHONDONTWRITEBYTECODE ISAAC_PATH ISAACLAB_PATH

cd "$SMART_PROJECT"
CHECKPOINT="$SMART_PROJECT/outputs/aws-0715/groot_so101_synthetic_finetune/aws_real_run_001_wrist_pick/aws_real_so101_run_001_wrist_pick/checkpoint-10000"
RUN_DIR="$SMART_PROJECT/experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/runs/company_eval/real001_seed42_trial01"
if [[ ! -d "$CHECKPOINT" ]]; then printf '[FAIL] missing checkpoint: %s\n' "$CHECKPOINT" >&2; exit 1; fi
mkdir -p "$SMART_PROJECT/experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/runs/company_eval"
if [[ -e "$RUN_DIR" ]]; then printf '[FAIL] change trial number; run dir exists: %s\n' "$RUN_DIR" >&2; exit 1; fi
mkdir "$RUN_DIR"
set -o pipefail

"$GROOT_ROOT/.venv/bin/python" \
  experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/groot_bridge_server.py \
    --deployment-mode so101-finetuned \
    --camera-layout wrist-only \
    --modality-config-path "$EXPERIMENT_ROOT/full_finetune_so101/so101_synthetic_groot_wrist_only_config.py" \
    --model-path "$CHECKPOINT" \
    --host 127.0.0.1 \
    --port 5581 \
    --device cuda \
    --seed 42 \
  2>&1 | tee "$RUN_DIR/bridge.log"
```

通过标记：

```text
[bridge] inference seed: 42
[bridge] camera layout: wrist-only
[bridge] listening on 127.0.0.1:5581
```

### 终端 2：Isaac runner

```bash
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LEISAAC_ENV"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$SMART_PROJECT/experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task:$LEISAAC_ROOT/source/leisaac:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

cd "$SMART_PROJECT"
RUN_DIR="$SMART_PROJECT/experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/runs/company_eval/real001_seed42_trial01"
if [[ ! -d "$RUN_DIR" ]]; then printf '[FAIL] terminal 1 did not create: %s\n' "$RUN_DIR" >&2; exit 1; fi
set -o pipefail

"$CONDA_PREFIX/bin/python" \
  experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/run_smart_task_closed_loop.py \
    --robot so101 \
    --deployment-mode so101-finetuned \
    --control-mode joint \
    --camera-layout wrist-only \
    --isaac-wrist-camera-key camera1 \
    --so101-checkpoint-joint-units degrees \
    --so101-initial-pose dataset-start \
    --so101-initial-dataset-state \
      1.714286 1.758242 5.186813 92.659340 -95.780220 1.234568 \
    --task LeIsaac-SO101-SmartTask-v0 \
    --scene-profile tray-red24 \
    --red24-material real-red \
    --so101-robot-material real-white \
    --target-object-key red_2x4_lego_brick \
    --instruction "pick up block" \
    --bridge-host 127.0.0.1 \
    --bridge-port 5581 \
    --so101-arm-target-scale 1 \
    --so101-max-arm-step-rad 0.08 \
    --so101-max-gripper-step-rad 0 \
    --no-dynamic-reset-gripper-effort-limit \
    --max-policy-calls 30 \
    --action-horizon 0 \
    --policy-action-hz 30 \
    --warmup-frames 16 \
    --settle-env-steps 0 \
    --seed 42 \
    --ignore-terminations \
    --no-headless \
    --render-sleep-s 0.01 \
    --capture-video \
    --capture-dir "$RUN_DIR" \
    --capture-name real001_ep46_white_seed42_30calls \
    --capture-width 960 \
    --capture-height 540 \
    --capture-fps 15 \
    --capture-bitrate-mbps 2 \
    --capture-every-nth-frames 1 \
    --capture-wait-timeout-s 90 \
    --debug-cameras \
    --debug-camera-frame-dir "$RUN_DIR/camera_preflight" \
    --action-trace-jsonl "$RUN_DIR/action_trace.jsonl" \
  2>&1 | tee "$RUN_DIR/rollout.log"
```

预期结果：最接近目标时约 `14.7 mm`，但 gripper 仍为正值并保持打开；目标物高度几乎
不变，不发生抓取和抬起。该历史基线没有 gripper delta 限制，也没有传 fixed-effort 参数；
`--no-dynamic-reset-gripper-effort-limit` 固定保留 authored actuator 的
`effort_limit_sim=10`。

## 7. 运行卡的历史证据

| case | 固定运行卡的直接证据 |
|---|---|
| sim002 | [`action_trace.jsonl`](runs/isolated_checkpoint_eval_20260728/sim002/action_trace.jsonl) |
| sim004 | [`action_trace.jsonl`](runs/isolated_checkpoint_eval_20260728/fidelity_corrections_20260728/sim004_yaw_only/action_trace.jsonl) |
| real003 | [`RUN_SUMMARY_ZH.md`](runs/real003_gripper_guard_ab_20260730/RUN_SUMMARY_ZH.md) / [`seed43 action_trace.jsonl`](runs/real003_gripper_guard_ab_20260730/live_seed43_delta004_effort010/action_trace.jsonl) |
| real007 | [`action_trace.jsonl`](runs/isolated_checkpoint_eval_20260728/fidelity_corrections_20260728/real007_ep67_reset_ab/action_trace.jsonl) |
| real001 | [`action_trace.jsonl`](runs/robot_white_ab_20260729/real001_ep46_white/action_trace.jsonl) |

sim002、sim004、real001、real007 的阶段判断和更完整指标见
[`EVAL_SUMMARY_20260728.md`](runs/isolated_checkpoint_eval_20260728/EVAL_SUMMARY_20260728.md)。

## 8. 每次运行后的统一检查

runner 必须正常退出，且目录中至少存在：

```bash
test -s "$RUN_DIR/bridge.log"
test -s "$RUN_DIR/rollout.log"
test -s "$RUN_DIR/action_trace.jsonl"
```

检查 seed、相机映射、单位和调用数：

```bash
rg -n "inference seed|camera layout|listening" "$RUN_DIR/bridge.log"
sed -n '1p' "$RUN_DIR/action_trace.jsonl"
rg -n "policy call|final|lift|gripper|camera" "$RUN_DIR/rollout.log" | tail -80
```

五张 GUI 卡都应有 MP4：

```bash
find "$RUN_DIR" -maxdepth 1 -type f -name '*.mp4' -print
```

失败时不要立刻换 checkpoint 或修改 USD。先记录：

```text
case / trial 编号
bridge seed / runner seed
checkpoint
action_trace.jsonl
rollout.log
视频（仅 GUI 卡）
实际观察到的 call 区间
```

## 9. 保留的运行与诊断能力

下面能力仍保留在 runner 中，不需要修改 vendor LeIsaac：

| 目的 | 在对应 runner 命令上如何改 |
|---|---|
| 只检查场景/相机，不连接 bridge | 加 `--debug-cameras --debug-cameras-only` |
| 请求 action 但不执行 | 加 `--dry-run --max-policy-calls 1` |
| 只执行一个 policy action | 改为 `--max-policy-calls 1 --action-horizon 1` |
| 保存 reset 时的一次性 policy 图像 | 加 `--debug-cameras --debug-camera-frame-dir "$RUN_DIR/camera_preflight"` |
| 记录指爪/目标接触 | 加 `--contact-probe`，同时保留 `--action-trace-jsonl` |
| 原动作 replay | 启动 `exact_action_replay_bridge.py`，并按技术说明关闭二次 arm/gripper 限幅 |
| base GR00T 对照 | 保留 `--deployment-mode zero-shot-oxe --camera-layout dual`；不属于上面五张 checkpoint 卡 |
| Franka 对照 | 保留 `--robot franka` 的 joint/EEF 路径；单独按 Franka 协议验证 |

GUI 不稳定时，把任一运行卡的 `--no-headless` 和整段
`--capture-video ...` 参数移除，改为：

```bash
    --headless
```

headless 模式不会创建 Isaac viewport，也不会生成 viewport 视频。仍可保存
`action_trace.jsonl`、rollout log 和 contact probe；需要视觉证据时只能在显示链稳定后使用
GUI `--capture-video`，或只用 `--debug-camera-frame-dir` 保存 reset 时的一次性相机检查图。
headless 运行要单独编号，不能覆盖或冒充历史 GUI 基线。

exact-action replay、contact trace 字段和侵入问题的判断顺序见
[代码地图与诊断说明](CODE_MAP_AND_DIAGNOSTICS_ZH.md)。
