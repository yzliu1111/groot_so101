# SmartTask 部署代码地图与诊断说明

这份文件用于自己学习、读代码和继续定位问题，不是公司现场的复制运行手册。

公司检验时直接使用
[SmartTask 公司检验操作手册](COMPANY_EVAL_RUNBOOK_ZH.md)；跨文件稳定语义见
[技术契约](../TECHNICAL_CONTRACTS_ZH.md)。

## 1. 当前原则

正式模型质量检验按数据来源闭环：

```text
真机数据 checkpoint -> 真机检验
仿真数据 checkpoint -> 同协议仿真检验
```

真机数据 checkpoint 在 Isaac 中仍有价值，但用途是检查部署链路、相机 slot、单位、reset、
action envelope 和接触物理，不把交叉域成功率当成真机模型成功率。

当前五个 case 的位置是：

| case | 作用 |
|---|---|
| sim002 | 已完成的 wrist-only 仿真正例 |
| sim004 | 已完成的 triple、横放仿真正例 |
| real003 | 继续做 Isaac 部署和接触诊断，优先级低于同域正式检验 |
| real007 | 已定位为能 pick/lift、不能完成 place 的阶段性失败 |
| real001 | 已定位为接近目标但不启动闭爪的阶段性失败 |

## 2. 两进程数据流

```text
Isaac policy camera + SO101 joint state
                 |
                 v
run_smart_task_closed_loop.py
  - 构造 observation
  - 通过 wire.py 请求 action
                 |
                 v
groot_bridge_server.py  (独立 GR00T Python 3.12 进程)
  - 加载 checkpoint
  - 校验 modality
  - inference / decode_action
                 |
                 v
run_smart_task_closed_loop.py
  - dataset units -> Isaac radians
  - arm / gripper envelope
  - Isaac env.step()
  - trace / image / contact evidence
```

分成两个进程不是临时绕法：GR00T 与 Isaac/LeIsaac 属于不同 Python 环境。bridge 默认只监听
`127.0.0.1`，协议是本地可信环境使用的 pickle socket，不应直接暴露到不可信网络。

## 3. 文件构成

| 文件 | 责任 | 修改时最先检查什么 |
|---|---|---|
| `groot_bridge_server.py` | 加载 GR00T、modality 校验、推理和 action decode | layout、checkpoint、bridge `--seed` |
| `run_smart_task_closed_loop.py` | Isaac 启动、observation、action 执行、证据记录 | units、reset、相机映射、runner `--seed` |
| `wire.py` | bridge/runner 消息帧和请求协议 | 消息大小、schema、timeout |
| `action_chunk.py` | 校验和切分 `(B,T,D)` action chunk | 所有 action key 的时间维一致 |
| `action_timing.py` | policy Hz 到 physics steps 的整数映射 | 当前 30 Hz policy / 60 Hz physics |
| `so101_joint_units.py` | motor/degrees 与 Isaac radians 的转换和限幅 | arm 单位与 gripper `[0,100]` 不混写 |
| `scene_profiles.py` | tray/LEGO 布局、材质和 target 绑定 | case 使用的 profile 是否完全一致 |
| `so101_eval_cameras.py` | 进程内注入 top/left camera | 不修改 vendor LeIsaac |
| `contact_probe.py` | 指爪与目标物的接触证据 | sensor pair、force、separation、tip point |
| `exact_action_replay_bridge.py` | 从 trace 重放已执行的绝对 joint targets | replay 时关闭二次单步限幅 |
| `tests/` | 不启动 Isaac 的纯 Python 回归 | 每次代码修改后全部通过 |

训练侧 modality 定义位于：

```text
../full_finetune_so101/so101_synthetic_groot_config.py
../full_finetune_so101/so101_synthetic_groot_wrist_only_config.py
../full_finetune_so101/so101_synthetic_groot_triple_config.py
```

## 4. seed 不是代码常量

生产代码没有把 `42` 或 `43` 写死：

- bridge 的 `--seed` 默认是 `None`，在模型加载前设置 Python、NumPy 和 torch；
- runner 的 `--seed` 默认是 `None`，传给 Isaac env 并写入 action trace；
- 两个进程的随机状态相互独立，必须分别在启动命令中显式指定。

公司手册因此直接写：

```text
groot_bridge_server.py ... --seed 42
run_smart_task_closed_loop.py ... --seed 42
```

real003 当前已验证 trial 使用 bridge `43`、runner `42`。相同 seed 只是尽力复现，不代表
CUDA diffusion 或闭环物理逐位确定。

## 5. 相机：物理位置和模型 slot 是两层

进程内注入 camera 的物理角色固定为：

```text
camera3 = top/front
camera2 = left
camera1 = wrist
```

模型 slot 由 runner 参数决定，不能只看编号猜语义：

| case | model slot mapping |
|---|---|
| sim002 / real001 | `wrist <- camera1` |
| sim004 / real007 | `top <- camera3, left <- camera2, wrist <- camera1` |
| real003 | `top <- camera2, left <- camera1, wrist <- camera3` |

real003 的排列来自 prepared checkpoint 的历史 slot 语义；它不是通用 triple 排列，不能套到
sim004 或 real007。

## 6. action 落地顺序

SO101 微调路线中，bridge 返回的是 checkpoint 数据集坐标中的绝对 target。runner 按下面
顺序处理：

```text
检查 shape / NaN / infinity
-> 按 checkpoint 数据集单位 clip
-> motor units 或 degrees 转 Isaac radians
-> arm_target_scale
-> max_arm_step_rad
-> max_gripper_step_rad
-> Isaac runtime soft limits
-> env.step()
```

`arm_target_scale` 和两个 max-step 都是部署侧 actuator envelope，不改变 policy 输出语义。
`--so101-gripper-effort-limit-sim` 固定 PhysX effort 时会关闭 LeIsaac 的动态 effort reset。

这两天新增的 gripper `delta rad` / effort 配置只用于 real003 当前基线。sim002、sim004、
real001、real007 的公司运行卡保留各自历史配置，没有统一套用 real003 参数。

## 7. reset、settle 和第一帧

`--so101-initial-dataset-state` 记录的是 checkpoint 数据坐标。runner 在创建 env 前转换成
Isaac radians，并关闭 `JointPositionAction` 的 default offset，避免：

```text
absolute target + reset pose
```

reset 后必须重新 render，再从同一姿态读取 proprioception 和 camera。不能在 reset 后只调用
`set_joint_position_target()` 冒充完整状态重建。

`--settle-env-steps` 用于让动态目标物落到支撑面。real003 当前卡使用 30 steps，并在 settle
后恢复 frame-0 机器人关节状态；其余四张历史卡固定为 0，避免改变原轨迹。

## 8. USD、LeIsaac cfg 还是动作输出：如何区分

“任何一个指尖接触 LEGO 远离 robot 的侧面后出现侵入”目前不能只凭画面断言是 USD 或
LeIsaac cfg。按层做单变量证据：

```text
同一模型 live inference
        |
        +-- 保存 applied_commands_rad
        |
        v
exact-action replay（动作完全相同）
        |
        +-- 同一 USD / 不同 contact 或 actuator 配置
        +-- 不同 USD / 同一执行配置
        |
        v
比较 tip、contact separation、force、LEGO pose 和 joint tracking
```

判断方向：

- 同一 exact action 在不同 USD 下才发生异常，优先检查指尖/LEGO collider、scale、collision
  approximation、contact offset 和资产组合；
- 同一 USD 在不同 actuator/contact cfg 下才发生异常，优先检查 effort、joint tracking、
  solver/contact 配置和 LeIsaac runtime patch；
- replay 不异常、live inference 才异常，优先检查闭环图像分布偏移和后续 action；
- contact probe 没有负 separation，但画面像侵入，先排除相机遮挡、材质和渲染误判。

此前增加但没有改变结果的 solver 实验项已从生产 runner 路径移除。当前保留的是能够提供
区分证据的 contact probe 和 exact replay，不把“加 solver 参数”当成答案。

## 9. exact-action replay

replay bridge 不加载 GR00T，只读取 source trace 中每个 `policy_call` 的
`applied_commands_rad`：

```bash
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LEISAAC_ENV"
cd "$SMART_PROJECT/experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task"

"$CONDA_PREFIX/bin/python" exact_action_replay_bridge.py \
  --action-trace-jsonl /absolute/path/to/source/action_trace.jsonl \
  --camera-layout triple \
  --host 127.0.0.1 \
  --port 5577 \
  --exit-after-replay
```

runner 必须使用与 source trace 相同的 scene、reset、units、相机映射、settle 和材质，并设置：

```text
--so101-arm-target-scale 1
--so101-max-arm-step-rad 0
--so101-max-gripper-step-rad 0
```

否则 runner 会对 source target 再限幅，结果不再是 exact replay。

## 10. 图像证据和远程桌面

当前保留两种图像检查：

| 模式 | 优点 | 注意 |
|---|---|---|
| `--capture-video` | 接近肉眼观察的 viewport MP4 | 要求 `--no-headless`，受 XFCE/GPU 显示链影响 |
| `--debug-camera-frame-dir` | 保存 reset 时模型实际看到的一次性 PNG | 它不是视频，也不连续记录 rollout |

IOMMU 警告说明 bare-metal GPU peer-to-peer/display 可能不稳定，但警告本身不证明本次
policy tensor 已损坏。排查时保留终端日志和一次性 camera preflight，并且不要在同一 GPU
会话同时启动多个 Isaac 实例。纯 headless rollout 不创建 viewport，也不再提供连续 PNG。

real003 已验证 success trial 的 `camera1` 连续 PNG 来自远程排障期间的临时 recorder。
该 recorder 现已从生产 runner 删除；历史 artifact 保留，但不再作为运行能力或推荐配置。
当前公司运行卡与其他 case 一样使用 `--no-headless --capture-video`，固定物理 left camera
生成 viewport MP4；这个录像视角不参与 policy observation。

## 11. 学习推进顺序

这条部署链适合作为当前学习计划中的一个小闭环：

1. 先从公司手册选一张固定运行卡，写出预期结果。
2. 读 trace 第一行，解释 units、camera mapping、seed、reset 和 envelope。
3. 选一个失败区间，手工定位对应 `policy_call`。
4. 提出一个可以被 exact replay 或单变量 A/B 推翻的假设。
5. 先写出要比较的输入、输出和 pass/fail，再让代码执行。
6. 把结果回写为实验记录，不把单次视频印象直接升级成根因。

## 12. 修改后的最低验证

不启动 Isaac 的回归测试：

```bash
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LEISAAC_ENV"
cd "$SMART_PROJECT/experiments"

PYTHONDONTWRITEBYTECODE=1 "$CONDA_PREFIX/bin/python" -m unittest discover \
  -s groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/tests
```

代码静态通过后，运行顺序仍是：

```text
camera-only -> dry-run -> one action -> full canonical trial
```
