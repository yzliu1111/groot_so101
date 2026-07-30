# GR00T bridge + LeIsaac SmartTask

这个目录实现 GR00T checkpoint 到 LeIsaac SmartTask 的两进程部署。README 只说明边界和文档
入口；公司现场不要从 README 临时拼命令。

## 从哪里开始

| 目的 | 文档 |
|---|---|
| 公司快速复现 sim002、sim004、real003、real007、real001 | [公司检验操作手册](COMPANY_EVAL_RUNBOOK_ZH.md) |
| 理解代码组成、侵入诊断、exact replay 和图像证据 | [代码地图与诊断说明](CODE_MAP_AND_DIAGNOSTICS_ZH.md) |
| 核对相机、单位、reset、action 和 scene 的稳定语义 | [技术契约](../TECHNICAL_CONTRACTS_ZH.md) |
| 查看五个 case 的阶段汇总 | [`EVAL_SUMMARY_20260728.md`](runs/isolated_checkpoint_eval_20260728/EVAL_SUMMARY_20260728.md) |
| 查看 real003 最新 live / gripper 证据 | [`RUN_SUMMARY_ZH.md`](runs/real003_live_inference_20260730/RUN_SUMMARY_ZH.md) / [`gripper RUN_SUMMARY_ZH.md`](runs/real003_gripper_guard_ab_20260730/RUN_SUMMARY_ZH.md) |

## 当前验证原则

```text
真机采集数据 -> 真机检验
仿真采集数据 -> 同协议仿真检验
```

真机数据 checkpoint 在 Isaac 中仍可用于部署链路和接触物理诊断，但交叉域结果不替代真机
模型质量结论。当前 real003 指尖侧面侵入问题继续保留诊断能力，优先级低于同域正式检验。

## 运行结构

```text
终端 1：GR00T Python 3.12
groot_bridge_server.py
        |
        | 127.0.0.1 pickle socket
        v
终端 2：LeIsaac / Isaac
run_smart_task_closed_loop.py
```

两边 Python 环境不能混用：

- bridge：`$GROOT_ROOT/.venv/bin/python`
- runner：conda `$LEISAAC_ENV`

bridge 与 runner 的 seed 是两个独立 CLI 参数。生产代码默认都是 `None`，公司手册在每条
启动命令中直接写 `--seed 42` 或 real003 bridge 的 `--seed 43`，不通过 shell seed 变量
间接传递。

## 当前 case 状态

| case | 状态 | 固定结果 |
|---|---|---|
| sim002 | 已完成 | pick、lift、hold 成功 |
| sim004 | 已完成 | 横放场景 pick、lift、hold 成功 |
| real003 | 正在推进 | 已出现一次 pick/hold；继续诊断接触侵入 |
| real007 | 阶段结论明确 | pick/lift 成功，place 失败 |
| real001 | 阶段结论明确 | 接近目标，夹爪不闭合 |

每个 case 的 checkpoint、reset、scene、相机 slot、材质、控制参数、直接 CLI seed 和证据路径
都固定在公司操作手册中。

## 保留的能力

当前实现继续保留：

- wrist-only / dual / triple checkpoint layout；
- sim motor units 与 real degrees；
- dataset-state reset 和 settle；
- process-local SO101 eval camera 注入；
- camera-only、dry-run、one-action 和 full chunk；
- GUI viewport MP4 与 reset 时的一次性 camera preflight PNG；
- arm/gripper 单步 envelope 和可选 gripper effort；
- action JSONL trace、contact probe 和 exact-action replay；
- SO101 finetuned、zero-shot OXE 对照和 Franka 对照路径。

这些能力的入口和约束见代码地图；不要为了简化公司运行卡而删除可工作的实现。

## 修改边界

实验实现保持在本目录，不修改 vendor `leisaac/`。代码改动后先运行纯 Python tests，再按：

```text
camera-only -> dry-run -> one action -> full trial
```

完整测试命令见代码地图。
