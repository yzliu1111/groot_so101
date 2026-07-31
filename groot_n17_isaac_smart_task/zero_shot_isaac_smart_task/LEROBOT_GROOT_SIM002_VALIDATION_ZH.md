# LeRobot GR00T N1.7 sim002 部署验证

## 结论

2026-07-31 在本机 RTX 5060 Ti 16GB 上完成：

- LeRobot `0d0737ab57f27c05d7b35fcf27e701f6003a5f3a`
- sim002 `checkpoint-10000`
- LeRobot GR00T N1.7 推理 bridge
- 原 LeIsaac sim002 runner、场景、相机、单位转换和安全限幅
- seed 42、20 个 policy calls、每次完整执行 16-step chunk

结果为成功抓取、抬升并保持到第 20 call。新 trace 最终
`lego_z_minus_base=0.09594 m`，历史 NVIDIA GR00T bridge trace 为
`0.11965 m`。两者都完成稳定抓取，但轨迹不是数值完全相同的 replay。

本次证据：

- `runs/lerobot_sim002_20260731_seed42_trial07_bf16_20calls/rollout.log`
- `runs/lerobot_sim002_20260731_seed42_trial07_bf16_20calls/action_trace.jsonl`

## 为什么使用 BF16 参数

LeRobot 默认的 FP32 参数副本单独推理时峰值约 14GB；与 Isaac 同卡运行会让
Isaac RTX renderer 报 `ERROR_OUT_OF_DEVICE_MEMORY`。加载后把推理副本转换为
BF16，常驻显存约 6.2GB，模型和 Isaac 可以同时运行。

这不修改 checkpoint。训练配置仍然保留 FP32 参数、BF16 compute；这里只是
16GB 单卡部署模式：

```text
--device cuda --parameter-dtype bf16
```

首次 action chunk 推理约 `0.459 s`，warmup 后约 `0.1–0.2 s`。它不是 30Hz
逐步推理；系统每次推理产生 16-step chunk，再按 30Hz 动作频率执行。

## Bridge

使用：

```bash
export HF_HOME=/home/yzliu/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

/home/yzliu/miniforge3/envs/lerobot/bin/python \
  experiments/groot_n17_isaac_smart_task/zero_shot_isaac_smart_task/lerobot_groot_bridge_server.py \
  --model-path "$CHECKPOINT" \
  --device cuda \
  --parameter-dtype bf16 \
  --seed 42 \
  --host 127.0.0.1 \
  --port 5587 \
  --offline
```

仿真 runner 使用 `COMPANY_EVAL_RUNBOOK_ZH.md` 中 sim002 的原参数，只需：

- bridge port 改为 `5587`；
- headless 时按第 9 节移除 viewport capture；
- 可显式加 `--timeout-s 120`。

## 已处理的兼容边界

1. 输入把旧 bridge 的 nested NEW_EMBODIMENT schema 转为 LeRobot 的
   `observation.images.wrist`、`observation.state` 和 `task`。
2. 相对 arm action 必须先对完整 chunk 做 postprocess，再交给 runner；不能对
   raw relative chunk 逐步缓存或调用 `select_action()`。
3. LeRobot 环境使用 NumPy 2，LeIsaac 环境使用 NumPy 1。跨进程 action 以普通
   list 传输，避免 pickle ndarray 时接收端缺少 `numpy._core`。
4. `backbone.model.lm_head.weight` 的 load report 不表示缺少 backbone；LeRobot
   加载后会把这个未独立保存的 tied weight 绑定到 language embedding。

