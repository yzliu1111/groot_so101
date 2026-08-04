# SO101 低显存训练：freeze / projector / LoRA

> 历史 / 小显存 fallback：本目录不是 2026-08-04 H100 八份 full fine-tune 的活动入口，
> 也没有继承当前 manifest 的逐数据集相机、清洗和输出防碰撞合同。今晚只能使用
> `aws_training/aws_training_pipeline.sh`。需要重新启用本路线时，先把同等级安全门
> 移植过来再训练。

full fine-tune OOM 时使用本入口。推荐顺序是：

```text
projector-only 1-step
-> diffusion-lora 1-step
-> 选能稳定运行且最容易部署的方案
-> 再增加 steps
```

不要一开始跑组合策略，也不要从 batch size 32 起步。

## 1. 三种策略怎么选

| strategy | 训练内容 | 输出形式 | 建议 |
|---|---|---|---|
| `projector-only` | action head 的 projector/encoder/decoder | 普通 GR00T checkpoint | 首选，部署最简单 |
| `diffusion-lora` | diffusion Transformer 的 LoRA adapter | PEFT adapter | projector-only 不够或需要更少可训练参数时尝试 |
| `projector-plus-diffusion-lora` | projector + diffusion LoRA | adapter + modules_to_save | 前两者通过后再试 |

LLM、视觉骨干和 VLLN 默认冻结。`diffusion-lora` 不是“训练保存成功就能直接部署”：bridge
需要支持 base model + adapter，或者先把 adapter merge 成可读取模型目录。

## 2. 新训练终端恢复状态

目标机：

```bash
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target
```

本机：

```bash
source /home/yzliu/physical_ai/company_project/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh local
```

上面二选一。训练终端不要带 Isaac 环境变量：

```bash
conda deactivate 2>/dev/null || true
unset PYTHONPATH PYTHONHOME PYTHONNOUSERSITE PYTHONDONTWRITEBYTECODE ISAAC_PATH ISAACLAB_PATH
cd "$SMART_PROJECT"
```

验证 Python、PyTorch、PEFT 和 GPU：

```bash
"$GROOT_ROOT/.venv/bin/python" - <<'PY'
import sys
import peft
import torch

print("python:", sys.version)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("peft:", peft.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
assert sys.version_info[:2] == (3, 12)
PY
```

真实训练若需要 DeepSpeed/CUDA extension，`CUDA_HOME/bin/nvcc` 必须存在。优先让 toolkit
匹配 `torch.version.cuda`；不要只根据 `nvidia-smi` 显示的 CUDA 上限选择 toolkit。

## 3. 先 dry-run

这条命令不启动训练，只检查数据、modality config 和 low-memory patch：

如果从这里新开终端，先执行：

```bash
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target
# source /home/yzliu/physical_ai/company_project/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh local
conda deactivate 2>/dev/null || true
unset PYTHONPATH PYTHONHOME PYTHONNOUSERSITE PYTHONDONTWRITEBYTECODE ISAAC_PATH ISAACLAB_PATH
cd "$SMART_PROJECT"
```

```bash
"$GROOT_ROOT/.venv/bin/python" \
  experiments/groot_n17_isaac_smart_task/lowmem_lora_freeze_so101/train_so101_synthetic_groot_lowmem.py \
    --strategy projector-only \
    --skip-stats \
    --dry-run \
    --max-steps 1 \
    --save-steps 1 \
    --global-batch-size 1 \
    --gradient-accumulation-steps 1 \
    --dataloader-num-workers 0
```

检查输出里的 datasets、camera layout、modality config、strategy 和 output directory。

## 4. Projector-only

### 4.1 1-step smoke

目标机没有 stats 时去掉 `--skip-stats`。

新训练终端先执行：

```bash
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target
# source /home/yzliu/physical_ai/company_project/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh local
conda deactivate 2>/dev/null || true
unset PYTHONPATH PYTHONHOME PYTHONNOUSERSITE PYTHONDONTWRITEBYTECODE ISAAC_PATH ISAACLAB_PATH
cd "$SMART_PROJECT"
```

```bash
"$GROOT_ROOT/.venv/bin/python" \
  experiments/groot_n17_isaac_smart_task/lowmem_lora_freeze_so101/train_so101_synthetic_groot_lowmem.py \
    --strategy projector-only \
    --skip-stats \
    --max-steps 1 \
    --save-steps 1 \
    --global-batch-size 1 \
    --gradient-accumulation-steps 1 \
    --dataloader-num-workers 0 \
    --experiment-name so101_projector_only_smoke
```

### 4.2 较长训练起点

只在 1-step 完成并写出 checkpoint 后运行：

```bash
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target
# source /home/yzliu/physical_ai/company_project/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh local
conda deactivate 2>/dev/null || true
unset PYTHONPATH PYTHONHOME PYTHONNOUSERSITE PYTHONDONTWRITEBYTECODE ISAAC_PATH ISAACLAB_PATH
cd "$SMART_PROJECT"
```

```bash
"$GROOT_ROOT/.venv/bin/python" \
  experiments/groot_n17_isaac_smart_task/lowmem_lora_freeze_so101/train_so101_synthetic_groot_lowmem.py \
    --strategy projector-only \
    --skip-stats \
    --max-steps 2000 \
    --save-steps 500 \
    --global-batch-size 1 \
    --gradient-accumulation-steps 16 \
    --dataloader-num-workers 2 \
    --experiment-name so101_projector_only_bs1_acc16
```

projector-only 输出是普通 checkpoint，可直接交给 `groot_bridge_server.py --model-path`。

## 5. Diffusion LoRA

先用 rank 4 做 1-step：

```bash
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target
# source /home/yzliu/physical_ai/company_project/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh local
conda deactivate 2>/dev/null || true
unset PYTHONPATH PYTHONHOME PYTHONNOUSERSITE PYTHONDONTWRITEBYTECODE ISAAC_PATH ISAACLAB_PATH
cd "$SMART_PROJECT"
```

```bash
"$GROOT_ROOT/.venv/bin/python" \
  experiments/groot_n17_isaac_smart_task/lowmem_lora_freeze_so101/train_so101_synthetic_groot_lowmem.py \
    --strategy diffusion-lora \
    --skip-stats \
    --max-steps 1 \
    --save-steps 1 \
    --global-batch-size 1 \
    --gradient-accumulation-steps 1 \
    --dataloader-num-workers 0 \
    --lora-rank 4 \
    --lora-alpha 8 \
    --lora-dropout 0.05 \
    --experiment-name so101_diffusion_lora_r4_smoke
```

通过后再试 rank 8 或增加 steps：

```bash
--lora-rank 8 \
--lora-alpha 16 \
--max-steps 2000 \
--save-steps 500 \
--gradient-accumulation-steps 16
```

部署前必须单独完成 adapter load/merge 验证；不要把 adapter 目录当普通 GR00T checkpoint。

## 6. Projector + Diffusion LoRA

仅在 projector-only 和 diffusion-lora 都已通过时尝试：

```bash
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target
# source /home/yzliu/physical_ai/company_project/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh local
conda deactivate 2>/dev/null || true
unset PYTHONPATH PYTHONHOME PYTHONNOUSERSITE PYTHONDONTWRITEBYTECODE ISAAC_PATH ISAACLAB_PATH
cd "$SMART_PROJECT"
```

```bash
"$GROOT_ROOT/.venv/bin/python" \
  experiments/groot_n17_isaac_smart_task/lowmem_lora_freeze_so101/train_so101_synthetic_groot_lowmem.py \
    --strategy projector-plus-diffusion-lora \
    --skip-stats \
    --max-steps 2000 \
    --save-steps 500 \
    --global-batch-size 1 \
    --gradient-accumulation-steps 16 \
    --dataloader-num-workers 2 \
    --lora-rank 8 \
    --lora-alpha 16 \
    --experiment-name so101_projector_plus_lora_r8_bs1_acc16
```

## 7. 相机布局和数据

默认使用 dual prepared 数据。其他布局显式加：

```text
--camera-layout wrist-only
--camera-layout triple
```

prepared 数据和 config 来自
[full_finetune_so101/README_ZH.md](../full_finetune_so101/README_ZH.md)。训练、bridge、runner
必须使用同一个 layout。
对应的代码约定见 [TECHNICAL_CONTRACTS_ZH.md](../TECHNICAL_CONTRACTS_ZH.md)。

## 8. 成功标志和 OOM 处理

1-step 至少要看到：

```text
Loaded modality config: ...
Low-memory strategy trainable parameters: ...
Creating custom train dataloader
完成 1/1 step
写出 checkpoint 或 adapter
```

OOM 时按顺序处理：

```text
1. 保持 global batch size=1
2. dataloader workers=0
3. LoRA rank 8 -> 4
4. 组合策略 -> 单一策略
5. 仍然 OOM：换更大显存机器
```

记录策略、trainable parameters、OOM 位于 forward/backward/optimizer/save 的哪一段，以及
`nvidia-smi` 峰值。16GB 本机 OOM 只说明机器不适合该训练，不代表策略无效。

## 9. 部署

projector-only checkpoint 的 bridge、camera dry-run 和 one-step 命令统一放在
[zero_shot_isaac_smart_task/README_ZH.md](../zero_shot_isaac_smart_task/README_ZH.md)。

当前项目状态：

| 目标 | 状态 |
|---|---|
| LeIsaac 仿真部署普通 checkpoint | 已有 runner |
| LeIsaac 仿真部署 LoRA adapter | 需要 adapter load/merge 验证 |
| 真实 SO101 | 尚无 hardware runner |

## 10. 可选日志

需要 WandB 时加：

```bash
--use-wandb --wandb-project __FILL_PROJECT_NAME__
```

不使用 WandB 时不要加。训练产物统一写到
`$SMART_PROJECT/outputs/groot_so101_synthetic_finetune/`。
