# Low-memory / LoRA：SO101 冻结参数与 adapter 微调

本文记录两台机器共用的 GR00T N1.7 低显存训练入口：

- 本机 RTX 5060 Ti 16GB：主要做数据 / 配置 / loader / stats / 1-step smoke test。
- 公司 remote RTX 5090 32GB：主要做低显存策略的正式训练 smoke 和较长 steps 实验。

范围只覆盖已经 prepared 的 SO101 v2.1 数据微调，不覆盖 LeRobot v3 转换和真机部署。所有命令尽量通过 `SMART_PROJECT` 和 `GROOT_ROOT` 两个变量写成同一套；先按机器设置 profile，再复制后面的训练命令。

## 当前结论

- 远程 5090 PC 上虽然 `nvidia-smi` / 官方建议显示 CUDA 13.0，但目前能跑到 OOM，没有出现明确 CUDA 13.0 版本相关报错。
- 当前主矛盾不是 CUDA 版本，而是 GR00T N1.7 默认 fine-tune 训练参数量太大。
- 全量 / 默认 action-head 微调在 32GB 5090 上仍然 OOM，需要先尝试冻结更多权重，或用 LoRA / adapter 路线。
- 本机 5060 Ti 16GB 不应该作为“正式训练是否可行”的主要判断依据；它适合快速验证脚本、数据、stats 和最小训练 step，OOM 是预期风险。
- CUDA 12.8 仍是 PyTorch `cu128` 栈的稳妥保底方案；但如果 CUDA 13.0 toolkit 已经能跑到 OOM，先不急着切 sudo 账号重装 12.8。

## 先把这件事讲清楚

这条链路的目标不是“让脚本能跑起来”本身，而是让 GR00T N1.7 学会在 SO101 embodiment 下，根据两路相机、当前关节状态和语言指令，输出能完成任务的 SO101 action。当前任务可以先粗略理解为：

```text
看图像 + 看当前 SO101 状态 + 读指令
-> 预测接下来一段 SO101 关节 / 夹爪动作
-> 在仿真或真机上把小块捡起来
```

训练脚本能跑，只说明工程链路通了；模型有没有学到，还要继续看 loss、open-loop action、仿真闭环和最后的真机表现。

### 1. 冻结权重微调到底行不行

工程上行，训练上不保证一定有效。

“冻结权重”不是不训练，而是只训练一小部分参数。GR00T N1.7 很大，里面大致可以分成：

| 模块 | 可以怎么理解 | 默认是否容易很吃显存 |
|---|---|---:|
| 语言 / 视觉 backbone | 负责理解图像和语言，相当于“看懂场景” | 是 |
| action head projector / encoder / decoder | 把 SO101 状态和动作空间接到 GR00T 内部表示上 | 中等 |
| diffusion Transformer action model | 负责生成一段动作轨迹 | 是 |
| VLLN / vision-language bridge 小模块 | 对视觉语言特征再整理 | 较小 |

冻结策略的核心假设是：GR00T 原模型已经有足够的视觉语言和机器人先验，我们只需要让它适配 SO101 的状态 / 动作维度、相机视角和当前任务数据。这个假设在“任务简单、数据干净、目标动作空间差异不大”的情况下可能成立；如果 SO101 合成数据和模型原先见过的数据差异很大，只训很小一部分可能学不够。

所以它不是魔法，而是第一条低显存基线：先证明少量参数训练能不能让 loss 下降、动作更像数据，再决定是否需要训练更多模块。

### 2. LoRA 是不是可以实现的方案

可以实现，而且 Isaac-GR00T 环境里已经有 `peft==0.17.1` 这个依赖。但这里要分清两句话：

```text
LoRA 能被工程接上    -> 是，脚本已经提供 experimental 路线
LoRA 训练出来一定有效 -> 不保证，需要实验验证
```

LoRA 的思路是冻结原始大权重，只在某些 Linear 层旁边加很小的可训练 adapter。训练时只更新 adapter，显存和 optimizer state 都会小很多。这里的 `diffusion-lora` 只打到 `action_head.model` 里的 diffusion Transformer，不碰 Qwen/Cosmos 语言视觉骨干。

LoRA 的一个额外现实问题是：checkpoint 往往是 “base model + adapter”，不是一个普通完整模型目录。训练能保存，不等于现有推理脚本立刻知道怎么加载 adapter；这一步后面需要单独验证。

### 3. Isaac-GR00T 官方训练 API 有没有现成冻结 / LoRA

官方 fine-tune API 有一部分现成冻结开关，但没有看到现成 LoRA 训练入口。

本地 `Isaac-GR00T` 里可以看到：

```text
gr00t/configs/finetune_config.py
  tune_llm
  tune_visual
  tune_projector
  tune_diffusion_model

gr00t/model/gr00t_n1d7/gr00t_n1d7.py
  tune_projector
  tune_diffusion_model
  tune_vlln
```

也就是说，官方已经支持“哪些大模块参与训练”的冻结思路。官方 `launch_finetune.py` 暴露了 `tune_llm / tune_visual / tune_projector / tune_diffusion_model`，但没有暴露 `tune_vlln`，也没有看到官方 CLI 直接提供 `--use-lora` 这种入口。

源码核对结论：

| 问题 | 结论 |
|---|---|
| `tune_llm` / `tune_visual` / `tune_projector` / `tune_diffusion_model` 是否是官方 fine-tune 参数 | 是。它们定义在 `gr00t/configs/finetune_config.py`，并由 `launch_finetune.py` 写入 `config.model`。 |
| `tune_vlln` 是否存在 | 是。它定义在 `gr00t/configs/model/gr00t_n1d7.py`，默认 `True`，并在 `Gr00tN1d7ActionHead.set_trainable_parameters()` 中真正控制 `vlln` / `vl_self_attention`。 |
| `tune_vlln` 是否在官方 `launch_finetune.py` CLI 里暴露 | 没有。快捷 fine-tune CLI 不提供这个参数，所以默认会继续训练 VLLN。 |
| 是否可以通过官方更底层 `launch_train.py` / 完整 `Config` 设 `tune_vlln` | 理论上可以，因为 `launch_train.py` 接受完整 `Config`；但这不是官方 new-embodiment fine-tune 文档推荐的快捷入口。 |
| 官方是否封装了 LoRA fine-tune 参数 | 没看到。全仓搜索 `get_peft_model` / `LoraConfig` / `PeftModel` / `lora`，GR00T 训练代码没有现成 LoRA 训练封装。 |
| 为什么项目里有 `peft` | `peft==0.17.1` 是官方依赖，但依赖存在不等于已经封装了 GR00T LoRA 训练入口。 |

所以当前脚本做了两件事：

- 复用官方已有的冻结能力，并额外暴露 `tune_vlln`。
- 在官方训练 pipeline 外围加一个 LoRA experimental patch，不改 Isaac-GR00T 官方源码。

这里要特别注意：`tune_vlln` 没暴露，不等于官方没有这块冻结逻辑；它是“模型里有，快捷 fine-tune 参数里没有”。LoRA 则是“依赖里有，但官方 GR00T fine-tune 代码里没封装训练入口”。

### 3.1 SO101 embodiment 差异为什么重要

SO101 和 GR00T 预训练里常见的 DROID/Franka-like 机械臂不是同一个 embodiment。直观差异包括：

- 关节数量和结构不同；
- action 语义不同；
- 相机安装视角不同；
- 夹爪和末端执行器行为不同；
- 数据来自 LeIsaac 合成环境，而不是原始 DROID 分布。

所以更严谨的判断是：

```text
任务语义可能简单：看见小块 -> 抓起来
embodiment 迁移不一定简单：7DoF / Franka-like 经验 -> SO101 关节动作
```

这也是为什么 `projector-only` 只是第一条低显存基线。它主要让 SO101 的状态/action 接入已有 GR00T 表示；如果 embodiment 差异导致 action 生成本身不适配，可能就需要训练 diffusion action model，甚至需要更完整的微调。

### 3.2 建议的实验优先级

不要把 5060 Ti 或 5090 变成无限调参战场。更合理的顺序是：

```text
1. 官方冻结能力能覆盖的策略：projector-only / 关闭 diffusion / 关闭 vlln
2. LoRA experimental：只给 action_head.model 的 diffusion Transformer 加 adapter
3. 如果 loss / open-loop / 仿真闭环都不理想，申请云算力做更完整微调
```

也就是说，冻结和 LoRA 是为了低成本判断“这条路线有没有希望”。如果低显存策略跑通但效果不好，不要过度沉迷构建更复杂 adapter；应该把结论整理出来，转向公司云端 full / larger-scope fine-tune。

### 4. 能跑起来以后，目的是什么

目的分阶段，不是一跳到“真机捡起来”。

最低目标：

```text
1-step 能跑完，不 OOM，不报 schema / modality / CUDA 错。
```

训练目标：

```text
训练 loss 能下降，模型输出的 SO101 action 更接近合成数据里的专家动作。
```

机器人任务目标：

```text
在 SmartTask / SO101 场景里，根据指令把目标小块捡起来。
```

最终真机目标：

```text
把同一类策略迁移到真实 SO101 机械臂和真实相机上。
```

因此，“能跑”只是第 1 关；“有用”至少要过 open-loop action 检查、仿真闭环成功率检查，最后才是真机。

### 5. 从能跑到有效果，建议按这个验收阶梯走

| 阶段 | 看什么 | 说明 |
|---|---|---|
| 工程 smoke | `--max-steps 1` 能完成并保存 checkpoint | 只证明环境、数据、模型能接起来 |
| 训练 smoke | 训练 100-500 steps，loss 不爆炸，最好有下降趋势 | 证明优化过程不是坏的 |
| open-loop 检查 | 给数据集里的观测，模型输出动作是否接近 expert action | 不接机器人，先看动作像不像 |
| 仿真闭环 | 在 LeIsaac / IsaacLab SmartTask 中跑成功率 | 证明动作反馈循环可用 |
| 真实 SO101 | 相机标定、动作缩放、安全限幅、延迟处理 | 这是另一层部署问题 |

当前这份文档主要解决前两关：让本机 5060 Ti 能做最小工程验证，让 5090 32GB 有机会跑过训练 smoke，并把训练策略说清楚。

## 新训练入口

维护中的入口脚本：

```text
experiments/groot_n17_isaac_smart_task/lowmem_lora_freeze_so101/train_so101_synthetic_groot_lowmem.py
```

新命令直接使用这个子目录入口。

它和 full fine-tune 入口分工不同：

| 脚本 | 用途 |
|---|---|
| `full_finetune_so101/train_so101_synthetic_groot.py` | 数据准备 + 官方默认 fine-tune launcher，适合转换 v3 -> v2.1 或复现默认路线 |
| `lowmem_lora_freeze_so101/train_so101_synthetic_groot_lowmem.py` | 只在 GR00T 环境里运行，默认复用 prepared v2.1 数据，提供冻结 / LoRA 低显存策略 |

低显存脚本不会 import LeRobot，也不会修改原始 `dataset/`。

默认数据路径仍然是：

```text
outputs/groot_so101_synthetic_datasets/so101_lego_pick_0609_1722
outputs/groot_so101_synthetic_datasets/so101_lego_pick_0609_1722_mimic
```

## 机器 Profile 与环境检查

后面所有训练命令都假设已经设置了这两个变量：

```text
SMART_PROJECT -> 当前 smart_project 仓库
GROOT_ROOT    -> Isaac-GR00T 仓库，且里面已经有 .venv
```

### Profile A：本机 5060 Ti

本机主要用来确认 prepared 数据、modality config、stats 和最小训练 step 是否连得上。路径按当前机器写：

```bash
export SMART_PROJECT=/home/yzliu/smart_project
export GROOT_ROOT=/home/yzliu/Isaac-GR00T

# 只指向本机实际存在的 CUDA toolkit；12.8 最贴合 PyTorch cu128 栈。
unset CUDA_HOME
if [ -d /usr/local/cuda-12.8 ]; then
  export CUDA_HOME=/usr/local/cuda-12.8
elif [ -d /usr/local/cuda ]; then
  export CUDA_HOME=/usr/local/cuda
fi

if [ -n "${CUDA_HOME:-}" ]; then
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi
```

如果本机没有可用 `$CUDA_HOME/bin/nvcc`，先不要急着装系统 CUDA；可以先跑下面的 Python 环境检查和 `--dry-run`。只有 DeepSpeed / CUDA extension 明确报 `CUDA_HOME does not exist` 或 CUDA op 编译失败时，再补本机 toolkit。

本机建议执行顺序：

```text
1. 环境检查
2. dataset / config dry-run
3. 必要时生成或刷新 stats
4. projector-only 1-step smoke test
5. 如果 16GB OOM，记录即可，不把它当作策略失败结论
```

### Profile B：公司 remote 5090

remote 5090 是低显存训练的主要实验机器。在 `guest1` 账号下：

```bash
export SMART_PROJECT=/home/guest1/smart_project
export GROOT_ROOT=/home/guest1/Isaac-GR00T

# 如果 CUDA 13.0 toolkit 已经存在，先试它。
export CUDA_HOME=/usr/local/cuda-13.0
# 如果后续确认需要 12.8，再切成：
# export CUDA_HOME=/usr/local/cuda-12.8

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

remote 5090 建议执行顺序：

```text
1. 环境检查
2. projector-only 1-step smoke test
3. diffusion-lora 1-step smoke test
4. 能过以后再跑 100-500 steps 训练 smoke
5. 最后才跑 2000 steps 起步实验
```

### 通用环境验证

两台机器都用同一条验证命令：

```bash
cd "$SMART_PROJECT"
"$GROOT_ROOT/.venv/bin/python" - <<'PY'
import torch
import peft
from torch.utils.cpp_extension import CUDA_HOME
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("peft:", peft.__version__)
print("extension CUDA_HOME:", CUDA_HOME)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
else:
    print("gpu:", "<unavailable>")
PY
```

`peft` 是 Isaac-GR00T 官方 `pyproject.toml` 里的依赖，正常 `uv sync --python 3.10 && uv pip install -e .` 后应该已经存在。

### 本机 5060 Ti 快速 dry-run

这条命令不进入训练，只检查 prepared 数据路径、modality config、GR00T config 构造和低显存策略 patch。适合作为本机第一条命令：

```bash
cd "$SMART_PROJECT"
"$GROOT_ROOT/.venv/bin/python" \
  experiments/groot_n17_isaac_smart_task/lowmem_lora_freeze_so101/train_so101_synthetic_groot_lowmem.py \
    --strategy projector-only \
    --skip-stats \
    --max-steps 1 \
    --save-steps 1 \
    --global-batch-size 1 \
    --gradient-accumulation-steps 1 \
    --dataloader-num-workers 0 \
    --experiment-name so101_local_5060ti_dryrun \
    --dry-run
```

如果本机需要生成或刷新 GR00T stats，直接调用 GR00T stats 脚本。注意：低显存训练脚本的 `--dry-run` 会把 stats 命令也 dry-run 掉，所以不要用它来实际生成 stats。

```bash
cd "$GROOT_ROOT"
"$GROOT_ROOT/.venv/bin/python" \
  gr00t/data/stats.py \
    --dataset-path "$SMART_PROJECT/outputs/groot_so101_synthetic_datasets/so101_lego_pick_0609_1722" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "$SMART_PROJECT/experiments/groot_n17_isaac_smart_task/full_finetune_so101/so101_synthetic_groot_config.py"

"$GROOT_ROOT/.venv/bin/python" \
  gr00t/data/stats.py \
    --dataset-path "$SMART_PROJECT/outputs/groot_so101_synthetic_datasets/so101_lego_pick_0609_1722_mimic" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "$SMART_PROJECT/experiments/groot_n17_isaac_smart_task/full_finetune_so101/so101_synthetic_groot_config.py"
```

## 策略 1：Projector-only

这是第一个推荐基线。它冻结 LLM、视觉骨干、diffusion Transformer 和 VLLN，只训练 action head 里的 projector / encoder / decoder 相关模块。

优点：

- 不使用 LoRA，保存出来是普通 GR00T checkpoint。
- 比默认路线少很多可训练参数。
- 对新 embodiment 的状态 / 动作映射最直接。

1-step smoke test。两台机器都可以先跑这条；本机 5060 Ti 建议先只跑到这里：

```bash
cd "$SMART_PROJECT"
"$GROOT_ROOT/.venv/bin/python" \
  experiments/groot_n17_isaac_smart_task/lowmem_lora_freeze_so101/train_so101_synthetic_groot_lowmem.py \
    --strategy projector-only \
    --skip-stats \
    --max-steps 1 \
    --save-steps 1 \
    --global-batch-size 1 \
    --gradient-accumulation-steps 1 \
    --dataloader-num-workers 0
```

如果是在本机留档，可以额外加：

```text
--experiment-name so101_local_5060ti_projector_only_smoke
```

remote 5090 正式起步：

```bash
cd "$SMART_PROJECT"
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

如果 `--skip-stats` 前目标机还没有 stats，可以去掉它，让脚本先跑 GR00T stats。

## 策略 2：Diffusion LoRA

这条路线冻结 projector 和 VLLN，只给 `action_head.model` 里的 diffusion Transformer Linear 层加 LoRA。脚本会让 diffusion 子模块保持 train mode，但由 PEFT 冻结 base 权重，只训练 adapter 参数。

优点：

- 可训练参数最少。
- 更适合测试 “LoRA 能不能让低显存机器跑起来”。

注意：

- 输出是 PEFT adapter 形式，不是普通完整 GR00T checkpoint。
- 后续推理需要按 “base model + adapter” 的方式加载，部署前还要单独验证。

1-step smoke test。两台机器都可以跑；本机 5060 Ti 如果 rank 8 OOM，下一步直接降 rank 4：

```bash
cd "$SMART_PROJECT"
"$GROOT_ROOT/.venv/bin/python" \
  experiments/groot_n17_isaac_smart_task/lowmem_lora_freeze_so101/train_so101_synthetic_groot_lowmem.py \
    --strategy diffusion-lora \
    --skip-stats \
    --max-steps 1 \
    --save-steps 1 \
    --global-batch-size 1 \
    --gradient-accumulation-steps 1 \
    --dataloader-num-workers 0 \
    --lora-rank 8 \
    --lora-alpha 16 \
    --lora-dropout 0.05
```

本机 5060 Ti 更保守的 rank 4 smoke test：

```bash
cd "$SMART_PROJECT"
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
    --experiment-name so101_local_5060ti_diffusion_lora_r4_smoke
```

remote 5090 正式起步：

```bash
cd "$SMART_PROJECT"
"$GROOT_ROOT/.venv/bin/python" \
  experiments/groot_n17_isaac_smart_task/lowmem_lora_freeze_so101/train_so101_synthetic_groot_lowmem.py \
    --strategy diffusion-lora \
    --skip-stats \
    --max-steps 2000 \
    --save-steps 500 \
    --global-batch-size 1 \
    --gradient-accumulation-steps 16 \
    --dataloader-num-workers 2 \
    --lora-rank 8 \
    --lora-alpha 16 \
    --experiment-name so101_diffusion_lora_r8_bs1_acc16
```

如果显存仍然紧张，把 `--lora-rank 8` 降到 `4`。

## 策略 3：Projector + Diffusion LoRA

这条路线同时训练 projector，并给 diffusion Transformer 加 LoRA。diffusion base 权重仍由 PEFT 冻结，只训练 LoRA adapter。它比前两条更有表达力，但显存和保存路径也更复杂。

脚本会把这些 projector 模块放进 PEFT `modules_to_save`：

```text
state_encoder
action_encoder
action_decoder
position_embedding
```

建议只有在前两条能跑通后再试。本机 5060 Ti 不建议一开始跑这条；优先把它当作 remote 5090 或更大显存机器上的第三阶段实验：

```bash
cd "$SMART_PROJECT"
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

如果这条仍然 OOM，优先退回 `projector-only` 或 `diffusion-lora`，不要继续加 batch size。

## 参数说明

关键默认值：

```text
--global-batch-size 1
--gradient-accumulation-steps 16
--gradient-checkpointing
--num-shards-per-epoch 4096
--optim adamw_torch
```

这里的 `--global-batch-size` 在单卡下基本就是 per-device batch size。5060 Ti 和 32GB 5090 都不要再从 32 起步；本文件所有低显存训练都先从 1 起步。

LoRA 默认 target regex：

```text
action_head\.model\..*(to_q|to_k|to_v|to_out\.0|proj_out_1|proj_out_2)$
```

也就是只打到 action head diffusion Transformer 的注意力投影和输出投影，不碰 Qwen/Cosmos 语言视觉骨干。

## 成功标志

看到类似日志说明已经进入训练：

```text
Creating custom train dataloader
Rank 0, Worker 0: Caching shard...
Low-memory strategy trainable parameters: ...
```

真正成功要至少完成 `1/1` step，并写出 checkpoint。

## OOM 后怎么判断

按顺序记录：

```bash
nvidia-smi
du -sh outputs/groot_so101_synthetic_finetune/*
```

并保存完整报错里的：

- 策略名；
- trainable parameter 数量；
- OOM 发生在 forward、backward、optimizer step，还是 save checkpoint；
- 峰值显存。

如果 `projector-only` 和 `diffusion-lora` 都在本机 5060 Ti 上 OOM，只能说明本机不适合承担训练，不代表策略失败。把日志和参数记录下来，转到 remote 5090 继续判断。

如果 `projector-only` 和 `diffusion-lora` 都在 remote 5090 32GB 上 OOM，基本可以判断 32GB 级别单卡不适合作为 GR00T N1.7 微调训练机，需要申请更大显存云端资源，或者进一步做更激进的冻结、量化训练、离线特征缓存。
