# AWS H100 full fine-tune 运行手册

本文件只解释 AWS 边界和故障定位。实际安装、预检和训练全部调用
[`aws_training_pipeline.sh`](../aws_training/aws_training_pipeline.sh)；八份数据配置全部来自
[`aws_tuning_8_manifest.json`](../aws_training/aws_tuning_8_manifest.json)。AWS 专用入口集中在
同级 `aws_training/`；不要复制旧 terminal 日志重建环境。

## 已验证基线

2026-07-15 成功训练使用的关键环境是：

| 项 | 固定值 |
|---|---|
| host | Ubuntu 24.04 x86_64 |
| GPU | 1× H100 80GB |
| Isaac-GR00T | commit `9c7e746b2cd37a810070a98ef41d290a07e806c2` |
| GR00T-N1.7-3B | revision `2fc962b973bccdd5d8ce4f67cc63b264d6886495` |
| Python / uv | 3.12 / 0.11.29 |
| torch / CUDA wheel | 2.9.0+cu128 / 12.8 |
| triton / torchcodec | 3.5.0 / 0.8.0 |
| flash-attn / DeepSpeed | 2.8.3 / 0.17.6 |
| transformers / pyarrow | 4.57.3 / 23.0.1 |

当时宿主 driver 595.71.05、CUDA toolkit 13.2 与 cu128 PyTorch 已经实跑成功。这不是要求
把所有实例的 driver 改成同一个版本：脚本不会安装、降级或替换 driver/CUDA。它验证实际
H100、`nvcc`、PyTorch CUDA、`torch.compile` 和 torchcodec 解码，失败才停止。

不要固定修改 Triton PTX 版本。已验证的 Triton 3.5.0 原生输出可用 PTX；只有 preflight 的
`torch.compile` 真正失败时才单独诊断，不能每次新实例都盲打历史补丁。

## 磁盘布局

默认所有大文件必须真实落在同一块 NVMe：

```text
/opt/dlami/nvme/
├── Isaac-GR00T/
├── smart_project/
│   ├── experiments/...
│   ├── outputs/...prepared datasets...
│   └── outputs/groot_so101_synthetic_finetune/
├── cache/{huggingface,uv,uv-python,xdg,torch,torchinductor,triton}/
└── tmp/
```

实际脚本还把 torch、torch.compile、Triton 和临时目录放到同一 NVMe，避免它们回落到 root
filesystem。

0715 曾因 checkpoint 写入 154GB root filesystem，在保存 optimizer 时出现
`PytorchStreamWriter failed writing file`。所以 preflight 同时检查 realpath、mount point 和
free space，而不是只看路径字符串。

manifest 最多保留 34 个 checkpoint。按每份 45–55GiB 加模型/cache/data，默认要求
2500GiB 可用。若明确采用“每个 run 完成后立刻上传并删旧 checkpoint”的策略，可显式设置
较小的 `AWS_MIN_FREE_GIB`，但这会改变存储风险，不应静默绕过。

## 新实例操作

项目与 prepared data 同步到 NVMe 后：

```bash
cd /opt/dlami/nvme/smart_project/experiments/groot_n17_isaac_smart_task/aws_training
./aws_training_pipeline.sh bootstrap
./aws_training_pipeline.sh auth
./aws_training_pipeline.sh preflight
./aws_training_pipeline.sh stats all --run-tag aws-third-20260804
```

`bootstrap` 只安装最小系统依赖、checkout 固定 commit 并执行：

```text
uv sync --frozen --python 3.12
```

上传 prepared v2.1 后，AWS 不需要 conda、LeRobot、Isaac Sim、IsaacLab、LeIsaac 或机器人
assets。raw v3 只在 AWS 现场重新做 `audit/prepare` 时才需要；正常训练不上传 raw。
`auth` 把 token 写入训练实际使用的 NVMe `HF_HOME`，不能改成裸 `hf auth login`。
`preflight` 下载并验证 manifest 固定的 base-model revision。
`stats all` 必须在固定 GR00T commit 下隔离上传的旧 cache、强制重算，并通过
relative-action span-ratio 检查；stats 生成过程失败时脚本会自动恢复旧 stats，若生成成功但
span-ratio 门禁拒绝，则保留新 stats 供诊断但不会进入 smoke/train。
本机 prepared 的不含 stats 全树 SHA256 已固定在 manifest；AWS 不一致会立即停止。
本次只需同步 `outputs/groot_so101_synthetic_datasets/aws_third_training_20260804/` 这一棵
prepared root，不要把历史 candidate/ablation 目录一并传上去。

## 开训顺序

```bash
./aws_training_pipeline.sh smoke all --run-tag aws-third-20260804
./aws_training_pipeline.sh train all --run-tag aws-third-20260804
```

`smoke` 和 `train` 都会自动重新执行 preflight，不能绕过。`train all` 顺序启动八个独立
run，不会混合数据，也不会并行争抢一张 H100。两者还要求同一 run-tag 下已有完全匹配的
fresh-stats lineage，因此不能跳过上一节的 `stats all`。

每份 stats/smoke/train lineage 都记录 manifest、GR00T、base model、batch/wrapper/modality
代码以及 prepared 全树和 stats 的 SHA256；同名、同 episode 数但内容不同的副本不能静默
冒充本次输入。

不要复用已经非空的 `run-tag/dataset-id` 目录。需要重跑时使用新 run tag；中断恢复前先核对
checkpoint、optimizer 和 trainer state，不要把“允许非空目录”当成 resume。

## 常见阻塞

| preflight / training 现象 | 处理 |
|---|---|
| commit 不一致 | 不要 `git pull main`；重新执行 `bootstrap` checkout 固定 commit |
| prepared contract mismatch | 停止；重新同步 manifest 指向的精确 prepared copy |
| 找不到 pyarrow/GR00T 包 | 必须使用 `.venv/bin/python`，不能把符号链接 resolve 成裸 Python |
| uv 不是 0.11.29 | 重新运行 `bootstrap`；脚本会装到 NVMe 并精确校验 |
| FFmpeg/torchcodec 不能解 AV1 | 使用 Ubuntu 24.04 的 FFmpeg 4–7 包；先通过实际视频 decode |
| `torch.compile` PTX 错误 | 记录完整 traceback、driver/nvcc/torch/triton；不要先打历史补丁 |
| Hugging Face 403/404 | 执行 `./aws_training_pipeline.sh auth` 并确认模型访问权限 |
| NVMe 空间不足 | 扩大 volume，或明确降低保留量并逐 run 上传；不要写 root filesystem |
| relative stats span ratio > 5 | 返回数据清洗；不能加 `--allow-relative-stats-outliers` 草率开训 |
| run directory not empty | 核对 lineage，换新 run tag；不要覆盖或混入另一个数据的 checkpoint |

环境通过只能证明 runtime 和数据合同可运行。训练 loss 很低仍不代表闭环策略成功；最终判断
必须按域进行：real checkpoint 在真机验证，sim checkpoint 在固定 Isaac/LeIsaac 协议验证。
