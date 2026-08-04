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

版本门禁比较的是版本号字段，不比较整行展示文本：例如 uv 的平台后缀和 Python wheel 的
合法本地构建后缀不会造成假失败；不同 release、rc/dev/post 版本仍会被拒绝。GR00T commit、
base-model revision、Python 主次版本和 PyTorch CUDA 版本继续分别严格校验。

当时宿主 driver 595.71.05、CUDA toolkit 13.2 与 cu128 PyTorch 已经实跑成功。这不是要求
把所有实例的 driver 改成同一个版本：脚本不会安装、降级或替换 driver/CUDA。它验证实际
H100、`nvcc`、PyTorch CUDA、`torch.compile` 和 torchcodec 解码，失败才停止。

这里必须区分三层：DLAMI 的 driver/toolkit、PyTorch wheel 自带的 cu128 runtime，以及
Triton 的 PTX 映射。锁定 commit 的 frozen lock 实际安装 PyTorch 2.9.0 + Triton 3.5.0；
Triton 3.5.0 已原生实现 CUDA major `>=13`，实际 `nvcc` 为 13.2 时必须映射为 PTX 92。
本次 x86 cu128 wheel 默认使用 Triton 随包的 ptxas 12.8（映射 PTX 87），而不是把系统
13.2 ptxas 强塞进 venv；两层会分别打印和校验。
仓库仍保留的 `scripts/patch_triton_cuda13.sh` 注释针对旧 PyTorch 2.7 / Triton 3.3.1，不能
应用到本次 frozen venv。`bootstrap` 和 `preflight` 都会拒绝遗留 `.pth`/源码补丁，核对原生
13+ 分支和实际 PTX 映射；随后 `torch.compile` 再触发一次真实 GPU 编译。

## 今晚固定的 AWS 路径

目标机已经用 `df -h` 确认：`/` 约 193GB，`/opt/dlami/nvme` 约 3.5TB。
今晚不再猜设备、不再另建挂载点，直接使用已经挂载好的 `/opt/dlami/nvme`。
项目代码可以位于目标机任意正常路径；脚本从自身位置自动识别 `SMART_PROJECT`。

唯一的项目软链接是：

```text
$AWS_PROJECT_DIR/outputs
  -> /opt/dlami/nvme/smart_project_outputs
```

在目标机设置实际项目路径并创建链接：

```bash
export AWS_PROJECT_DIR=/home/ubuntu/smart_project   # 按目标机实际 checkout 修改
export AWS_STORAGE_ROOT=/opt/dlami/nvme
export AWS_OUTPUTS_TARGET=/opt/dlami/nvme/smart_project_outputs

cd "$AWS_PROJECT_DIR/experiments/groot_n17_isaac_smart_task/aws_training"
./aws_training_pipeline.sh paths
./aws_training_pipeline.sh storage-link
```

`storage-link` 只负责这一个 `outputs` 链接。若 `$AWS_PROJECT_DIR/outputs` 已是实体目录、
错误链接或悬空链接，它会停止且不移动、不删除、不覆盖现有数据。

八份 prepared v2.1 数据必须上传到这个精确的物理目录：

```text
/opt/dlami/nvme/smart_project_outputs/groot_so101_synthetic_datasets/aws_third_training_20260804/
```

本机到 AWS 的目录映射是：

```text
本机源：/home/yzliu/physical_ai/company_project/smart_project/outputs/groot_so101_synthetic_datasets/aws_third_training_20260804/
AWS 目标：/opt/dlami/nvme/smart_project_outputs/groot_so101_synthetic_datasets/aws_third_training_20260804/
```

其内部结构是：

```text
/opt/dlami/nvme/smart_project_outputs/
└── groot_so101_synthetic_datasets/
    └── aws_third_training_20260804/
        ├── real001/...
        ├── sim002/...
        ├── real003/...
        ├── sim004/...
        ├── sim005/...
        ├── real006/...
        ├── real007/...
        └── sim008/...
```

公共根目录下必须保留以下八个 trainer 实际读取的 leaf，不能多套或少套一级目录：

| ID | 相对 `aws_third_training_20260804/` 的 prepared leaf |
|---|---|
| real001 | `real001/20260624_pickupblock_randomposition_wristonly_ep50_wrist_only` |
| sim002 | `sim002/pick_camera1_notray_50_0625_1320_wrist_only` |
| real003 | `real003/real003_drop_ep0_v3_triple` |
| sim004 | `sim004/so101_test_lego_pick_triple` |
| sim005 | `sim005/20260611_pickplace_madeInIssac_merged_002_triple` |
| real006 | `real006/real006_drop_ep0_100_v3_triple` |
| real007 | `real007/real007_drop_ep0_53_100_121_200_211_v3_triple` |
| sim008 | `sim008/merged_3tasks_triple` |

训练输出写到同一块大盘：

```text
/opt/dlami/nvme/smart_project_outputs/groot_so101_synthetic_finetune/
└── <run-tag>/<dataset-id>/checkpoint-*
```

项目内看到的等价逻辑路径分别是：

```text
$AWS_PROJECT_DIR/outputs/groot_so101_synthetic_datasets/aws_third_training_20260804/
$AWS_PROJECT_DIR/outputs/groot_so101_synthetic_finetune/
```

GR00T checkout、HF/uv/Torch/Triton cache 和临时文件不需要第二条软链接，脚本直接放到
`/opt/dlami/nvme/{Isaac-GR00T,cache,tmp}`。`storage-link`、`bootstrap` 和所有 batch 动作
都会核对 `/opt/dlami/nvme` 与 `/` 不是同一设备，并默认要求至少 2500GiB 可用。

## 新实例环境配置

仍在目标机实际的 `aws_training/` 目录执行：

```bash
./aws_training_pipeline.sh bootstrap
./aws_training_pipeline.sh auth
./aws_training_pipeline.sh preflight
```

`bootstrap` 只安装最小系统依赖、checkout 固定 commit 并执行：

```text
uv sync --frozen --python 3.12
```

随后脚本读取真实 `nvcc --version`。若 toolkit major 为 13 或更高，它会验证锁定的
Triton 3.5.0 原生 `major >= 13` 分支和实际 PTX 映射，拒绝旧
`triton_cuda13_patch.pth` / 直接改写过的 `compiler.py`，并清除外部 `TRITON_PTXAS_PATH`。
同时它会确认实际编译器是 venv 中随 Triton 提供的 ptxas 12.8。这一步不会把系统 CUDA 13.2
改成 12.8，也不会把 cu128 wheel 改成 cu132。

上传 prepared v2.1 后，AWS 不需要 conda、LeRobot、Isaac Sim、IsaacLab、LeIsaac 或机器人
assets。raw v3 只在 AWS 现场重新做 `audit/prepare` 时才需要；正常训练不上传 raw。
`auth` 把 token 写入训练实际使用的 NVMe `HF_HOME`，不能改成裸 `hf auth login`。
`preflight` 下载并验证 manifest 固定的 base-model revision。
每份数据的 `stats` 必须在固定 GR00T commit 下隔离上传的旧 cache、强制重算，并通过
relative-action span-ratio 检查；stats 生成过程失败时脚本会自动恢复旧 stats，若生成成功但
span-ratio 门禁拒绝，则保留新 stats 供诊断但不会进入对应的 smoke/train。
本机 prepared 的不含 stats 全树 SHA256 已固定在 manifest；AWS 不一致会立即停止。
本次只需同步 `outputs/groot_so101_synthetic_datasets/aws_third_training_20260804/` 这一棵
prepared root，不要把历史 candidate/ablation 目录一并传上去。

## 八份数据分别训练

开始任何一份 `stats` 之前，八个 prepared leaf 必须已经全部上传。当前 formal preflight 会先
统一执行 `verify all` 和 `dry-run all`，所以不能只上传 real001 后就直接单跑 real001。

同一份数据的 `stats`、`smoke`、`train` 必须使用完全相同的 run-tag。下面是八份数据可以
分别复制执行的完整指令：

```bash
./aws_training_pipeline.sh stats real001 --run-tag aws-third-20260804-real001
./aws_training_pipeline.sh smoke real001 --run-tag aws-third-20260804-real001
./aws_training_pipeline.sh train real001 --run-tag aws-third-20260804-real001

./aws_training_pipeline.sh stats sim002 --run-tag aws-third-20260804-sim002
./aws_training_pipeline.sh smoke sim002 --run-tag aws-third-20260804-sim002
./aws_training_pipeline.sh train sim002 --run-tag aws-third-20260804-sim002

./aws_training_pipeline.sh stats real003 --run-tag aws-third-20260804-real003
./aws_training_pipeline.sh smoke real003 --run-tag aws-third-20260804-real003
./aws_training_pipeline.sh train real003 --run-tag aws-third-20260804-real003

./aws_training_pipeline.sh stats sim004 --run-tag aws-third-20260804-sim004
./aws_training_pipeline.sh smoke sim004 --run-tag aws-third-20260804-sim004
./aws_training_pipeline.sh train sim004 --run-tag aws-third-20260804-sim004

./aws_training_pipeline.sh stats sim005 --run-tag aws-third-20260804-sim005
./aws_training_pipeline.sh smoke sim005 --run-tag aws-third-20260804-sim005
./aws_training_pipeline.sh train sim005 --run-tag aws-third-20260804-sim005

./aws_training_pipeline.sh stats real006 --run-tag aws-third-20260804-real006
./aws_training_pipeline.sh smoke real006 --run-tag aws-third-20260804-real006
./aws_training_pipeline.sh train real006 --run-tag aws-third-20260804-real006

./aws_training_pipeline.sh stats real007 --run-tag aws-third-20260804-real007
./aws_training_pipeline.sh smoke real007 --run-tag aws-third-20260804-real007
./aws_training_pipeline.sh train real007 --run-tag aws-third-20260804-real007

./aws_training_pipeline.sh stats sim008 --run-tag aws-third-20260804-sim008
./aws_training_pipeline.sh smoke sim008 --run-tag aws-third-20260804-sim008
./aws_training_pipeline.sh train sim008 --run-tag aws-third-20260804-sim008
```

`smoke` 和 `train` 都会自动重新执行 preflight，不能绕过；二者还要求同一 run-tag 下已有
完全匹配的 fresh-stats lineage。若某份需要重跑，使用新 tag（例如追加 `-r2`），不要复用
已经非空的 run 目录。

仍然保留 `stats/smoke/train all --run-tag ...` 作为顺序批处理入口，但今晚逐份运行更容易
观察、停机和上传 checkpoint；`all` 也从不混合数据或并行争抢一张 H100。

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
| 明明有 MP4 但 preflight 报 `no prepared MP4` | 项目 `outputs` 是软链接；更新 pipeline。脚本使用 `find -H` 跟随入口链接，但不会跟随 prepared 内部链接 |
| CUDA 13+ / PTX 门禁失败 | 记录完整 traceback 和 `nvcc/torch/triton`；清掉旧 `.pth`/手改 venv 后重跑 `bootstrap`，不要执行 2.7/3.3.1 历史补丁 |
| Hugging Face 403/404 | 执行 `./aws_training_pipeline.sh auth` 并确认模型访问权限 |
| NVMe 空间不足 | 扩大 volume，或明确降低保留量并逐 run 上传；不要写 root filesystem |
| relative stats span ratio > 5 | 返回数据清洗；不能加 `--allow-relative-stats-outliers` 草率开训 |
| run directory not empty | 核对 lineage，换新 run tag；不要覆盖或混入另一个数据的 checkpoint |

环境通过只能证明 runtime 和数据合同可运行。训练 loss 很低仍不代表闭环策略成功；最终判断
必须按域进行：real checkpoint 在真机验证，sim checkpoint 在固定 Isaac/LeIsaac 协议验证。
