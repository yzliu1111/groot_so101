# SO101 八份数据 full fine-tune

当前路线是 2026-08-04 的第三次 AWS 训练。唯一配置来源是
[`aws_tuning_8_manifest.json`](../aws_training/aws_tuning_8_manifest.json)，唯一批量入口是
[`aws_training_pipeline.sh`](../aws_training/aws_training_pipeline.sh)。AWS 专用文件集中在
同级 `aws_training/`，不要再从历史日志拼八条命令。

## 已固定的训练契约

- 八份数据始终启动八个独立 run；`all` 只是依次循环，绝不建立一个 mixture。
- 两份单相机数据使用 `wrist-only`，六份三相机数据使用 `triple`。没有 dual 数据。
- 相机数字没有全局语义；manifest 为每份数据显式指定 top、left、wrist。
- 原始 `dataset/aws_tuning/` 只读。清洗写到
  `outputs/groot_so101_cleaned_source_datasets/`，prepared v2.1 写到 `outputs/`。
- 多任务文本逐条保留。只有 sim004 的单任务 raw 标签经视频确认错误，prepared copy 显式改为
  `pick up block`。
- 正式 baseline 固定 batch 32、accumulation 1、learning rate `1e-4`、
  `state_dropout_prob=0`、显式零 ColorJitter。
- relative-action stats 在 AWS 固定 commit 下强制隔离旧 cache 后重算；full-range / q01-q99
  span ratio 超过 5 时训练 fail closed，不能靠低 loss 放行。
- 输出 run 目录必须为空；历史 real007 写进 real003 目录的碰撞不能再次发生。
- GR00T code、base model revision、pipeline code、prepared 全树和 stats SHA256 都写入 lineage；
  不含 stats 的本机 golden SHA256 固定在 manifest 中供 AWS 上传后比对。

## 今晚八个 run

| ID | layout | 训练 source | steps / save |
|---|---|---|---:|
| real001 | wrist-only，wrist=camera1 | raw 50 ep | 10000 / 2500 |
| sim002 | wrist-only，wrist=camera1 | raw 50 ep | 10000 / 2500 |
| real003 | triple，top=c3、left=c1、wrist=c2 | clean49，删 raw ep0 | 10000 / 2500 |
| sim004 | triple，top=front、left=left、wrist=wrist | raw49；修正单任务文本 | 10000 / 2500 |
| sim005 | triple，top=c1、left=c2、wrist=c3 | raw100 | 10000 / 2500 |
| real006 | triple，top=c3、left=c1、wrist=c2 | clean198，删 raw ep0/100 | 15000 / 3000 |
| real007 | triple，top=c3、left=c1、wrist=c2 | clean294，删 raw ep0/53/100/121/200/211 | 15000 / 3000 |
| sim008 | triple，top=c3、left=c1、wrist=c2 | raw287 | 10000 / 2500 |

episode、frame、task 文本和 prepared path 的精确期望值也在 manifest 中；目录名相似但内容
不一致时，`audit` 会直接停止。

## 本机：审计、准备和 dry-run

本机数据准备使用 conda `lerobot`。下面三步都以 manifest 为准：

```bash
cd /home/yzliu/physical_ai/company_project/smart_project/experiments

/home/yzliu/miniforge3/envs/lerobot/bin/python \
  groot_n17_isaac_smart_task/aws_training/aws_training_batch.py audit all

/home/yzliu/miniforge3/envs/lerobot/bin/python \
  groot_n17_isaac_smart_task/aws_training/aws_training_batch.py prepare all \
  --data-python /home/yzliu/miniforge3/envs/lerobot/bin/python \
  --groot-root /home/yzliu/Isaac-GR00T-py312

/home/yzliu/Isaac-GR00T-py312/.venv/bin/python \
  groot_n17_isaac_smart_task/aws_training/aws_training_batch.py dry-run all \
  --groot-root /home/yzliu/Isaac-GR00T-py312 \
  --run-tag local-contract-20260804
```

`prepare` 只写 prepared copy；不能对 `dataset/aws_tuning` 使用 `--force-prepare`。上传 AWS 时
同步实验代码、manifest 和这一棵 27GB prepared root：

```text
outputs/groot_so101_synthetic_datasets/aws_third_training_20260804/
```

AWS 不需要 raw v3、LeRobot、Isaac Sim、IsaacLab、LeIsaac 或 assets。

## AWS：统一入口

先把项目和 prepared data 放到 `/opt/dlami/nvme/smart_project`。在新实例上：

```bash
cd /opt/dlami/nvme/smart_project/experiments/groot_n17_isaac_smart_task/aws_training

./aws_training_pipeline.sh bootstrap
./aws_training_pipeline.sh auth
./aws_training_pipeline.sh preflight
./aws_training_pipeline.sh stats all --run-tag aws-third-20260804
```

`bootstrap` 固定已验证的 GR00T commit 和 uv 0.11.29，并执行
`uv sync --frozen --python 3.12`；不安装、
降级或替换 NVIDIA driver，也不安装 Isaac。`preflight` 检查 H100、FFmpeg/AV1、
torchcodec、`torch.compile`、精确依赖版本、八份 prepared contract、Hugging Face 访问和
NVMe 真实挂载/空间，并下载 manifest 固定 revision 的 base model。`auth` 与训练共用 NVMe
上的 `HF_HOME`。`stats all` 随后在同一个固定 commit 下强制重算 stats、执行 span-ratio
门控并记录 SHA256；不要只依赖另一台机器生成的 stats。
`smoke/train` 会强制查找同一个 run-tag 的 stats lineage；没有它或任何 SHA 不一致都会停止。

先逐份或全部做 1-step smoke：

```bash
./aws_training_pipeline.sh smoke all --run-tag aws-third-20260804
```

全部 smoke PASS 后，今晚的通用训练指令只有一条：

```bash
./aws_training_pipeline.sh train all --run-tag aws-third-20260804
```

`all` 不是并行启动。每个 run 完成后才进入下一个，checkpoint 位于：

```text
/opt/dlami/nvme/smart_project/outputs/groot_so101_synthetic_finetune/
└── aws-third-20260804/{real001,sim002,real003,sim004,sim005,real006,real007,sim008}
```

只跑一份时把 `all` 换成 ID，例如：

```bash
./aws_training_pipeline.sh stats real003 --run-tag aws-third-20260804-real003
./aws_training_pipeline.sh smoke real003 --run-tag aws-third-20260804-real003
./aws_training_pipeline.sh train real003 --run-tag aws-third-20260804-real003
```

## 不再使用的活动入口

- `lowmem_lora_freeze_so101/` 是小显存 fallback，不是 H100 八份 full fine-tune 的并行入口。
- 旧 dual real003、旧 real007 clean3、两份 0609 prepared 数据保留为历史/消融证据，但
  manifest 永远不会自动发现或训练它们。
- `00_docs/` 中 Python 3.10/3.11、旧 CUDA 和旧绝对路径属于历史采集或环境记录，不能覆盖
  本 README、manifest 与 AWS preflight 的当前契约。

参数解释见 [TRAINING_PARAMETER_REFERENCE_ZH.md](TRAINING_PARAMETER_REFERENCE_ZH.md)；AWS
故障定位见 [AWS_UBUNTU_FULL_FINETUNE_ZH.md](AWS_UBUNTU_FULL_FINETUNE_ZH.md)。
