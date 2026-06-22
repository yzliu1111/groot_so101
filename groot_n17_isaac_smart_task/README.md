# GR00T N1.7 + LeIsaac SmartTask 实验索引

这个实验现在按三步组织。根目录只保留这个索引；代码、说明和运行产物都放进对应阶段子目录。

| 阶段 | 目录 | 主要文件 |
|---|---|---|
| Zero-shot 推理 | `zero_shot_isaac_smart_task/` | `groot_bridge_server.py`、`run_smart_task_closed_loop.py`、`wire.py`、`franka_smart_task/` |
| Full fine-tune | `full_finetune_so101/` | `train_so101_synthetic_groot.py`、`so101_synthetic_groot_config.py` |
| Low-memory / LoRA / freeze | `lowmem_lora_freeze_so101/` | `train_so101_synthetic_groot_lowmem.py` |

推荐阅读顺序：

```text
1. zero_shot_isaac_smart_task/README_ZH.md
2. full_finetune_so101/README_ZH.md
3. lowmem_lora_freeze_so101/README_ZH.md
```

所有命令都以子目录入口为准。
