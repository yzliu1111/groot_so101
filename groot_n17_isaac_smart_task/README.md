# GR00T N1.7 + LeIsaac SmartTask 实验索引

根目录只做导航。按当前任务选择一份 README，不需要从头读完全部文档。

操作手册负责“怎么跑”；跨文件语义和代码入口见
[TECHNICAL_CONTRACTS_ZH.md](TECHNICAL_CONTRACTS_ZH.md)。

当前默认 GR00T 环境是 Python 3.12 checkout：`~/Isaac-GR00T-py312`。旧的
`~/Isaac-GR00T` Python 3.10 checkout 只作为回滚 / 对照环境，不再作为新命令默认路径。
迁移目标机使用实际路径 `/home/guest1/Isaac-GR00T` 和
`/home/guest1/smart_project`；目录名不同不改变 Python 3.12 要求，具体 export 与自检命令见各阶段 README。

| 阶段 | 目录 | 主要文件 |
|---|---|---|
| Bridge + Isaac 部署 | `zero_shot_isaac_smart_task/` | camera check、dry-run、one-step、zero-shot 对照 |
| Full fine-tune | `full_finetune_so101/` | 训练入口、AWS Ubuntu 迁移说明、参数参考、modality configs |
| Low-memory / LoRA / freeze | `lowmem_lora_freeze_so101/` | `train_so101_synthetic_groot_lowmem.py` |
| Ubuntu 环境搭建 | `ubuntu_env_setup/` | `README_ZH.md` |

新终端不需要重新找整段 `export`。项目同步完成后，按机器执行一次：

```bash
# 目标机
source /home/guest1/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh target

# 本机（与上一条二选一）
source /home/yzliu/physical_ai/company_project/smart_project/experiments/groot_n17_isaac_smart_task/terminal_env.sh local
```

脚本只恢复路径变量；具体阶段仍会明确激活 GR00T、LeRobot 或 LeIsaac 环境。

按任务进入：

```text
新机器 / runtime 不通 -> ubuntu_env_setup/README_ZH.md
准备数据 / full FT     -> full_finetune_so101/README_ZH.md
AWS 只跑 full FT       -> full_finetune_so101/AWS_UBUNTU_FULL_FINETUNE_ZH.md
full FT OOM            -> lowmem_lora_freeze_so101/README_ZH.md
部署 checkpoint        -> zero_shot_isaac_smart_task/README_ZH.md
```

所有命令都以子目录入口为准；根目录只保留这个 README。
