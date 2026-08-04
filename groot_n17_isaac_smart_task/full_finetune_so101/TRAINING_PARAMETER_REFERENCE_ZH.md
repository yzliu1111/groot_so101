# SO101 full fine-tune 参数参考

这是一份起跑参数表，不是对所有数据集都成立的最优超参数。先固定数据、相机 layout、
learning rate 和模型版本，只改变 batch / accumulation / steps，才能解释实验差异。

2026-08-04 的八份正式 run 以 `../aws_training/aws_tuning_8_manifest.json` 为唯一配置；本文件的通用区间不能
覆盖 manifest 中已按历史可比性固定的 10000/15000 steps。

官方参数语义见
[FinetuneConfig](https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/configs/finetune_config.py)，
单卡配置见
[NVIDIA hardware recommendations](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/hardware_recommendation.md)。

## 1. 三个参数到底控制什么

### `global_batch_size`

它是一次 forward/backward 在所有 GPU 上合计的 batch，发生在 gradient accumulation 之前。

```text
单卡 micro batch = global_batch_size
多卡每卡 batch = global_batch_size / num_gpus
```

因此多卡时 `global_batch_size` 至少要能被 GPU 数量整除。

### `gradient_accumulation_steps`

它表示多少次 forward/backward 后做一次 optimizer update：

```text
B_eff = global_batch_size × gradient_accumulation_steps
```

`B_eff` 是每个 optimizer step 累积看到的样本数。accumulation 不能修复单个 micro batch 已经
OOM 的问题；OOM 时先把 `global_batch_size` 降到能稳定运行的值。

### `max_steps`

它是 optimizer update 的次数，不是 frame 数，也不是 micro-batch 次数：

```text
大致处理的训练窗口数 = max_steps × B_eff
```

如果把 `B_eff` 从 16 改到 32，又希望保持相同样本预算：

```text
new_max_steps = old_max_steps × old_B_eff / new_B_eff
```

## 2. 用有效帧数估算训练预算

当前 SO101 modality config 使用 16-step action chunk。每个 episode 末尾不足 16 帧的起点
不能组成完整 action window，因此：

```text
N_eff = Σ max(0, episode_frames - 15)
近似 exposure 次数 P = max_steps × B_eff / N_eff
```

如果只知道总 frames，可以先用总 frames 粗算；episode 很多且很短时，必须改用上面的
`N_eff`。这里的 exposure 只是样本窗口预算，不等于独立同分布意义上的 epoch，相邻机器人
帧具有很强时间相关性。

多数据集一起训练时，先对所有 participating dataset 的 `N_eff` 求和。当前 GR00T 默认混合
权重大致随数据集长度增长，因此 27 万帧数据会显著压过 7000 帧数据；如果小数据代表一个
必须学会的独立任务，不能只看总帧数，最好分别训练或单独设计采样权重。

## 3. 单卡 H100：优先使用物理 batch 32

NVIDIA 当前硬件建议把 1x H100 的 quick-start global batch 设为 32。对于当前项目的
projector + diffusion 路线，推荐先直接测试物理 batch 32，而不是用 batch 1 累积 32 次：

| H100 配置 | `global_batch_size` | `gradient_accumulation_steps` | `B_eff` |
|---|---:|---:|---:|
| 1-step smoke | 1 | 1 | 1 |
| 20-step H100 pilot | 32 | 1 | 32 |
| batch 32 OOM 时回退 | 16 | 1 | 16 |
| 只有物理 batch 受限但仍需 `B_eff=32` | 16 | 2 | 32 |

选择顺序：

1. 用 batch 1 完成 1-step 环境 smoke。
2. 用 batch 32 完成 20-step H100 throughput pilot，并记录峰值显存。
3. batch 32 稳定后，把它作为第一条正式 baseline；不必先加 gradient accumulation。
4. batch 32 OOM 才回退到 16；只有需要保持 `B_eff=32` 时再设 accumulation 2。
5. 改 `B_eff` 时按上一节公式缩放 `max_steps`，否则对照实验的样本预算不公平。

如果 H100 在 batch 32 下显存、吞吐都很宽裕，可以把 batch 48 或 64 作为后续吞吐实验，但
它们不是第一条 baseline；改变 batch 后也要同步缩放 `max_steps`。

## 4. 不同数据量的起始 `max_steps`

下面按单卡 H100 `B_eff=32` 粗算，目标是先得到一个可比较 baseline，而不是一次训练到极限：

| 总有效帧数级别 | 建议初始 `max_steps` | 大致窗口 exposure | `save_steps` |
|---:|---:|---:|---:|
| 约 7k | 1k–2k | 约 4.6–9.1 次 | 250 |
| 约 30k | 2.5k–4k | 约 2.7–4.3 次 | 500 |
| 约 100k | 5k–7.5k | 约 1.6–2.4 次 | 500 |
| 约 270k | 10k–15k | 约 1.2–1.8 次 | 500–1000 |

更实用的 staged 做法：

```text
20 steps       H100 显存 / 吞吐 pilot
500 steps      loss 趋势与吞吐 pilot
表中下限       第一个 baseline
只在指标仍改善时延长到表中上限
```

如果回退到 `B_eff=16`，上表的 `max_steps` 约加倍；如果物理 batch 提到 64，约减半。

当前本地两份 prepared 数据合计 17771 raw frames、100 episodes。扣除每个 episode 最后的
15 个不完整 action-window 起点后，`N_eff≈16271`；用 H100 batch 32 跑 1000/2000 steps，
大约相当于 2.0/3.9 次窗口 exposure。

7000 帧的小数据最容易出现 training loss 很低但泛化变差。它不应该因为文件小就使用更大
batch；相反，要更早保存 checkpoint、更早做 held-out 检查，并避免无依据地跑几十次 exposure。

## 5. 其他参数的 baseline

| 参数 | 建议起点 | 说明 |
|---|---:|---|
| `learning_rate` | `1e-4` | 先保持项目/官方路线默认；不要与 batch、steps 同时改 |
| `dataloader_num_workers` | `4` | GPU 等数据且 CPU/RAM 足够时再试 8 |
| `save_total_limit` | manifest 的 4 或 5 | 保留平台期前后 checkpoint，并受 NVMe 空间门控 |
| `state_dropout_prob` | `0` | 上游 processor 与 model 会分别应用；SO101 baseline 不做双重随机遮蔽 |
| ColorJitter | 显式全零 | 不省略参数；省略可能继承 base processor 的非零配置 |

如果 loss 出现 NaN、突然放大或 grad norm 持续爆炸，优先停止并查数据、混合精度和 learning
rate，不要靠增加 `max_steps` 解决。

## 6. loss 下降能证明什么

loss 持续下降能够支持这些结论：

- data loader 能持续产出可计算 batch；
- forward、backward、optimizer update 基本在工作；
- 当前模型可以拟合当前训练 objective。

它不能保证：

- 相机角色、语言和 action 在时间上正确对齐；
- action 单位、关节顺序和 normalization 语义正确；
- 训练集没有泄漏、重复或常量标签；
- checkpoint 对未见 episode 或真实闭环任务有效。

错误但一致的数据同样可以得到漂亮的下降曲线。因此“上次 loss 在下降”说明训练循环跑通的
可信度提高了，但不能单独证明整个训练管道准确无误。

## 7. `loss ≈ 0.03` 是否算收敛

没有跨任务通用的 GR00T loss 阈值。绝对数值会随 normalization、action 维度、diffusion
采样/加权、augmentation、batch 和代码版本变化。只比较同一模型版本、同一数据语义和同一
配置下的曲线。

`0.03` 的合理解释是：

```text
它已经足够低，可以开始比较 checkpoint；
它本身不能证明已经收敛，更不能证明策略成功。
```

经验上用“相对平台期”而不是固定数字停止：

1. 对 training loss 做 100–200 optimizer steps 的滑动平均。
2. 在最近 500–1000 steps，或总预算最后 10%–20% 内，平滑 loss 的相对改善小于约
   1%–2%，可以认为训练 objective 接近平台期。
3. 同期 held-out / open-loop 指标没有继续改善时，停止更有依据。
4. training loss 继续下降而 held-out 指标变差，是过拟合，不是“更好地收敛”。

当前训练入口本身不提供足够强的 held-out 成功证明。如果只有 training loss，可以说
“训练 objective 看起来进入平台期”，不能说“策略已经收敛并可用”。

建议在 `0.03` 附近保存平台期之前、平台期中间和平台期之后三个 checkpoint，回传本地后使用
同一批 held-out episode 做 open-loop 对比。最好的 checkpoint 未必是 training loss 最低的
最后一个。

## 8. 一个可直接采用的初始决策

如果 AWS 是单卡 H100：

```text
1-step smoke: global_batch_size = 1, gradient_accumulation_steps = 1
正式 baseline: global_batch_size = 32, gradient_accumulation_steps = 1
B_eff = 32
learning_rate = 1e-4
state_dropout_prob = 0
ColorJitter = 0
save_steps / max_steps = manifest per-dataset values
```

然后按 `N_eff` 选择 `max_steps`：7000 帧先 1000，27 万帧先 10000；中间规模按表格插值。
到达下限后先看平台期和 held-out checkpoint 对比，再决定是否延长，不要仅因为 loss 还小于
某个绝对值就继续烧算力。
