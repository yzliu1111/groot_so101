"""Register the Franka version of the LeIsaac SmartTask experiment.

这个包只存在于 experiments 目录里，不修改 LeIsaac 主工程。导入本包时会向
gymnasium 注册一个新的 task id：

    Groot-Franka-SmartTask-v0

它复用 LeIsaac SmartTask 的场景和 lego 资产，但机器人替换为 Franka Panda。
"""

import gymnasium as gym


gym.register(
    id="Groot-Franka-SmartTask-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_smart_task_env_cfg:FrankaSmartTaskEnvCfg",
    },
)
