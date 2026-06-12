"""Franka Panda variant of the LeIsaac SmartTask scene.

设计目标：
    - 不修改 LeIsaac 原始 SO101 SmartTask。
    - 复用同一个 SmartScene USD、相机、红色 2x4 lego 资产。
    - 只把 robot / ee_frame / wrist camera / action cfg 换成 Franka Panda。

这个 task 是为了验证 GR00T N1.7 zero-shot 能力时，尽量减少 SO101 形态不匹配
带来的干扰。Franka 的 7DoF arm、parallel gripper 和 IK 控制都更接近
GR00T OXE/DROID 预训练 embodiment。
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg, TiledCameraCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from leisaac.tasks.smart_task.smart_task_env_cfg import (
    SmartTaskEnvCfg,
    SmartTaskSceneCfg,
    SmartTaskTerminationCfg,
)


@configclass
class FrankaSmartTaskSceneCfg(SmartTaskSceneCfg):
    """SmartTask scene with the SO101 follower replaced by Franka Panda."""

    # Franka root prim 仍然叫 Robot，这样 LeIsaac/IsaacLab 里引用
    # SceneEntityCfg("robot") 的 observation/reward/termination 不需要改名。
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Franka 的 EEF frame：
    # - prim_path 设在 panda_link0，表示 frame transformer 的源坐标系是机器人 root。
    # - 第 0 个 target frame 用于 ee_frame_state，即 GR00T 的 state.eef_9d。
    # - 第 1 个 target frame 用于距离 lego 的诊断/抓取判断，保留 SmartTask
    #   原 object_grasped 里 target_pos_w[:, 1, :] 的访问约定。
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.1034)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="grasp_center",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.1034)),
            ),
        ],
    )

    # 腕部相机挂到 Franka panda_hand 下。这里采用 copied IsaacLab 官方
    # `stack_ik_rel_visuomotor_env_cfg.py` 里的 Franka wrist camera offset：
    # 它比最初的保守近似更接近“安装在手腕、朝向 gripper 工作区”的视角。
    wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.13, 0.0, -0.15),
            rot=(-0.70614, 0.03701, 0.03701, -0.70614),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.5,
            focus_distance=400.0,
            horizontal_aperture=36.83,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,
    )

    # SingleArmTaskSceneCfg 里还定义了 front camera。SmartTask 的 policy 虽然会删掉
    # front observation，但 InteractiveScene 仍会实例化 scene cfg 中的 front sensor。
    # 因此 Franka 版必须把它从 SO101 的 Robot/base 改到 Franka 的 panda_link0。
    front: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0/front_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, -0.6, 0.5),
            rot=(0.1650476, -0.9862856, 0.0, 0.0),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.7,
            focus_distance=400.0,
            horizontal_aperture=38.11,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,
    )


@configclass
class FrankaSmartTaskTerminationCfg(SmartTaskTerminationCfg):
    """Termination config adjusted for Franka body names."""

    def __post_init__(self) -> None:
        # SmartTask 原 success 逻辑比较松；runner 默认仍会禁用它。
        # 这里仅把 robot base body 从 SO101 的 "base" 改成 Franka 的 "panda_link0"，
        # 方便显式打开 --use-env-success-termination 时不报 body name 错。
        if self.success is not None:
            self.success.params["robot_base_name"] = "panda_link0"


@configclass
class FrankaSmartTaskEnvCfg(SmartTaskEnvCfg):
    """Full env config for `Groot-Franka-SmartTask-v0`."""

    scene: FrankaSmartTaskSceneCfg = FrankaSmartTaskSceneCfg(env_spacing=8.0)
    terminations: FrankaSmartTaskTerminationCfg = FrankaSmartTaskTerminationCfg()

    # Franka 不是 LeRobot/SO101 数据采集设备，关闭 LeIsaac 的动态 gripper effort
    # 调整逻辑，避免它按 SO101 假设访问 gripper/link。
    dynamic_reset_gripper_effort_limit: bool = False
    robot_name: str = "franka_panda"

    def __post_init__(self) -> None:
        super().__post_init__()

        # SmartTaskEnvCfg.__post_init__ 会把 robot init pos 改成 SO101 的摆放位置。
        # Franka base 更接近 IsaacLab lift task 的布局，这里重新覆盖。
        # 这个位置是第一版 smoke test 起点，后续可根据 viewport 精调到更正对 lego。
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.0)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        # Franka 的关节 feature 名称仅用于日志/兼容，不参与 GR00T 推理。
        self.default_feature_joint_names = [
            "panda_joint1.pos",
            "panda_joint2.pos",
            "panda_joint3.pos",
            "panda_joint4.pos",
            "panda_joint5.pos",
            "panda_joint6.pos",
            "panda_joint7.pos",
            "panda_finger_joint1.pos",
            "panda_finger_joint2.pos",
        ]

        # 默认给 Franka 配 IK action，runner 后续也会按 --control-mode 显式调用。
        self.use_teleop_device("franka_ik")

    def use_teleop_device(self, teleop_device) -> None:
        """Configure Franka action terms.

        支持两个实验控制模式：
        - `franka_ik`：action = EEF pose(7D) + binary gripper(1D)，总 8D。
        - `franka_joint`：action = arm joints(7D) + binary gripper(1D)，总 8D。
        """

        self.task_type = teleop_device

        if teleop_device == "franka_ik":
            self.actions.arm_action = mdp.DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=["panda_joint.*"],
                body_name="panda_hand",
                controller=mdp.DifferentialIKControllerCfg(
                    command_type="pose",
                    use_relative_mode=False,
                    ik_method="dls",
                ),
                body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.1034)),
            )
        elif teleop_device == "franka_joint":
            self.actions.arm_action = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["panda_joint.*"],
                scale=1.0,
                use_default_offset=False,
            )
        else:
            raise ValueError(f"Unsupported Franka teleop device: {teleop_device}")

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
