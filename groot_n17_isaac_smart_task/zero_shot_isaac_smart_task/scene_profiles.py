"""Deployment-only SmartTask scene composition, independent of Gym task IDs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any


DEFAULT_SO101_TASK = "LeIsaac-SO101-SmartTask-v0"

TASK_DEFAULT_SCENE_PROFILE = "task-default"
TRAY_RED24_SCENE_PROFILE = "tray-red24"
TABLE_RED24_SCENE_PROFILE = "table-red24"
MULTI_LEGO_TRAY_SCENE_PROFILE = "multi-lego-tray"
SCENE_PROFILE_CHOICES = (
    TASK_DEFAULT_SCENE_PROFILE,
    TRAY_RED24_SCENE_PROFILE,
    TABLE_RED24_SCENE_PROFILE,
    MULTI_LEGO_TRAY_SCENE_PROFILE,
)

RED24_OBJECT_KEY = "red_2x4_lego_brick"
RED22_OBJECT_KEY = "red_2x2_lego_brick"
BLUE24_OBJECT_KEY = "blue_2x4_lego_brick"

RED24_ASSET = "red_2x4_lego_brick.usd"
RED22_ASSET = "red_2x2_lego_brick.usd"

# Poses mirror the authored single-pick and multi-LEGO prims in LeIsaac
# smart_scene/scene.usd. The single-pick profiles share the same object spec;
# their scene composition differs by tray activation.
SINGLE_PICK_RED24_POS = (0.0, 0.25, 0.025)
MULTI_RED24_POS = (0.126, 0.25, 0.015)
MULTI_RED22_POS = (0.126, 0.15, 0.015)
MULTI_BLUE24_POS = (0.186, 0.15, 0.015)
RED24_ROT = (0.70710677, 0.0, 0.0, -0.70710677)
IDENTITY_ROT = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class SceneObjectSpec:
    key: str
    asset_filename: str
    pos: tuple[float, float, float]
    rot: tuple[float, float, float, float]
    diffuse_color: tuple[float, float, float] | None = None
    roughness: float | None = None


@dataclass(frozen=True)
class SceneProfile:
    name: str
    tray_active: bool
    objects: tuple[SceneObjectSpec, ...]
    default_target_key: str | None
    requires_explicit_instruction: bool

    @property
    def object_keys(self) -> tuple[str, ...]:
        return tuple(obj.key for obj in self.objects)


_SINGLE_PICK_RED24 = SceneObjectSpec(
    key=RED24_OBJECT_KEY,
    asset_filename=RED24_ASSET,
    pos=SINGLE_PICK_RED24_POS,
    rot=RED24_ROT,
)

_MULTI_RED24 = SceneObjectSpec(
    key=RED24_OBJECT_KEY,
    asset_filename=RED24_ASSET,
    pos=MULTI_RED24_POS,
    rot=RED24_ROT,
)

SCENE_PROFILES = {
    TRAY_RED24_SCENE_PROFILE: SceneProfile(
        name=TRAY_RED24_SCENE_PROFILE,
        tray_active=True,
        objects=(_SINGLE_PICK_RED24,),
        default_target_key=RED24_OBJECT_KEY,
        requires_explicit_instruction=False,
    ),
    TABLE_RED24_SCENE_PROFILE: SceneProfile(
        name=TABLE_RED24_SCENE_PROFILE,
        tray_active=False,
        objects=(_SINGLE_PICK_RED24,),
        default_target_key=RED24_OBJECT_KEY,
        requires_explicit_instruction=False,
    ),
    MULTI_LEGO_TRAY_SCENE_PROFILE: SceneProfile(
        name=MULTI_LEGO_TRAY_SCENE_PROFILE,
        tray_active=True,
        objects=(
            _MULTI_RED24,
            SceneObjectSpec(
                key=RED22_OBJECT_KEY,
                asset_filename=RED22_ASSET,
                pos=MULTI_RED22_POS,
                rot=IDENTITY_ROT,
            ),
            SceneObjectSpec(
                key=BLUE24_OBJECT_KEY,
                asset_filename=RED24_ASSET,
                pos=MULTI_BLUE24_POS,
                rot=RED24_ROT,
                diffuse_color=(0.0, 0.0, 1.0),
                roughness=0.66364145,
            ),
        ),
        default_target_key=None,
        requires_explicit_instruction=True,
    ),
}


def get_scene_profile(name: str) -> SceneProfile:
    """Return one explicit deployment profile."""

    try:
        return SCENE_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"scene profile {name!r} is not an explicit deployment profile") from exc


def validate_scene_profile_task(task: str, profile_name: str) -> None:
    """Keep scene composition separate from arbitrary registered task configs."""

    if profile_name != TASK_DEFAULT_SCENE_PROFILE and task != DEFAULT_SO101_TASK:
        raise ValueError(
            f"--scene-profile {profile_name} requires --task {DEFAULT_SO101_TASK}; "
            "--task selects the base environment while --scene-profile selects its object layout"
        )


def resolve_scene_profile_target(profile_name: str, requested_key: str) -> str | None:
    """Resolve the metric/debug target without guessing in the multi-LEGO profile."""

    if profile_name == TASK_DEFAULT_SCENE_PROFILE:
        return None

    profile = get_scene_profile(profile_name)
    if requested_key == "auto":
        if profile.default_target_key is not None:
            return profile.default_target_key
        raise ValueError(
            f"--scene-profile {profile_name} contains multiple LEGO objects; "
            f"pass --target-object-key with one of {list(profile.object_keys)}"
        )

    if requested_key not in profile.object_keys:
        raise ValueError(
            f"--target-object-key {requested_key!r} is not part of --scene-profile {profile_name}; "
            f"choose one of {list(profile.object_keys)}"
        )
    return requested_key


def build_scene_wrapper_text(source_scene_usd: Path, profile_name: str) -> str:
    """Compose the base scene while overriding tray and legacy LEGO activation."""

    profile = get_scene_profile(profile_name)
    source_scene_usd = source_scene_usd.expanduser().resolve()
    source_asset_path = source_scene_usd.as_posix()
    if "@" in source_asset_path:
        raise ValueError(f"USD source path cannot contain '@': {source_scene_usd}")
    tray_active = "true" if profile.tray_active else "false"

    return f'''#usda 1.0
(
    defaultPrim = "world"
    subLayers = [
        @{source_asset_path}@
    ]
)

over "world"
{{
    over "tray" (
        active = {tray_active}
    )
    {{
    }}

    over "red_2x4_lego_brick" (
        active = false
        payload = None
    )
    {{
    }}

    over "blue_2x4_lego_brick" (
        active = false
        payload = None
    )
    {{
    }}

    over "red_2x2_lego_brick" (
        active = false
        payload = None
    )
    {{
    }}

    over "red_2x4_lego_brick_pick" (
        active = false
        payload = None
    )
    {{
    }}

    over "camera_left_xform"
    {{
        over "camera_left_preview"
        {{
            over "OmniverseKitViewportCameraMesh" (
                active = false
                references = None
            )
            {{
            }}
        }}
    }}

    over "camera_front_xform"
    {{
        over "camera_front_preview"
        {{
            over "OmniverseKitViewportCameraMesh" (
                active = false
                references = None
            )
            {{
            }}
        }}
    }}
}}
'''


def write_scene_wrapper(source_scene_usd: Path, profile_name: str, output_dir: Path) -> Path:
    """Write the file-backed layer seen by both LeIsaac's parser and live simulation."""

    source_scene_usd = source_scene_usd.expanduser().resolve()
    content = build_scene_wrapper_text(source_scene_usd, profile_name)
    source_digest = hashlib.sha256(str(source_scene_usd).encode("utf-8")).hexdigest()[:10]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{profile_name}_{source_digest}.usda"
    if not output_path.exists() or output_path.read_text(encoding="utf-8") != content:
        output_path.write_text(content, encoding="utf-8")
    return output_path


def add_scene_profile_objects(env_cfg: Any, profile_name: str, asset_dir: Path) -> tuple[str, ...]:
    """Add complete managed LEGO assets to an IsaacLab env cfg."""

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    profile = get_scene_profile(profile_name)
    asset_dir = asset_dir.expanduser().resolve()
    added_keys: list[str] = []
    for obj in profile.objects:
        asset_path = asset_dir / obj.asset_filename
        if not asset_path.is_file():
            raise FileNotFoundError(f"scene-profile LEGO asset does not exist: {asset_path}")
        spawn_kwargs: dict[str, Any] = {
            "usd_path": str(asset_path),
            "rigid_props": sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            "collision_props": sim_utils.CollisionPropertiesCfg(),
        }
        if obj.diffuse_color is not None:
            spawn_kwargs["visual_material"] = sim_utils.PreviewSurfaceCfg(
                diffuse_color=obj.diffuse_color,
                roughness=obj.roughness,
            )

        setattr(
            env_cfg.scene,
            obj.key,
            RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Scene/{obj.key}_managed",
                spawn=sim_utils.UsdFileCfg(**spawn_kwargs),
                init_state=RigidObjectCfg.InitialStateCfg(pos=obj.pos, rot=obj.rot),
            ),
        )
        added_keys.append(obj.key)
    return tuple(added_keys)


def apply_scene_profile_target(env_cfg: Any, profile_name: str, target_key: str) -> None:
    """Retarget inherited SmartTask terms and keep explicit profile poses deterministic."""

    from isaaclab.managers import SceneEntityCfg

    profile = get_scene_profile(profile_name)
    if target_key not in profile.object_keys:
        raise ValueError(f"target {target_key!r} is not part of scene profile {profile_name!r}")

    target_cfg = SceneEntityCfg(target_key)
    env_cfg.target_object_name = target_key
    env_cfg.target_object_label = target_key

    observations = getattr(env_cfg, "observations", None)
    subtask_terms = getattr(observations, "subtask_terms", None)
    for term_name in ("lego_grasped", "lego_lifted", "lego_placed_on_tray"):
        term = getattr(subtask_terms, term_name, None)
        params = getattr(term, "params", None)
        if isinstance(params, dict) and "object_cfg" in params:
            params["object_cfg"] = target_cfg

    terminations = getattr(env_cfg, "terminations", None)
    success = getattr(terminations, "success", None)
    success_params = getattr(success, "params", None)
    if isinstance(success_params, dict) and "object_cfg" in success_params:
        success_params["object_cfg"] = target_cfg
    if profile_name == MULTI_LEGO_TRAY_SCENE_PROFILE and success is not None:
        from leisaac.tasks.smart_task import mdp as smart_task_mdp

        success.func = smart_task_mdp.object_placed_on_tray

    # Inherited SmartTask randomizers reference legacy keys or would overwrite
    # the authored deployment layout. Explicit profiles therefore disable all
    # domain-randomization terms while preserving unrelated reset events.
    events = getattr(env_cfg, "events", None)
    if events is not None:
        for event_name in dir(events):
            if event_name.startswith("domain_randomize_"):
                setattr(events, event_name, None)
