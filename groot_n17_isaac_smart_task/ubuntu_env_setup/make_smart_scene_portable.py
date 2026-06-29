"""Create a portable USDA copy of the LeIsaac SmartTask scene.

The original SmartTask USD may contain absolute paths from the machine that
authored the asset. This script exports the binary USD to USDA text and rewrites
known repo-local references to relative paths under smart_scene/.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]

ABSOLUTE_PATH_RE = re.compile(r"(?:file:)?/(?:home|Users)/[^@\n\r\"')\]]+")
PACKMAN_CAMERA_LINE_RE = re.compile(r"(?m)^.*(?:/home/ubuntu/\.cache/packman/|resources/models/camera/camera\.usd).*\n?")


def add_path(path: Path) -> None:
    path = path.resolve()
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


def resolve_leisaac_root() -> Path:
    smart_project = Path(os.environ.get("SMART_PROJECT", REPO_ROOT)).expanduser().resolve()
    local_leisaac_root = smart_project / "leisaac"
    default_leisaac_root = local_leisaac_root if local_leisaac_root.exists() else Path.home() / "LeIsaac"
    return Path(os.environ.get("LEISAAC_ROOT", default_leisaac_root)).expanduser().resolve()


def configure_workspace_paths(leisaac_root: Path) -> None:
    isaaclab_source = leisaac_root / "dependencies" / "IsaacLab" / "source"
    os.environ.setdefault("LEISAAC_ASSETS_ROOT", str((leisaac_root / "assets").resolve()))
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    add_path(EXPERIMENT_ROOT / "zero_shot_isaac_smart_task")
    add_path(leisaac_root / "source" / "leisaac")
    for package_name in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_mimic"):
        add_path(isaaclab_source / package_name)


def import_sdf(leisaac_root: Path):
    try:
        from pxr import Sdf
    except ModuleNotFoundError as exc:
        print("[INFO] pxr is not importable before Kit startup; launching headless Isaac app once.")
        configure_workspace_paths(leisaac_root)
        try:
            from isaaclab.app import AppLauncher

            app_launcher = AppLauncher({"headless": True, "enable_cameras": False})
            simulation_app = app_launcher.app
            from pxr import Sdf
        except Exception as app_exc:  # noqa: BLE001 - diagnostic setup path
            raise SystemExit(
                "pxr is not importable, and starting Isaac/Kit did not make it available. "
                "Run this inside the Isaac/LeIsaac environment and set "
                'LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" before Python.'
            ) from app_exc
        return Sdf, simulation_app
    return Sdf, None


def export_usd_to_text(source_path: Path, leisaac_root: Path) -> str:
    Sdf, simulation_app = import_sdf(leisaac_root)
    layer = Sdf.Layer.FindOrOpen(str(source_path))
    try:
        if layer is None:
            raise SystemExit(f"Could not open USD layer: {source_path}")
        return layer.ExportToString()
    finally:
        if simulation_app is not None:
            try:
                simulation_app.close()
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] SimulationApp close failed: {exc!r}")


def make_portable_text(text: str) -> str:
    replacements = {
        "file:/home/ubuntu/Khor/ss-leisaac/assets/scenes/smart_scene/assets/": "./assets/",
        "/home/ubuntu/Khor/ss-leisaac/assets/scenes/smart_scene/assets/": "./assets/",
        "file:/home/ubuntu/Khor/ss-leisaac/assets/scenes/smart_scene/": "./",
        "/home/ubuntu/Khor/ss-leisaac/assets/scenes/smart_scene/": "./",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # The camera preview mesh is an Omniverse UI artifact. The task defines real
    # cameras through TiledCameraCfg, so dropping stale packman preview references
    # avoids noisy "Could not open asset ... camera.usd" warnings.
    return PACKMAN_CAMERA_LINE_RE.sub("", text)


def find_absolute_paths(text: str) -> list[str]:
    return sorted(set(ABSOLUTE_PATH_RE.findall(text)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Write scene_portable.usda.")
    parser.add_argument("--check", action="store_true", help="Print remaining absolute paths after rewrite.")
    parser.add_argument("--source", default=None, help="Optional source USD path. Defaults to smart_scene/scene.usd.")
    parser.add_argument("--output", default=None, help="Optional output USDA path. Defaults to smart_scene/scene_portable.usda.")
    args = parser.parse_args()

    leisaac_root = resolve_leisaac_root()
    configure_workspace_paths(leisaac_root)
    assets_root = Path(os.environ.get("LEISAAC_ASSETS_ROOT", leisaac_root / "assets")).expanduser().resolve()
    scene_dir = assets_root / "scenes" / "smart_scene"
    source_path = Path(args.source).expanduser().resolve() if args.source else scene_dir / "scene.usd"
    output_path = Path(args.output).expanduser().resolve() if args.output else scene_dir / "scene_portable.usda"

    if not source_path.exists():
        raise SystemExit(f"Source scene does not exist: {source_path}")

    text = export_usd_to_text(source_path, leisaac_root)
    portable_text = make_portable_text(text)

    remaining_paths = find_absolute_paths(portable_text)
    if args.check:
        if remaining_paths:
            print("[WARN] Remaining absolute USD paths:")
            for path in remaining_paths:
                print(f"  {path}")
        else:
            print("[ OK ] No absolute /home or /Users USD paths remain after rewrite.")

    if args.write:
        output_path.write_text(portable_text, encoding="utf-8")
        print(f"[ OK ] Wrote portable scene: {output_path}")
    elif not args.check:
        print(portable_text)

    if remaining_paths:
        sys.exit(2)


if __name__ == "__main__":
    main()
