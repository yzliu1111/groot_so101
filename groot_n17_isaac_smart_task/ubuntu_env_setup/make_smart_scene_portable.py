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


def resolve_leisaac_root() -> Path:
    smart_project = Path(os.environ.get("SMART_PROJECT", REPO_ROOT)).expanduser().resolve()
    local_leisaac_root = smart_project / "leisaac"
    default_leisaac_root = local_leisaac_root if local_leisaac_root.exists() else Path.home() / "LeIsaac"
    return Path(os.environ.get("LEISAAC_ROOT", default_leisaac_root)).expanduser().resolve()


def import_sdf():
    try:
        from pxr import Sdf
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "pxr is not importable. Run this inside the Isaac/LeIsaac environment, "
            'for example after `conda activate "$LEISAAC_ENV"`.'
        ) from exc
    return Sdf


def export_usd_to_text(source_path: Path) -> str:
    Sdf = import_sdf()
    layer = Sdf.Layer.FindOrOpen(str(source_path))
    if layer is None:
        raise SystemExit(f"Could not open USD layer: {source_path}")
    return layer.ExportToString()


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
    assets_root = Path(os.environ.get("LEISAAC_ASSETS_ROOT", leisaac_root / "assets")).expanduser().resolve()
    scene_dir = assets_root / "scenes" / "smart_scene"
    source_path = Path(args.source).expanduser().resolve() if args.source else scene_dir / "scene.usd"
    output_path = Path(args.output).expanduser().resolve() if args.output else scene_dir / "scene_portable.usda"

    if not source_path.exists():
        raise SystemExit(f"Source scene does not exist: {source_path}")

    text = export_usd_to_text(source_path)
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
