"""Probe Isaac Sim / IsaacLab / LeIsaac imports without installing anything.

Run this inside the target Isaac terminal after activating the candidate env:

    python experiments/groot_n17_isaac_smart_task/ubuntu_env_setup/isaac_stack_probe.py

Add --launch-app only when you want to test whether Isaac/Kit can start.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import traceback
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]


def add_path(path: Path) -> None:
    path = path.resolve()
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


def configure_workspace_paths() -> None:
    smart_project = Path(os.environ.get("SMART_PROJECT", REPO_ROOT)).expanduser().resolve()
    local_leisaac_root = smart_project / "leisaac"
    default_leisaac_root = local_leisaac_root if local_leisaac_root.exists() else Path.home() / "LeIsaac"
    leisaac_root = Path(os.environ.get("LEISAAC_ROOT", default_leisaac_root)).expanduser().resolve()
    isaaclab_source = leisaac_root / "dependencies" / "IsaacLab" / "source"

    os.environ.setdefault("LEISAAC_ASSETS_ROOT", str((leisaac_root / "assets").resolve()))

    add_path(EXPERIMENT_ROOT / "zero_shot_isaac_smart_task")
    add_path(leisaac_root / "source" / "leisaac")
    for package_name in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_mimic"):
        add_path(isaaclab_source / package_name)


def print_header() -> None:
    print("== environment ==")
    print("sys.executable:", sys.executable)
    print("python:", sys.version.replace("\n", " "))
    for key in (
        "CONDA_PREFIX",
        "SMART_PROJECT",
        "LEISAAC_ROOT",
        "LEISAAC_ASSETS_ROOT",
        "ISAACSIM_ROOT",
        "ISAAC_PATH",
        "ISAACLAB_PATH",
    ):
        print(f"{key}:", os.environ.get(key, ""))
    print("PYTHONPATH:", os.environ.get("PYTHONPATH", ""))
    print()


def try_import(module_name: str) -> bool:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        print(f"[FAIL] import {module_name}: {exc!r}")
        traceback.print_exc(limit=2)
        return False

    print(f"[ OK ] import {module_name}: {getattr(module, '__file__', '<namespace>')}")
    return True


def launch_app_probe() -> None:
    print()
    print("== SimulationApp probe ==")
    try:
        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher({"headless": True, "enable_cameras": True})
        simulation_app = app_launcher.app
        print("[ OK ] AppLauncher started SimulationApp")
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        print(f"[FAIL] AppLauncher/SimulationApp start: {exc!r}")
        traceback.print_exc()
        return

    try:
        print()
        print("== post-SimulationApp import probe ==")
        for module_name in (
            "isaaclab_tasks",
            "isaaclab_assets",
            "isaaclab_mimic",
            "leisaac",
            "leisaac.tasks",
            "gymnasium",
        ):
            try_import(module_name)

        try:
            import omni.kit.app

            manager = omni.kit.app.get_app().get_extension_manager()
            for extension_name in ("omni.physx", "omni.physx.tensors", "omni.physics.tensors"):
                try:
                    enabled = manager.is_extension_enabled(extension_name)
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] extension status {extension_name}: {exc!r}")
                else:
                    print(f"[INFO] extension {extension_name}: enabled={enabled}")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] cannot query extension manager: {exc!r}")

        try_import("omni.physics.tensors.impl.api")
    finally:
        try:
            simulation_app.close()
            print("[ OK ] SimulationApp closed")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] SimulationApp close: {exc!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch-app", action="store_true", help="Start headless Isaac SimulationApp and query physx extensions.")
    args = parser.parse_args()

    configure_workspace_paths()
    print_header()

    print("== pre-SimulationApp import probe ==")
    for module_name in (
        "numpy",
        "torch",
        "isaacsim",
        "isaaclab",
        "isaaclab.app",
    ):
        try_import(module_name)

    if args.launch_app:
        launch_app_probe()
    else:
        print()
        print("Run again with --launch-app before checking isaaclab_tasks/leisaac/omni extension imports.")


if __name__ == "__main__":
    main()
