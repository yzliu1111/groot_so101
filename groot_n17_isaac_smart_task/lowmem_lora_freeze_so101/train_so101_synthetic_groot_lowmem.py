#!/usr/bin/env python3
"""Low-memory GR00T N1.7 fine-tuning entrypoint for SO101 prepared data.

This script is meant to run inside the Isaac-GR00T Python environment. It does
not import LeRobot and does not convert raw datasets. Use
``train_so101_synthetic_groot.py`` first if the prepared v2.1 copies under
``outputs/groot_so101_synthetic_datasets`` do not exist yet.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
DEFAULT_GROOT_ROOT = Path(os.environ.get("GROOT_ROOT", "/home/yzliu/Isaac-GR00T"))
DEFAULT_PREPARED_ROOT = REPO_ROOT / "outputs" / "groot_so101_synthetic_datasets"
DEFAULT_PREPARED_DATASETS = (
    DEFAULT_PREPARED_ROOT / "so101_lego_pick_0609_1722",
    DEFAULT_PREPARED_ROOT / "so101_lego_pick_0609_1722_mimic",
)
MODALITY_CONFIG_PATH = EXPERIMENT_ROOT / "full_finetune_so101" / "so101_synthetic_groot_config.py"

DEFAULT_LORA_TARGET_REGEX = (
    r"action_head\.model\..*(to_q|to_k|to_v|to_out\.0|proj_out_1|proj_out_2)$"
)
PROJECTOR_MODULES_TO_SAVE = [
    "state_encoder",
    "action_encoder",
    "action_decoder",
    "position_embedding",
]

STRATEGY_DEFAULTS: dict[str, dict[str, Any]] = {
    "projector-only": {
        "use_lora": False,
        "tune_projector": True,
        "tune_diffusion_model": False,
        "tune_vlln": False,
        "modules_to_save": [],
    },
    "diffusion-lora": {
        "use_lora": True,
        "tune_projector": False,
        # Keep the diffusion module in train mode; PEFT freezes base weights and
        # leaves only adapter parameters trainable.
        "tune_diffusion_model": True,
        "tune_vlln": False,
        "modules_to_save": [],
    },
    "projector-plus-diffusion-lora": {
        "use_lora": True,
        "tune_projector": True,
        # Keep the diffusion module in train mode; PEFT freezes base weights and
        # leaves only adapter parameters trainable.
        "tune_diffusion_model": True,
        "tune_vlln": False,
        "modules_to_save": PROJECTOR_MODULES_TO_SAVE,
    },
}


def _load_modality_config(modality_config_path: Path) -> None:
    path = modality_config_path.resolve()
    if not path.exists() or path.suffix != ".py":
        raise FileNotFoundError(f"Modality config path does not exist: {path}")
    if str(path.parent) not in sys.path:
        sys.path.append(str(path.parent))
    importlib.import_module(path.stem)
    print(f"Loaded modality config: {path}")


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _run(cmd: list[str], *, cwd: Path, dry_run: bool) -> None:
    printable = " ".join(cmd)
    print(f"[run] cd {cwd} && {printable}")
    if not dry_run:
        subprocess.run(cmd, cwd=cwd, check=True)


def _split_path_values(values: list[str] | None) -> list[Path]:
    if values is None:
        return [path.resolve() for path in DEFAULT_PREPARED_DATASETS]

    paths: list[Path] = []
    for value in values:
        for item in value.split(os.pathsep):
            if item:
                paths.append(Path(item).resolve())
    return paths


def _json_object(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return parsed


def _check_dataset_paths(dataset_paths: list[Path]) -> None:
    missing: list[Path] = []
    for path in dataset_paths:
        if not (path / "meta" / "info.json").exists():
            missing.append(path)
    if missing:
        joined = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Prepared dataset meta/info.json not found:\n  {joined}")


def _generate_stats(dataset_path: Path, groot_root: Path, dry_run: bool) -> None:
    _run(
        [
            sys.executable,
            "gr00t/data/stats.py",
            "--dataset-path",
            str(dataset_path),
            "--embodiment-tag",
            "NEW_EMBODIMENT",
            "--modality-config-path",
            str(MODALITY_CONFIG_PATH),
        ],
        cwd=groot_root,
        dry_run=dry_run,
    )


def _count_parameters(model: Any) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def _log_parameter_summary(model: Any, label: str) -> None:
    total, trainable = _count_parameters(model)
    ratio = 100 * trainable / total if total else 0.0
    logging.info("%s total parameters: %s", label, f"{total:,}")
    logging.info("%s trainable parameters: %s (%.4f%%)", label, f"{trainable:,}", ratio)


def _strategy_settings(args: argparse.Namespace) -> dict[str, Any]:
    settings = dict(STRATEGY_DEFAULTS[args.strategy])
    if args.tune_projector is not None:
        settings["tune_projector"] = args.tune_projector
    if args.tune_diffusion_model is not None:
        settings["tune_diffusion_model"] = args.tune_diffusion_model
    if args.tune_vlln is not None:
        settings["tune_vlln"] = args.tune_vlln
    if args.use_lora is not None:
        settings["use_lora"] = args.use_lora
    if args.lora_modules_to_save is not None:
        settings["modules_to_save"] = args.lora_modules_to_save
    return settings


def _patch_groot_pipeline(args: argparse.Namespace, settings: dict[str, Any]) -> None:
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.model import MODEL_REGISTRY
    from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline

    class LowMemGr00tN1d7Pipeline(Gr00tN1d7Pipeline):
        def _create_model(self):  # type: ignore[override]
            model = super()._create_model()
            strategy_path = self.save_cfg_dir / "lowmem_strategy.json"
            _json_dump(
                strategy_path,
                {
                    "strategy": args.strategy,
                    "use_lora": settings["use_lora"],
                    "tune_projector": settings["tune_projector"],
                    "tune_diffusion_model": settings["tune_diffusion_model"],
                    "tune_vlln": settings["tune_vlln"],
                    "lora_rank": args.lora_rank,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": args.lora_dropout,
                    "lora_target_regex": args.lora_target_regex,
                    "lora_modules_to_save": settings["modules_to_save"],
                },
            )

            if settings["use_lora"]:
                try:
                    from peft import LoraConfig, get_peft_model
                except ImportError as exc:  # pragma: no cover - env issue
                    raise RuntimeError(
                        "PEFT is required for LoRA strategies. Isaac-GR00T's "
                        "pyproject currently pins peft==0.17.1."
                    ) from exc

                lora_config = LoraConfig(
                    r=args.lora_rank,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    bias="none",
                    target_modules=args.lora_target_regex,
                    modules_to_save=settings["modules_to_save"] or None,
                )
                model = get_peft_model(model, lora_config)
                if hasattr(model, "print_trainable_parameters"):
                    model.print_trainable_parameters()

            _log_parameter_summary(model, "Low-memory strategy")
            return model

    MODEL_REGISTRY[Gr00tN1d7Config] = LowMemGr00tN1d7Pipeline


def _build_config(
    args: argparse.Namespace,
    dataset_paths: list[Path],
    settings: dict[str, Any],
):
    from gr00t.configs.base_config import get_default_config
    from gr00t.data.embodiment_tags import EmbodimentTag

    embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag).value
    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [str(path) for path in dataset_paths],
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    config.model.tune_llm = args.tune_llm
    config.model.tune_visual = args.tune_visual
    config.model.tune_projector = settings["tune_projector"]
    config.model.tune_diffusion_model = settings["tune_diffusion_model"]
    config.model.tune_vlln = settings["tune_vlln"]
    config.model.state_dropout_prob = args.state_dropout_prob
    config.model.random_rotation_angle = args.random_rotation_angle
    config.model.color_jitter_params = None if args.no_color_jitter else args.color_jitter_params
    config.model.extra_augmentation_config = (
        json.loads(args.extra_augmentation_config) if args.extra_augmentation_config else None
    )

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.model_name = args.backbone_model_name
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    config.training.experiment_name = args.experiment_name
    config.training.start_from_checkpoint = args.base_model_path
    config.training.optim = args.optim
    config.training.global_batch_size = args.global_batch_size
    config.training.dataloader_num_workers = args.dataloader_num_workers
    config.training.learning_rate = args.learning_rate
    config.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.training.output_dir = str(args.output_dir.resolve())
    config.training.save_steps = args.save_steps
    config.training.save_total_limit = args.save_total_limit
    config.training.num_gpus = args.num_gpus
    config.training.use_wandb = args.use_wandb
    config.training.max_steps = args.max_steps
    config.training.weight_decay = args.weight_decay
    config.training.warmup_ratio = args.warmup_ratio
    config.training.wandb_project = args.wandb_project
    config.training.gradient_checkpointing = args.gradient_checkpointing
    config.training.save_only_model = args.save_only_model
    config.training.skip_weight_loading = args.skip_weight_loading
    config.training.logging_steps = args.logging_steps

    config.data.shard_size = args.shard_size
    config.data.episode_sampling_rate = args.episode_sampling_rate
    config.data.num_shards_per_epoch = args.num_shards_per_epoch

    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        action="append",
        default=None,
        help="Prepared GR00T/LeRobot v2.1 dataset path. Repeat or pass os.pathsep-separated paths.",
    )
    parser.add_argument("--groot-root", type=Path, default=DEFAULT_GROOT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "groot_so101_synthetic_finetune",
    )
    parser.add_argument("--base-model-path", default="nvidia/GR00T-N1.7-3B")
    parser.add_argument("--backbone-model-name", default="nvidia/Cosmos-Reason2-2B")
    parser.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    parser.add_argument(
        "--strategy",
        choices=sorted(STRATEGY_DEFAULTS),
        default="projector-only",
        help=(
            "Low-memory strategy. projector-only saves a normal GR00T checkpoint; "
            "diffusion-lora saves a PEFT adapter; projector-plus-diffusion-lora "
            "also saves projector modules through PEFT modules_to_save."
        ),
    )
    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tune-projector", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tune-diffusion-model", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tune-vlln", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tune-llm", action="store_true")
    parser.add_argument("--tune-visual", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-regex", default=DEFAULT_LORA_TARGET_REGEX)
    parser.add_argument(
        "--lora-modules-to-save",
        action="append",
        default=None,
        help="Module suffix to save with a PEFT adapter. Repeat to override strategy defaults.",
    )
    parser.add_argument("--global-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--shard-size", type=int, default=2**10)
    parser.add_argument("--episode-sampling-rate", type=float, default=0.1)
    parser.add_argument("--num-shards-per-epoch", type=int, default=4096)
    parser.add_argument("--state-dropout-prob", type=float, default=0.2)
    parser.add_argument("--random-rotation-angle", type=int, default=None)
    parser.add_argument(
        "--color-jitter-params",
        type=_json_object,
        default={"brightness": 0.3, "contrast": 0.4, "saturation": 0.5, "hue": 0.08},
    )
    parser.add_argument("--no-color-jitter", action="store_true")
    parser.add_argument("--extra-augmentation-config", default=None)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-only-model", action="store_true")
    parser.add_argument("--skip-weight-loading", action="store_true")
    parser.add_argument("--skip-stats", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="finetune-gr00t-n1d7")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_paths = _split_path_values(args.dataset_path)
    _check_dataset_paths(dataset_paths)
    settings = _strategy_settings(args)

    if args.experiment_name is None:
        args.experiment_name = f"so101_smart_task_synthetic_{args.strategy}"

    if not args.skip_stats:
        for dataset_path in dataset_paths:
            _generate_stats(dataset_path, args.groot_root.resolve(), args.dry_run)

    _load_modality_config(MODALITY_CONFIG_PATH)
    _patch_groot_pipeline(args, settings)
    config = _build_config(args, dataset_paths, settings)

    print("[config] datasets:")
    for path in dataset_paths:
        print(f"  {path}")
    print(f"[config] strategy: {args.strategy}")
    print(f"[config] output_dir: {args.output_dir.resolve()}")
    print(f"[config] experiment_name: {args.experiment_name}")
    print(f"[config] CUDA_HOME: {os.environ.get('CUDA_HOME', '<unset>')}")

    if args.dry_run:
        print("[dry-run] config constructed; training not launched.")
        return

    from gr00t.experiment.experiment import run

    run(config)


if __name__ == "__main__":
    main()
