#!/usr/bin/env python3
"""
Stage 1: SFT training launcher using LLaMA-Factory.
Handles: environment setup, distributed launch, checkpoint management.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_dotenv() -> None:
    """Load a .env file from the project root (best-effort)."""
    try:
        from dotenv import load_dotenv  # noqa: PLC0415

        # Walk up from this file to find a .env
        here = Path(__file__).resolve()
        for parent in [here.parent, here.parent.parent, here.parent.parent.parent]:
            env_file = parent / ".env"
            if env_file.is_file():
                load_dotenv(env_file)
                print(f"[run_sft] Loaded .env from {env_file}", flush=True)
                return
        print("[run_sft] No .env file found; continuing without it.", flush=True)
    except ImportError:
        print("[run_sft] python-dotenv not installed; skipping .env loading.", flush=True)


def _project_root() -> Path:
    """Return the root directory of the project (contains train/, src/, …)."""
    return Path(__file__).resolve().parent.parent.parent


def _validate_config(config_path: Path) -> None:
    if not config_path.is_file():
        print(f"[run_sft] ERROR: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    import yaml  # noqa: PLC0415 (lazy; may not be installed in all envs)

    with config_path.open() as fh:
        cfg = yaml.safe_load(fh)

    output_dir = cfg.get("output_dir") or cfg.get("training_args", {}).get("output_dir")
    if output_dir:
        parent = Path(output_dir).parent
        if not parent.exists():
            print(
                f"[run_sft] WARNING: output_dir parent does not exist: {parent}. "
                "Creating it now.",
                flush=True,
            )
            parent.mkdir(parents=True, exist_ok=True)


def _build_env(num_gpus: int, project_root: Path) -> dict[str, str]:
    """Build the environment variable dict for the subprocess."""
    env = os.environ.copy()

    # CUDA devices
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))

    # PYTHONPATH: prepend project src/ so custom modules are importable
    src_path = str(project_root / "src")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_path}:{existing_pythonpath}" if existing_pythonpath else src_path

    # Weights & Biases – pull from .env / existing environment
    for wvar in ("WANDB_API_KEY", "WANDB_PROJECT", "WANDB_ENTITY", "WANDB_RUN_GROUP"):
        val = os.environ.get(wvar)
        if val:
            env[wvar] = val

    # Silence tokenizer warnings in distributed settings
    env["TOKENIZERS_PARALLELISM"] = "false"

    return env


def merge_lora_weights(config_path: Path, project_root: Path) -> None:
    """
    Merge LoRA adapters into the base model by calling::

        llamafactory-cli export <export_config>

    The export config is expected to live at ``configs/sft/export_merge.yaml``
    relative to the project root, or at a path derived from the training config
    name (e.g. ``train_sft.yaml`` → ``export_sft.yaml``).
    """
    # Derive export config path
    export_config = project_root / "configs" / "sft" / "export_merge.yaml"
    if not export_config.is_file():
        alt = config_path.parent / config_path.name.replace("train", "export").replace("sft", "export_merge")
        if alt.is_file():
            export_config = alt
        else:
            print(
                f"[run_sft] merge_lora_weights: export config not found at {export_config}. "
                "Skipping LoRA merge.",
                flush=True,
            )
            return

    llamafactory_cli = shutil.which("llamafactory-cli")
    if not llamafactory_cli:
        print("[run_sft] merge_lora_weights: llamafactory-cli not found in PATH. Skipping.", flush=True)
        return

    cmd = [llamafactory_cli, "export", str(export_config)]
    print(f"[run_sft] Merging LoRA: {' '.join(cmd)}", flush=True)

    try:
        subprocess.run(cmd, check=True)
        merged_dir = project_root / "checkpoints" / "sft" / "merged"
        print(
            f"\n[run_sft] LoRA merge complete.\n"
            f"  Merged model should be at: {merged_dir}\n"
            f"  Next step: run GRPO with --sft-checkpoint {merged_dir}",
            flush=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"[run_sft] merge_lora_weights failed (exit {exc.returncode}). Check the logs.", file=sys.stderr)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1: SFT training launcher using LLaMA-Factory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the LLaMA-Factory YAML training config.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=4,
        dest="num_gpus",
        help="Number of GPUs to use for training.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in output_dir.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Print the command that would be executed, then exit.",
    )
    parser.add_argument(
        "--merge-lora",
        action="store_true",
        dest="merge_lora",
        help="After training, merge LoRA adapters into the base model.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    _load_dotenv()

    project_root = _project_root()
    config_path = args.config.expanduser().resolve()

    print(f"[run_sft] Project root : {project_root}", flush=True)
    print(f"[run_sft] Config       : {config_path}", flush=True)
    print(f"[run_sft] Num GPUs     : {args.num_gpus}", flush=True)
    print(f"[run_sft] Resume       : {args.resume}", flush=True)

    _validate_config(config_path)

    # ------------------------------------------------------------------
    # Locate llamafactory-cli
    # ------------------------------------------------------------------
    llamafactory_cli = shutil.which("llamafactory-cli")
    if not llamafactory_cli:
        print(
            "[run_sft] ERROR: llamafactory-cli not found in PATH.\n"
            "  Install with: pip install llamafactory",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Build command
    #
    # LLaMA-Factory recommended launch:
    #   CUDA_VISIBLE_DEVICES=0,1,... llamafactory-cli train config.yaml
    #
    # For true multi-GPU DDP training the config should set:
    #   finetuning_type: lora
    #   ddp_find_unused_parameters: false
    #   (torchrun is invoked internally by accelerate/deepspeed if configured)
    # ------------------------------------------------------------------
    cmd: list[str] = [llamafactory_cli, "train", str(config_path)]

    if args.resume:
        cmd += ["--resume_from_checkpoint", "true"]

    env = _build_env(args.num_gpus, project_root)

    # ------------------------------------------------------------------
    # Print the command
    # ------------------------------------------------------------------
    env_prefix = f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}"
    full_cmd_str = f"{env_prefix} {' '.join(cmd)}"
    print(f"\n[run_sft] Command:\n  {full_cmd_str}\n", flush=True)

    if args.dry_run:
        print("[run_sft] --dry-run specified; exiting without executing.", flush=True)
        return

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------
    try:
        result = subprocess.run(cmd, env=env, check=False)
    except KeyboardInterrupt:
        print("\n[run_sft] Interrupted by user (KeyboardInterrupt). Exiting gracefully.", flush=True)
        sys.exit(130)

    if result.returncode != 0:
        print(
            f"\n[run_sft] Training exited with code {result.returncode}.",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(result.returncode)

    # ------------------------------------------------------------------
    # Post-training instructions
    # ------------------------------------------------------------------
    checkpoint_dir = project_root / "checkpoints" / "sft"
    print(
        f"\n[run_sft] Training complete!\n"
        f"  Checkpoint directory : {checkpoint_dir}\n"
        f"  Next steps:\n"
        f"    1. Merge LoRA adapters:\n"
        f"         python {__file__} --config {config_path} --merge-lora\n"
        f"       OR run manually:\n"
        f"         llamafactory-cli export configs/sft/export_merge.yaml\n"
        f"    2. Launch GRPO:\n"
        f"         python train/grpo/run_grpo.py \\\n"
        f"           --config configs/grpo/grpo_verl.yaml \\\n"
        f"           --sft-checkpoint {checkpoint_dir / 'merged'}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Optional: merge LoRA immediately
    # ------------------------------------------------------------------
    if args.merge_lora:
        merge_lora_weights(config_path, project_root)


if __name__ == "__main__":
    main()
