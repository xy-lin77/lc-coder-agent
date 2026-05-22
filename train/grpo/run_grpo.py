#!/usr/bin/env python3
"""
Stage 2: GRPO training launcher using verl.
Requires merged SFT model checkpoint.
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

        here = Path(__file__).resolve()
        for parent in [here.parent, here.parent.parent, here.parent.parent.parent]:
            env_file = parent / ".env"
            if env_file.is_file():
                load_dotenv(env_file)
                print(f"[run_grpo] Loaded .env from {env_file}", flush=True)
                return
        print("[run_grpo] No .env file found; continuing without it.", flush=True)
    except ImportError:
        print("[run_grpo] python-dotenv not installed; skipping .env loading.", flush=True)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _validate_sft_checkpoint(sft_checkpoint: Path) -> None:
    if not sft_checkpoint.exists():
        print(
            f"[run_grpo] ERROR: SFT checkpoint not found: {sft_checkpoint}\n"
            "  Run Stage 1 first:\n"
            "    python train/sft/run_sft.py --config configs/sft/train_sft.yaml --merge-lora",
            file=sys.stderr,
        )
        sys.exit(1)

    # Basic sanity: expect config.json or tokenizer_config.json
    has_model_files = any(
        (sft_checkpoint / f).is_file()
        for f in ("config.json", "tokenizer_config.json", "pytorch_model.bin")
    )
    if not has_model_files:
        print(
            f"[run_grpo] WARNING: SFT checkpoint directory exists but looks incomplete: {sft_checkpoint}",
            flush=True,
        )


def _log_gpu_memory() -> None:
    """Run nvidia-smi to show available GPU memory before launch."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        print("[run_grpo] nvidia-smi not found; skipping GPU memory check.", flush=True)
        return

    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,name,memory.total,memory.free,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            print("[run_grpo] GPU memory status (MiB):", flush=True)
            print("  index | name | total | free | used", flush=True)
            for line in result.stdout.strip().splitlines():
                print(f"  {line}", flush=True)
        else:
            print(f"[run_grpo] nvidia-smi returned {result.returncode}.", flush=True)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        print(f"[run_grpo] nvidia-smi failed: {exc}", flush=True)


def _build_env(num_gpus: int, project_root: Path) -> dict[str, str]:
    """Build the environment variable dict for the subprocess."""
    env = os.environ.copy()

    # Tokenizer / NCCL settings
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["NCCL_DEBUG"] = "WARN"

    # PYTHONPATH: prepend project src/ so verl can import CodeRewardManager
    src_path = str(project_root / "src")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_path}:{existing_pythonpath}" if existing_pythonpath else src_path

    # Weights & Biases
    for wvar in ("WANDB_API_KEY", "WANDB_PROJECT", "WANDB_ENTITY", "WANDB_RUN_GROUP"):
        val = os.environ.get(wvar)
        if val:
            env[wvar] = val

    # CUDA devices
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))

    return env


def _build_hydra_overrides(
    sft_checkpoint: Path,
    num_gpus: int,
    resume: bool,
    extra_overrides: list[str],
) -> list[str]:
    """
    Build the Hydra override list for verl.trainer.main_ppo.

    All overrides use Hydra's ``key=value`` format so they can be appended
    directly to the ``python -m verl.trainer.main_ppo`` command.
    """
    overrides: list[str] = [
        # Model path
        f"actor_rollout_ref.model.path={sft_checkpoint}",
        # GPU layout
        f"trainer.n_gpus_per_node={num_gpus}",
        # Custom reward function – verl loads this as a Python import path
        "custom_reward_fn.path=src.reward.reward_fn:CodeRewardManager",
        # Disable built-in reward model (we provide our own)
        "reward_model.enable=False",
    ]

    if resume:
        overrides.append("trainer.resume_training=True")

    overrides.extend(extra_overrides)
    return overrides


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2: GRPO training launcher using verl",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the verl Hydra YAML config (e.g. configs/grpo/grpo_verl.yaml).",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=4,
        dest="num_gpus",
        help="Number of GPUs to use per node.",
    )
    parser.add_argument(
        "--sft-checkpoint",
        type=Path,
        required=True,
        dest="sft_checkpoint",
        help="Path to the merged SFT model checkpoint directory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest GRPO checkpoint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Print the command that would be executed, then exit.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Extra Hydra override to append (can be specified multiple times). "
            "Example: --override trainer.total_epochs=3"
        ),
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
    sft_checkpoint = args.sft_checkpoint.expanduser().resolve()

    print(f"[run_grpo] Project root    : {project_root}", flush=True)
    print(f"[run_grpo] verl config     : {config_path}", flush=True)
    print(f"[run_grpo] SFT checkpoint  : {sft_checkpoint}", flush=True)
    print(f"[run_grpo] Num GPUs        : {args.num_gpus}", flush=True)
    print(f"[run_grpo] Resume          : {args.resume}", flush=True)

    # ------------------------------------------------------------------
    # Validations
    # ------------------------------------------------------------------
    if not config_path.is_file():
        print(f"[run_grpo] ERROR: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    _validate_sft_checkpoint(sft_checkpoint)

    # ------------------------------------------------------------------
    # GPU memory snapshot
    # ------------------------------------------------------------------
    _log_gpu_memory()

    # ------------------------------------------------------------------
    # Build Hydra overrides
    # ------------------------------------------------------------------
    overrides = _build_hydra_overrides(
        sft_checkpoint=sft_checkpoint,
        num_gpus=args.num_gpus,
        resume=args.resume,
        extra_overrides=args.override,
    )

    # ------------------------------------------------------------------
    # Build verl launch command
    #
    # verl uses Hydra for config management.  The recommended launch is:
    #
    #   python -m verl.trainer.main_ppo \
    #       --config-path <abs-path-to-config-dir> \
    #       --config-name <yaml-name-without-extension> \
    #       key1=value1 key2=value2 ...
    #
    # We derive --config-path and --config-name from the --config argument.
    # ------------------------------------------------------------------
    config_dir = str(config_path.parent)
    config_name = config_path.stem  # filename without .yaml

    cmd: list[str] = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        f"--config-path={config_dir}",
        f"--config-name={config_name}",
        *overrides,
    ]

    env = _build_env(args.num_gpus, project_root)

    # ------------------------------------------------------------------
    # Print the command
    # ------------------------------------------------------------------
    print("\n[run_grpo] Launch command:", flush=True)
    # Pretty-print with line continuation for readability
    print("  " + " \\\n    ".join(cmd), flush=True)
    print(
        f"\n[run_grpo] Key env vars:\n"
        f"  CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n"
        f"  PYTHONPATH={env['PYTHONPATH']}\n"
        f"  TOKENIZERS_PARALLELISM={env['TOKENIZERS_PARALLELISM']}\n"
        f"  NCCL_DEBUG={env['NCCL_DEBUG']}",
        flush=True,
    )

    if args.dry_run:
        print("\n[run_grpo] --dry-run specified; exiting without executing.", flush=True)
        return

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------
    try:
        result = subprocess.run(cmd, env=env, check=False)
    except KeyboardInterrupt:
        print("\n[run_grpo] Interrupted by user (KeyboardInterrupt). Exiting gracefully.", flush=True)
        sys.exit(130)

    if result.returncode != 0:
        print(
            f"\n[run_grpo] verl training exited with code {result.returncode}.",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(result.returncode)

    # ------------------------------------------------------------------
    # Post-training summary
    # ------------------------------------------------------------------
    grpo_checkpoint_dir = project_root / "checkpoints" / "grpo"
    print(
        f"\n[run_grpo] GRPO training complete!\n"
        f"  Checkpoint directory : {grpo_checkpoint_dir}\n"
        f"  Next steps:\n"
        f"    1. Evaluate the model:\n"
        f"         python eval/run_eval.py --model {grpo_checkpoint_dir}\n"
        f"    2. Or run inference directly with vLLM / HF generate.",
        flush=True,
    )


if __name__ == "__main__":
    main()
