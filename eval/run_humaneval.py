#!/usr/bin/env python3
"""
HumanEval benchmark evaluation.
Measures pass@1 and pass@10 using the standard HumanEval test harness.

Usage:
    python eval/run_humaneval.py --model-path checkpoints/grpo-final
    python eval/run_humaneval.py --model-path checkpoints/grpo-final --greedy
    python eval/run_humaneval.py --model-path checkpoints/sft --num-samples-per-task 20 --temperature 0.8
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Rich is used for display; fall back gracefully if not installed
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box as rich_box

    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    console = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print(msg: str, style: str = "") -> None:
    if HAS_RICH:
        console.print(msg, style=style)
    else:
        print(msg)


def _info(msg: str) -> None:
    _print(f"[INFO] {msg}", style="bold cyan")


def _success(msg: str) -> None:
    _print(f"[OK]   {msg}", style="bold green")


def _warn(msg: str) -> None:
    _print(f"[WARN] {msg}", style="bold yellow")


def _error(msg: str) -> None:
    _print(f"[ERR]  {msg}", style="bold red")


# ---------------------------------------------------------------------------
# Code post-processing
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def extract_code(text: str) -> str:
    """
    Extract the first Python code block from a markdown-fenced response.
    If no fence is found, return the text as-is (model produced raw code).
    """
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1)
    # Some models emit a leading newline before the function
    return text


def sanitize_completion(prompt: str, raw_completion: str) -> str:
    """
    Strip the echoed prompt from the completion if the model repeated it,
    then extract the actual function body.
    """
    code = extract_code(raw_completion)

    # If the model echoed the entire prompt back, strip it
    if code.startswith(prompt.strip()):
        code = code[len(prompt.strip()):]

    # Remove stray trailing triple-backticks
    code = code.rstrip("`").rstrip()

    return code


# ---------------------------------------------------------------------------
# vLLM inference
# ---------------------------------------------------------------------------

def load_model(model_path: str, tensor_parallel_size: int):
    """
    Load the model with vLLM.  Returns an LLM instance.
    Raises SystemExit with a clear message if vllm is not installed.
    """
    try:
        from vllm import LLM  # noqa: PLC0415
    except ImportError:
        _error("vllm is not installed.  Install with:  pip install vllm>=0.5.0")
        sys.exit(1)

    _info(f"Loading model from {model_path}  (tp={tensor_parallel_size}) ...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        max_model_len=4096,
    )
    _success("Model loaded.")
    return llm


def build_sampling_params(temperature: float, max_tokens: int, n: int):
    """Build a vLLM SamplingParams object."""
    try:
        from vllm import SamplingParams  # noqa: PLC0415
    except ImportError:
        _error("vllm is not installed.  Install with:  pip install vllm>=0.5.0")
        sys.exit(1)

    if temperature == 0.0:
        return SamplingParams(
            n=n,
            temperature=0.0,
            max_tokens=max_tokens,
            stop=["```", "\nclass ", "\ndef ", "\n# "],
        )
    return SamplingParams(
        n=n,
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
        stop=["```", "\nclass ", "\ndef ", "\n# "],
    )


def generate_completions(
    llm,
    prompts: list[str],
    sampling_params,
    batch_size: int | None,
) -> list[list[str]]:
    """
    Run batched generation.  Returns a list-of-lists:
        completions[i] = list of `n` raw text completions for prompts[i].

    If batch_size is set, we split the prompt list into mini-batches to avoid
    OOM on smaller GPUs.  vLLM's internal scheduler already handles memory
    efficiently, so batch_size is mostly a courtesy guardrail.
    """
    if batch_size is None or batch_size >= len(prompts):
        outputs = llm.generate(prompts, sampling_params)
        result: list[list[str]] = []
        for req in outputs:
            result.append([o.text for o in req.outputs])
        return result

    result = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        _info(f"  Batch {start // batch_size + 1} / {-(-len(prompts) // batch_size)}")
        outputs = llm.generate(chunk, sampling_params)
        for req in outputs:
            result.append([o.text for o in req.outputs])
    return result


# ---------------------------------------------------------------------------
# HumanEval loading
# ---------------------------------------------------------------------------

def load_humaneval_problems() -> dict[str, Any]:
    """
    Load the 164 canonical HumanEval problems.
    Requires the `human_eval` package from https://github.com/openai/human-eval
    """
    try:
        from human_eval.data import read_problems  # noqa: PLC0415
    except ImportError:
        _error(
            "The `human_eval` package is not installed.\n"
            "Install it with:\n"
            "    pip install git+https://github.com/openai/human-eval.git\n"
            "Note: this package patches exec() — only use in isolated eval envs."
        )
        sys.exit(1)

    problems = read_problems()
    _success(f"Loaded {len(problems)} HumanEval problems.")
    return problems


# ---------------------------------------------------------------------------
# Sampling-file writer
# ---------------------------------------------------------------------------

def write_samples(path: Path, samples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    _success(f"Wrote {len(samples)} samples to {path}")


# ---------------------------------------------------------------------------
# Evaluate via human_eval harness
# ---------------------------------------------------------------------------

def run_evaluation(
    samples_path: Path,
    k_values: list[int],
    n_workers: int = 4,
) -> dict[str, float]:
    """
    Call human_eval.evaluation.evaluate_functional_correctness.
    Returns a dict like {"pass@1": 0.85, "pass@10": 0.95}.
    """
    try:
        from human_eval.evaluation import evaluate_functional_correctness  # noqa: PLC0415
    except ImportError:
        _error("human_eval package missing — cannot evaluate.")
        sys.exit(1)

    _info("Running functional correctness evaluation ...")
    results = evaluate_functional_correctness(
        str(samples_path),
        k=k_values,
        n_workers=n_workers,
    )
    return results


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

def print_results_table(results: dict, model_path: str) -> None:
    if not HAS_RICH:
        print("\n=== HumanEval Results ===")
        print(f"  Model : {model_path}")
        for metric, value in results.items():
            if isinstance(value, float):
                print(f"  {metric:15s}: {value * 100:.2f}%")
        return

    table = Table(
        title="HumanEval Evaluation Results",
        box=rich_box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric", style="bold white", min_width=18)
    table.add_column("Value", style="bold green", justify="right")

    table.add_row("Model", model_path)
    for metric in ["pass@1", "pass@5", "pass@10"]:
        if metric in results:
            table.add_row(metric, f"{results[metric] * 100:.2f}%")

    console.print(table)


# ---------------------------------------------------------------------------
# Per-problem breakdown (requires re-reading the annotated samples file)
# ---------------------------------------------------------------------------

def compute_per_problem_breakdown(annotated_samples_path: Path) -> dict[str, Any]:
    """
    human_eval writes a *_results.jsonl file next to the samples.
    Read it and compute per-task pass rates.
    """
    results_path = annotated_samples_path.with_suffix("").with_suffix(
        "_results.jsonl"
    )
    # human_eval appends _results suffix differently; try both conventions
    if not results_path.exists():
        results_path = Path(str(annotated_samples_path).replace(".jsonl", "_results.jsonl"))
    if not results_path.exists():
        _warn("Per-problem results file not found; skipping breakdown.")
        return {}

    per_task: dict[str, dict] = {}
    with open(results_path, encoding="utf-8") as fh:
        for line in fh:
            entry = json.loads(line)
            tid = entry["task_id"]
            if tid not in per_task:
                per_task[tid] = {"passed": 0, "total": 0}
            per_task[tid]["total"] += 1
            if entry.get("passed", False):
                per_task[tid]["passed"] += 1

    breakdown = {
        tid: {
            "passed": v["passed"],
            "total": v["total"],
            "pass_rate": v["passed"] / v["total"] if v["total"] else 0.0,
        }
        for tid, v in sorted(per_task.items())
    }
    return breakdown


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned model on the HumanEval benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the fine-tuned model (local dir or HF hub id).",
    )
    parser.add_argument(
        "--output-dir",
        default="eval/results/humaneval",
        help="Directory to write samples.jsonl and results.json.",
    )
    parser.add_argument(
        "--num-samples-per-task",
        type=int,
        default=10,
        help="Number of completions to sample per problem (for pass@k).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature.  Ignored when --greedy is set.",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding (temp=0, n=1).  Computes pass@1 only.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum number of new tokens per completion.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (vLLM TP).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Mini-batch size for generation (defaults to all-at-once).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Unused — kept for CLI parity.  GPU assignment is handled by CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--n-eval-workers",
        type=int,
        default=4,
        help="Parallel workers for functional-correctness test execution.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Resolve greedy mode
    # ------------------------------------------------------------------
    if args.greedy:
        temperature = 0.0
        num_samples = 1
        k_values = [1]
        _info("Greedy mode: temperature=0, n=1, measuring pass@1 only.")
    else:
        temperature = args.temperature
        num_samples = args.num_samples_per_task
        k_values = [k for k in [1, 5, 10] if k <= num_samples]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load problems
    # ------------------------------------------------------------------
    problems = load_humaneval_problems()

    # ------------------------------------------------------------------
    # Build prompts
    # ------------------------------------------------------------------
    task_ids = list(problems.keys())
    prompts = [problems[tid]["prompt"] for tid in task_ids]

    _info(f"Building {len(prompts)} prompts  (n={num_samples} each) ...")

    # ------------------------------------------------------------------
    # Load model & generate
    # ------------------------------------------------------------------
    llm = load_model(args.model_path, args.tensor_parallel_size)
    sampling_params = build_sampling_params(temperature, args.max_tokens, num_samples)

    t0 = time.time()
    completions_per_prompt = generate_completions(
        llm, prompts, sampling_params, args.batch_size
    )
    elapsed = time.time() - t0
    total_completions = sum(len(c) for c in completions_per_prompt)
    _success(
        f"Generated {total_completions} completions in {elapsed:.1f}s "
        f"({total_completions / elapsed:.1f} comp/s)"
    )

    # ------------------------------------------------------------------
    # Post-process and build sample list
    # ------------------------------------------------------------------
    samples: list[dict] = []
    for tid, prompt, completions in zip(task_ids, prompts, completions_per_prompt):
        for raw in completions:
            clean = sanitize_completion(prompt, raw)
            samples.append({"task_id": tid, "completion": clean})

    samples_path = output_dir / "samples.jsonl"
    write_samples(samples_path, samples)

    # ------------------------------------------------------------------
    # Functional correctness evaluation
    # ------------------------------------------------------------------
    _info(f"Evaluating with k={k_values} ...")
    pass_at_k = run_evaluation(samples_path, k_values, args.n_eval_workers)

    # ------------------------------------------------------------------
    # Per-problem breakdown
    # ------------------------------------------------------------------
    breakdown = compute_per_problem_breakdown(samples_path)

    # ------------------------------------------------------------------
    # Build results dict
    # ------------------------------------------------------------------
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    results: dict[str, Any] = {
        "model_path": args.model_path,
        "timestamp": timestamp,
        "benchmark": "HumanEval",
        "num_problems": len(problems),  # canonical = 164
        "num_samples_per_task": num_samples,
        "temperature": temperature,
        "max_tokens": args.max_tokens,
        "generation_time_seconds": round(elapsed, 2),
        **pass_at_k,
        "per_problem_breakdown": breakdown,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    _success(f"Results saved to {results_path}")

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    print_results_table(pass_at_k, args.model_path)


if __name__ == "__main__":
    main()
