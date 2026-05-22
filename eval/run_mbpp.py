#!/usr/bin/env python3
"""
MBPP (Mostly Basic Python Problems) benchmark evaluation.
Evaluates pass@1 using the sanitized MBPP test set (374 problems).

Usage:
    python eval/run_mbpp.py --model-path checkpoints/grpo-final
    python eval/run_mbpp.py --model-path checkpoints/grpo-final --greedy
    python eval/run_mbpp.py --model-path checkpoints/sft --split validation --subset 100
"""

import argparse
import ast
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Rich display helpers
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box as rich_box
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    console = None  # type: ignore[assignment]


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
# Prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Write concise, correct Python functions. "
    "Output only the function code — no explanation, no imports unless needed."
)


def build_prompt(problem: dict) -> str:
    """
    Build a chat-style prompt for MBPP.
    The model should only emit the function body (no test harness).
    """
    text: str = problem["text"].strip()
    # Use only the first assertion as a hint — avoids leaking all tests
    test_hint: str = problem["test_list"][0].strip() if problem.get("test_list") else ""

    lines = [
        "You are an expert Python programmer. Write a Python function for the following task.",
        "",
        f"Task: {text}",
    ]
    if test_hint:
        lines += [
            "",
            "Your code should pass the following tests:",
            f"  {test_hint}",
        ]
    lines += [
        "",
        "Write only the function, no explanation needed.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Code extraction / sanitization
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1)
    return text


def sanitize_completion(raw: str) -> str:
    """
    Extract a clean Python function from a raw model completion.
    Strips markdown fences, trailing explanations, and stray backticks.
    """
    code = extract_code(raw).strip()
    # Drop lines after the first blank line that follows the function def
    # (heuristic to cut off post-function commentary)
    lines = code.splitlines()
    kept: list[str] = []
    in_function = False
    blank_streak = 0
    for line in lines:
        if line.startswith("def "):
            in_function = True
            blank_streak = 0
        if in_function and line.strip() == "":
            blank_streak += 1
            if blank_streak > 1:
                break
        elif in_function:
            blank_streak = 0
        kept.append(line)
    return "\n".join(kept).strip()


# ---------------------------------------------------------------------------
# Sandboxed executor
# ---------------------------------------------------------------------------

_EXEC_TIMEOUT = 10  # seconds per test case


def _run_tests_in_subprocess(
    code: str,
    tests: list[str],
    timeout: int = _EXEC_TIMEOUT,
) -> dict[str, Any]:
    """
    Execute the generated function + each test assertion in a fresh subprocess.

    Returns:
        {
            "passed": bool,       # True if all tests passed
            "num_passed": int,
            "num_tests": int,
            "error_type": str | None,   # "syntax_error" | "runtime_error" | "wrong_answer"
            "error_message": str | None,
        }
    """
    script_lines = [
        "import sys",
        textwrap.dedent(code),
        "",
    ]
    for test in tests:
        # Wrap each assertion in try/except so we get meaningful output
        script_lines.append(
            f"try:\n    {test}\nexcept Exception as _e:\n    print(f'FAIL: {test!r} => ' + str(_e), file=sys.stderr)\n    sys.exit(1)"
        )
    script = "\n".join(script_lines)

    # Quick syntax check before spawning subprocess
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return {
            "passed": False,
            "num_passed": 0,
            "num_tests": len(tests),
            "error_type": "syntax_error",
            "error_message": str(exc),
        }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(script)
        tmp_path = tmp.name

    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode == 0:
            return {
                "passed": True,
                "num_passed": len(tests),
                "num_tests": len(tests),
                "error_type": None,
                "error_message": None,
            }
        stderr = proc.stderr.strip()
        # Distinguish runtime error from assertion failure
        error_type = "wrong_answer" if "AssertionError" in stderr else "runtime_error"
        return {
            "passed": False,
            "num_passed": 0,  # conservative: we stop at first failure
            "num_tests": len(tests),
            "error_type": error_type,
            "error_message": stderr[:500],
        }
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "num_passed": 0,
            "num_tests": len(tests),
            "error_type": "timeout",
            "error_message": f"Exceeded {timeout}s execution limit.",
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def evaluate_completions_parallel(
    items: list[dict],
    n_workers: int = 8,
) -> list[dict]:
    """
    Execute test cases for all items in parallel using a thread pool.
    Each item has keys: task_id, code, tests.
    Returns items augmented with execution results.
    """
    results: list[dict] = [{}] * len(items)
    work_q: queue.Queue = queue.Queue()
    for idx, item in enumerate(items):
        work_q.put((idx, item))

    def worker() -> None:
        while True:
            try:
                idx, item = work_q.get_nowait()
            except queue.Empty:
                return
            result = _run_tests_in_subprocess(item["code"], item["tests"])
            merged = {**item, **result}
            results[idx] = merged
            work_q.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(n_workers)]
    for t in threads:
        t.start()
    work_q.join()
    return results


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_mbpp(split: str, subset: int | None) -> list[dict]:
    """
    Load MBPP sanitized split from HuggingFace datasets.
    Falls back to the legacy 'mbpp' dataset if the sanitized variant is unavailable.
    """
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError:
        _error("HuggingFace `datasets` is not installed.  Run: pip install datasets")
        sys.exit(1)

    _info(f"Loading MBPP sanitized / {split} ...")
    try:
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split=split)
    except Exception as exc:  # noqa: BLE001
        _warn(f"Could not load sanitized MBPP ({exc}).  Falling back to 'mbpp'.")
        try:
            ds = load_dataset("mbpp", split=split)
        except Exception as exc2:  # noqa: BLE001
            _error(f"Failed to load MBPP: {exc2}")
            sys.exit(1)

    problems = list(ds)
    _success(f"Loaded {len(problems)} problems from split='{split}'.")

    if subset is not None and subset < len(problems):
        _info(f"Subsetting to first {subset} problems.")
        problems = problems[:subset]

    return problems


# ---------------------------------------------------------------------------
# vLLM helpers (shared pattern with run_humaneval.py)
# ---------------------------------------------------------------------------

def load_model(model_path: str, tensor_parallel_size: int):
    try:
        from vllm import LLM  # noqa: PLC0415
    except ImportError:
        _error("vllm not installed.  Run: pip install vllm>=0.5.0")
        sys.exit(1)

    _info(f"Loading model {model_path}  (tp={tensor_parallel_size}) ...")
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


def build_sampling_params(temperature: float, max_tokens: int):
    try:
        from vllm import SamplingParams  # noqa: PLC0415
    except ImportError:
        _error("vllm not installed.")
        sys.exit(1)

    if temperature == 0.0:
        return SamplingParams(temperature=0.0, max_tokens=max_tokens)
    return SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
    )


def generate_completions(llm, prompts: list[str], sampling_params) -> list[str]:
    _info(f"Generating {len(prompts)} completions ...")
    outputs = llm.generate(prompts, sampling_params)
    return [o.outputs[0].text for o in outputs]


# ---------------------------------------------------------------------------
# Results helpers
# ---------------------------------------------------------------------------

def compute_error_analysis(results: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {
        "syntax_error": 0,
        "runtime_error": 0,
        "wrong_answer": 0,
        "timeout": 0,
        "passed": 0,
    }
    for r in results:
        if r.get("passed"):
            counts["passed"] += 1
        else:
            etype = r.get("error_type") or "runtime_error"
            if etype in counts:
                counts[etype] += 1
            else:
                counts["runtime_error"] += 1
    return counts


def print_results_table(
    pass_at_1: float,
    error_analysis: dict,
    num_problems: int,
    model_path: str,
) -> None:
    if not HAS_RICH:
        print("\n=== MBPP Results ===")
        print(f"  Model    : {model_path}")
        print(f"  pass@1   : {pass_at_1 * 100:.2f}%  ({error_analysis['passed']}/{num_problems})")
        for k, v in error_analysis.items():
            if k != "passed":
                print(f"  {k:16s}: {v}")
        return

    # Summary table
    table = Table(
        title="MBPP Evaluation Results",
        box=rich_box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric", style="bold white", min_width=20)
    table.add_column("Value", style="bold green", justify="right")

    table.add_row("Model", model_path)
    table.add_row("pass@1", f"{pass_at_1 * 100:.2f}%")
    table.add_row(
        "Passed / Total",
        f"{error_analysis['passed']} / {num_problems}",
    )
    table.add_row("", "")  # spacer
    table.add_row("Syntax errors",  str(error_analysis.get("syntax_error", 0)))
    table.add_row("Runtime errors", str(error_analysis.get("runtime_error", 0)))
    table.add_row("Wrong answers",  str(error_analysis.get("wrong_answer", 0)))
    table.add_row("Timeouts",       str(error_analysis.get("timeout", 0)))

    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned model on the MBPP sanitized benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the fine-tuned model (local dir or HF hub id).",
    )
    parser.add_argument(
        "--output-dir",
        default="eval/results/mbpp",
        help="Directory to write results.",
    )
    parser.add_argument(
        "--split",
        choices=["test", "validation", "train"],
        default="test",
        help="Dataset split to evaluate on.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.  Defaults to 0 (greedy) for pass@1.",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Force greedy decoding (equivalent to --temperature 0).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens per completion.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism.",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=374,
        help="Evaluate on the first N problems (374 = full sanitized test set).",
    )
    parser.add_argument(
        "--n-eval-workers",
        type=int,
        default=8,
        help="Parallel threads for test execution.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    temperature = 0.0 if args.greedy else args.temperature

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    problems = load_mbpp(args.split, args.subset)

    # ------------------------------------------------------------------
    # Build prompts
    # ------------------------------------------------------------------
    prompts = [build_prompt(p) for p in problems]
    _info(f"Built {len(prompts)} prompts.")

    # ------------------------------------------------------------------
    # Load model & generate
    # ------------------------------------------------------------------
    llm = load_model(args.model_path, args.tensor_parallel_size)
    sampling_params = build_sampling_params(temperature, args.max_tokens)

    t0 = time.time()
    raw_completions = generate_completions(llm, prompts, sampling_params)
    elapsed_gen = time.time() - t0
    _success(f"Generation done in {elapsed_gen:.1f}s.")

    # ------------------------------------------------------------------
    # Sanitize completions
    # ------------------------------------------------------------------
    completions = [sanitize_completion(c) for c in raw_completions]

    # ------------------------------------------------------------------
    # Build evaluation items
    # ------------------------------------------------------------------
    eval_items: list[dict] = []
    for prob, code in zip(problems, completions):
        eval_items.append(
            {
                "task_id": str(prob.get("task_id", prob.get("source_file", ""))),
                "text": prob["text"],
                "code": code,
                "tests": prob["test_list"],
            }
        )

    # ------------------------------------------------------------------
    # Execute tests
    # ------------------------------------------------------------------
    _info(f"Executing test cases with {args.n_eval_workers} workers ...")
    t0 = time.time()
    exec_results = evaluate_completions_parallel(eval_items, n_workers=args.n_eval_workers)
    elapsed_eval = time.time() - t0
    _success(f"Evaluation done in {elapsed_eval:.1f}s.")

    # ------------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------------
    num_passed = sum(1 for r in exec_results if r.get("passed", False))
    pass_at_1 = num_passed / len(exec_results) if exec_results else 0.0
    error_analysis = compute_error_analysis(exec_results)

    # ------------------------------------------------------------------
    # Per-problem breakdown
    # ------------------------------------------------------------------
    per_problem: list[dict] = []
    for r in exec_results:
        per_problem.append(
            {
                "task_id": r["task_id"],
                "passed": r.get("passed", False),
                "error_type": r.get("error_type"),
                "error_message": r.get("error_message"),
            }
        )

    # ------------------------------------------------------------------
    # Save samples
    # ------------------------------------------------------------------
    samples_path = output_dir / "samples.jsonl"
    with open(samples_path, "w", encoding="utf-8") as fh:
        for prob, code, res in zip(problems, completions, exec_results):
            fh.write(
                json.dumps(
                    {
                        "task_id": res["task_id"],
                        "prompt": build_prompt(prob),
                        "completion": code,
                        "passed": res.get("passed", False),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    _success(f"Samples saved to {samples_path}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    results: dict[str, Any] = {
        "model_path": args.model_path,
        "timestamp": timestamp,
        "benchmark": "MBPP-sanitized",
        "split": args.split,
        "num_problems": len(problems),
        "temperature": temperature,
        "max_tokens": args.max_tokens,
        "generation_time_seconds": round(elapsed_gen, 2),
        "evaluation_time_seconds": round(elapsed_eval, 2),
        "pass@1": pass_at_1,
        "error_analysis": error_analysis,
        "per_problem_breakdown": per_problem,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    _success(f"Results saved to {results_path}")

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    print_results_table(pass_at_1, error_analysis, len(problems), args.model_path)


if __name__ == "__main__":
    main()
