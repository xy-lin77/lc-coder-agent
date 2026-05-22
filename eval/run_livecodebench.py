#!/usr/bin/env python3
"""
LiveCodeBench evaluation.
Uses time-stamped problems to prevent data contamination.
Evaluates on problems released after a given date.

Usage:
    python eval/run_livecodebench.py --model-path checkpoints/grpo-final
    python eval/run_livecodebench.py --model-path checkpoints/grpo-final \\
        --start-date 2024-06-01 --end-date 2025-01-01 --greedy
    python eval/run_livecodebench.py --model-path checkpoints/sft \\
        --start-date 2024-01-01 --max-problems 200
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
from datetime import datetime, date
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Rich helpers
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
# Date utilities
# ---------------------------------------------------------------------------

def parse_date(s: str) -> date:
    """Parse a YYYY-MM-DD string into a date object."""
    return datetime.strptime(s, "%Y-%m-%d").date()


def normalize_release_date(raw: Any) -> date | None:
    """
    LCB release_date may be a string "YYYY-MM-DD", a datetime object, or None.
    Returns a date or None if unparseable.
    """
    if raw is None:
        return None
    if isinstance(raw, (datetime,)):
        return raw.date()
    if isinstance(raw, date):
        return raw
    if isinstance(raw, str):
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
            try:
                return datetime.strptime(raw, fmt).date()
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# Problem loading
# ---------------------------------------------------------------------------

def load_livecodebench_native(scenario: str) -> list[dict] | None:
    """
    Attempt to load via the official `livecodebench` package.
    Returns a list of problem dicts or None if the package is unavailable.
    """
    try:
        import livecodebench as lcb  # noqa: PLC0415

        scenario_map = {
            "code_generation": lcb.Scenario.codegeneration,
            "code_generation_lite": lcb.Scenario.codegeneration,
            "self_repair": lcb.Scenario.selfrepair,
            "test_output_prediction": lcb.Scenario.testoutputprediction,
        }
        s = scenario_map.get(scenario, lcb.Scenario.codegeneration)
        problems = lcb.load_problems(s)
        return [p.__dict__ if hasattr(p, "__dict__") else dict(p) for p in problems]
    except Exception:  # noqa: BLE001
        return None


def load_livecodebench_hf(scenario: str) -> list[dict]:
    """
    Load LiveCodeBench from HuggingFace datasets.
    Primary: livecodebench/code_generation_lite
    Fallback: livecodebench/code_generation
    """
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError:
        _error("HuggingFace `datasets` not installed.  Run: pip install datasets")
        sys.exit(1)

    dataset_ids = [
        "livecodebench/code_generation_lite",
        "livecodebench/code_generation",
    ]
    if "lite" not in scenario:
        dataset_ids = list(reversed(dataset_ids))

    for ds_id in dataset_ids:
        try:
            _info(f"Trying dataset: {ds_id} ...")
            ds = load_dataset(ds_id, split="test")
            problems = list(ds)
            _success(f"Loaded {len(problems)} problems from {ds_id}.")
            return problems
        except Exception as exc:  # noqa: BLE001
            _warn(f"Could not load {ds_id}: {exc}")

    _error(
        "Could not load LiveCodeBench from HuggingFace.\n"
        "Install the official package with:\n"
        "    pip install livecodebench\n"
        "or ensure you have network access to HuggingFace."
    )
    sys.exit(1)


def load_problems(scenario: str) -> list[dict]:
    """Try native package first, then HF fallback."""
    native = load_livecodebench_native(scenario)
    if native is not None:
        _success(f"Loaded {len(native)} problems via livecodebench package.")
        return native
    _warn("livecodebench package not found; falling back to HuggingFace.")
    return load_livecodebench_hf(scenario)


def filter_by_date(
    problems: list[dict],
    start_date: date | None,
    end_date: date | None,
) -> list[dict]:
    """
    Filter problems by release_date.  Problems with unparseable dates are
    kept when no date filter is active, dropped otherwise.
    """
    if start_date is None and end_date is None:
        return problems

    filtered: list[dict] = []
    skipped_no_date = 0
    for p in problems:
        raw = p.get("release_date") or p.get("contest_date") or p.get("date")
        rd = normalize_release_date(raw)
        if rd is None:
            skipped_no_date += 1
            continue
        if start_date and rd < start_date:
            continue
        if end_date and rd > end_date:
            continue
        filtered.append(p)

    if skipped_no_date:
        _warn(f"Skipped {skipped_no_date} problems with missing/unparseable dates.")
    _success(f"Filtered to {len(filtered)} problems in date range.")
    return filtered


def get_difficulty(problem: dict) -> str:
    """Normalize difficulty field to easy/medium/hard."""
    raw = str(problem.get("difficulty") or problem.get("problem_difficulty") or "").lower()
    if raw in ("easy", "1"):
        return "easy"
    if raw in ("hard", "3"):
        return "hard"
    return "medium"


# ---------------------------------------------------------------------------
# Official prompt template
# ---------------------------------------------------------------------------

_OFFICIAL_PROMPT_TEMPLATE = """\
### Problem

{problem_statement}

### Starter Code

```python
{starter_code}
```

### Instructions

Write a complete Python solution for the problem above.
The function signature is provided in the starter code — do not change it.
Output only the Python code, with no additional explanation.
"""

_MINIMAL_PROMPT_TEMPLATE = """\
You are an expert competitive programmer. Solve the following problem in Python.

{problem_statement}

Write a complete Python solution. Output only the code, no explanation.
"""


def build_prompt(problem: dict) -> str:
    """
    Build an inference prompt from a LCB problem dict.
    Handles heterogeneous field names across dataset versions.
    """
    # Field name variants
    statement = (
        problem.get("question_content")
        or problem.get("problem_statement")
        or problem.get("description")
        or problem.get("content")
        or ""
    ).strip()

    starter = (
        problem.get("starter_code")
        or problem.get("code_prompt")
        or problem.get("template")
        or ""
    ).strip()

    if starter:
        return _OFFICIAL_PROMPT_TEMPLATE.format(
            problem_statement=statement,
            starter_code=starter,
        )
    return _MINIMAL_PROMPT_TEMPLATE.format(problem_statement=statement)


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    match = _FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


# ---------------------------------------------------------------------------
# Test case execution (lightweight fallback evaluator)
# ---------------------------------------------------------------------------

_EXEC_TIMEOUT = 15  # seconds — LCB problems can be more complex


def _extract_test_cases(problem: dict) -> list[str]:
    """
    Extract runnable test assertions from a LCB problem.
    Handles multiple field conventions.
    """
    tests: list[str] = []

    # Official package format: a JSON string or list of input/output pairs
    raw_tests = (
        problem.get("public_tests")
        or problem.get("test_cases")
        or problem.get("tests")
        or []
    )

    if isinstance(raw_tests, str):
        try:
            raw_tests = json.loads(raw_tests)
        except json.JSONDecodeError:
            raw_tests = []

    if isinstance(raw_tests, list):
        for tc in raw_tests:
            if isinstance(tc, dict):
                inp = tc.get("input", "")
                out = tc.get("output", "") or tc.get("expected_output", "")
                if inp is not None and out is not None:
                    # Build an assertion string
                    tests.append(
                        f"# input={inp!r}  expected={out!r}"
                    )
            elif isinstance(tc, str):
                tests.append(tc)

    return tests


def _run_with_io_redirection(
    code: str,
    test_input: str,
    expected_output: str,
    timeout: int = _EXEC_TIMEOUT,
) -> dict[str, Any]:
    """
    Run code with stdin=test_input and compare stdout to expected_output.
    Used when tests are I/O pairs rather than assertion strings.
    """
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return {"passed": False, "error_type": "syntax_error", "error_message": str(exc)}

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        actual = proc.stdout.strip()
        expected = expected_output.strip()

        # Normalize: compare line by line, ignore trailing whitespace
        actual_lines = [l.rstrip() for l in actual.splitlines()]
        expected_lines = [l.rstrip() for l in expected.splitlines()]

        if actual_lines == expected_lines:
            return {"passed": True, "error_type": None, "error_message": None}
        return {
            "passed": False,
            "error_type": "wrong_answer",
            "error_message": f"Expected:\n{expected}\nGot:\n{actual}",
        }
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "error_type": "timeout",
            "error_message": f"Exceeded {timeout}s.",
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def evaluate_problem(problem: dict, code: str) -> dict[str, Any]:
    """
    Evaluate a single completion against a LCB problem.

    Strategy:
    1. Try the official livecodebench evaluator if available.
    2. Fall back to I/O-based subprocess execution.
    3. If no test cases are parseable, mark as "unknown".
    """
    # --- Attempt 1: official evaluator ---
    try:
        import livecodebench as lcb  # noqa: PLC0415
        if hasattr(lcb, "evaluate_solution"):
            result = lcb.evaluate_solution(problem, code)
            return {
                "passed": bool(result.get("passed", False)),
                "error_type": result.get("error_type"),
                "error_message": result.get("error_message"),
            }
    except Exception:  # noqa: BLE001
        pass

    # --- Attempt 2: I/O-based execution ---
    raw_tests = (
        problem.get("public_tests")
        or problem.get("test_cases")
        or problem.get("tests")
        or []
    )
    if isinstance(raw_tests, str):
        try:
            raw_tests = json.loads(raw_tests)
        except json.JSONDecodeError:
            raw_tests = []

    if not raw_tests:
        return {
            "passed": False,
            "error_type": "no_tests",
            "error_message": "No test cases available for this problem.",
        }

    # Quick syntax check
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return {"passed": False, "error_type": "syntax_error", "error_message": str(exc)}

    # Run each test case; require ALL to pass
    for tc in raw_tests:
        if not isinstance(tc, dict):
            continue
        inp = tc.get("input", "")
        out = tc.get("output") or tc.get("expected_output") or ""
        if inp is None:
            inp = ""
        if out is None:
            out = ""
        result = _run_with_io_redirection(code, str(inp), str(out))
        if not result["passed"]:
            return result

    return {"passed": True, "error_type": None, "error_message": None}


def evaluate_all_parallel(
    items: list[dict],
    n_workers: int = 8,
) -> list[dict]:
    """
    Evaluate all items in parallel using a thread pool.
    Each item must have keys: problem, code.
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
            eval_result = evaluate_problem(item["problem"], item["code"])
            results[idx] = {**item, **eval_result}
            work_q.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(n_workers)]
    for t in threads:
        t.start()
    work_q.join()
    return results


# ---------------------------------------------------------------------------
# vLLM helpers
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
        max_model_len=8192,
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
    return SamplingParams(temperature=temperature, top_p=0.95, max_tokens=max_tokens)


def generate_completions(llm, prompts: list[str], sampling_params) -> list[str]:
    _info(f"Generating {len(prompts)} completions ...")
    outputs = llm.generate(prompts, sampling_params)
    return [o.outputs[0].text for o in outputs]


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

def compute_difficulty_breakdown(results: list[dict]) -> dict[str, dict]:
    breakdown: dict[str, dict] = {
        "easy": {"passed": 0, "total": 0},
        "medium": {"passed": 0, "total": 0},
        "hard": {"passed": 0, "total": 0},
    }
    for r in results:
        diff = r.get("difficulty", "medium")
        if diff not in breakdown:
            diff = "medium"
        breakdown[diff]["total"] += 1
        if r.get("passed", False):
            breakdown[diff]["passed"] += 1
    # Add pass_rate
    for diff, counts in breakdown.items():
        t = counts["total"]
        counts["pass_rate"] = counts["passed"] / t if t > 0 else 0.0
    return breakdown


def print_results_table(
    pass_at_1: float,
    difficulty_breakdown: dict,
    num_problems: int,
    model_path: str,
    date_range: str,
) -> None:
    if not HAS_RICH:
        print("\n=== LiveCodeBench Results ===")
        print(f"  Model      : {model_path}")
        print(f"  Date range : {date_range}")
        print(f"  pass@1     : {pass_at_1 * 100:.2f}%")
        for diff, counts in difficulty_breakdown.items():
            if counts["total"] > 0:
                print(
                    f"  {diff:8s}   : {counts['pass_rate'] * 100:.1f}%"
                    f"  ({counts['passed']}/{counts['total']})"
                )
        return

    table = Table(
        title="LiveCodeBench Evaluation Results",
        box=rich_box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Category", style="bold white", min_width=22)
    table.add_column("pass@1", style="bold green", justify="right")
    table.add_column("Passed / Total", style="dim", justify="right")

    num_passed = sum(1 for d in difficulty_breakdown.values() for _ in range(d["passed"]))
    table.add_row(
        f"Overall  [{date_range}]",
        f"{pass_at_1 * 100:.2f}%",
        f"{num_passed} / {num_problems}",
    )
    for diff in ("easy", "medium", "hard"):
        counts = difficulty_breakdown.get(diff, {"passed": 0, "total": 0, "pass_rate": 0.0})
        if counts["total"] == 0:
            continue
        table.add_row(
            f"  {diff.capitalize()}",
            f"{counts['pass_rate'] * 100:.1f}%",
            f"{counts['passed']} / {counts['total']}",
        )

    table.add_row("Model", model_path, "")
    console.print(table)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned model on LiveCodeBench.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the fine-tuned model (local dir or HF hub id).",
    )
    parser.add_argument(
        "--output-dir",
        default="eval/results/livecodebench",
        help="Directory to write results.",
    )
    parser.add_argument(
        "--start-date",
        default="2024-01-01",
        help="Include problems released on or after this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Include problems released on or before this date (YYYY-MM-DD). "
             "Defaults to today.",
    )
    parser.add_argument(
        "--scenario",
        default="code_generation",
        choices=[
            "code_generation",
            "code_generation_lite",
            "self_repair",
            "test_output_prediction",
        ],
        help="LCB scenario to evaluate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Force greedy decoding (temperature=0).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens per completion.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for vLLM tensor parallelism.",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Cap the number of problems (for debugging / quick runs).",
    )
    parser.add_argument(
        "--n-eval-workers",
        type=int,
        default=8,
        help="Parallel threads for test execution.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    temperature = 0.0 if args.greedy else args.temperature

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Date range
    # ------------------------------------------------------------------
    start_date = parse_date(args.start_date) if args.start_date else None
    end_date = parse_date(args.end_date) if args.end_date else date.today()
    date_range = f"{start_date} → {end_date}"
    _info(f"Date filter: {date_range}")

    # ------------------------------------------------------------------
    # Load & filter problems
    # ------------------------------------------------------------------
    all_problems = load_problems(args.scenario)
    problems = filter_by_date(all_problems, start_date, end_date)

    if not problems:
        _error(
            "No problems remain after date filtering.  "
            "Try a wider --start-date / --end-date range."
        )
        sys.exit(1)

    if args.max_problems and args.max_problems < len(problems):
        _info(f"Capping to {args.max_problems} problems (--max-problems).")
        problems = problems[: args.max_problems]

    _info(f"Evaluating {len(problems)} problems.")

    # ------------------------------------------------------------------
    # Annotate difficulty
    # ------------------------------------------------------------------
    for p in problems:
        p["_difficulty_normalized"] = get_difficulty(p)

    # ------------------------------------------------------------------
    # Build prompts
    # ------------------------------------------------------------------
    prompts = [build_prompt(p) for p in problems]

    # ------------------------------------------------------------------
    # Load model & generate
    # ------------------------------------------------------------------
    llm = load_model(args.model_path, args.tensor_parallel_size)
    sampling_params = build_sampling_params(temperature, args.max_tokens)

    t0 = time.time()
    raw_completions = generate_completions(llm, prompts, sampling_params)
    elapsed_gen = time.time() - t0
    _success(f"Generation done in {elapsed_gen:.1f}s.")

    completions = [extract_code(c) for c in raw_completions]

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    eval_items = [
        {
            "task_id": str(
                p.get("question_id")
                or p.get("problem_id")
                or p.get("id")
                or i
            ),
            "difficulty": p["_difficulty_normalized"],
            "problem": p,
            "code": code,
        }
        for i, (p, code) in enumerate(zip(problems, completions))
    ]

    _info(f"Executing test cases with {args.n_eval_workers} workers ...")
    t0 = time.time()
    exec_results = evaluate_all_parallel(eval_items, n_workers=args.n_eval_workers)
    elapsed_eval = time.time() - t0
    _success(f"Evaluation done in {elapsed_eval:.1f}s.")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    num_passed = sum(1 for r in exec_results if r.get("passed", False))
    pass_at_1 = num_passed / len(exec_results) if exec_results else 0.0
    difficulty_breakdown = compute_difficulty_breakdown(exec_results)

    # ------------------------------------------------------------------
    # Save samples
    # ------------------------------------------------------------------
    samples_path = output_dir / "samples.jsonl"
    with open(samples_path, "w", encoding="utf-8") as fh:
        for r, raw in zip(exec_results, raw_completions):
            fh.write(
                json.dumps(
                    {
                        "task_id": r["task_id"],
                        "difficulty": r.get("difficulty", "medium"),
                        "completion": r["code"],
                        "passed": r.get("passed", False),
                        "error_type": r.get("error_type"),
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
    per_problem: list[dict] = [
        {
            "task_id": r["task_id"],
            "difficulty": r.get("difficulty", "medium"),
            "passed": r.get("passed", False),
            "error_type": r.get("error_type"),
        }
        for r in exec_results
    ]

    results: dict[str, Any] = {
        "model_path": args.model_path,
        "timestamp": timestamp,
        "benchmark": "LiveCodeBench",
        "scenario": args.scenario,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "num_problems": len(problems),
        "temperature": temperature,
        "max_tokens": args.max_tokens,
        "generation_time_seconds": round(elapsed_gen, 2),
        "evaluation_time_seconds": round(elapsed_eval, 2),
        "pass@1": pass_at_1,
        "difficulty_breakdown": {
            d: {
                "pass@1": v["pass_rate"],
                "passed": v["passed"],
                "total": v["total"],
            }
            for d, v in difficulty_breakdown.items()
        },
        "per_problem_breakdown": per_problem,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    _success(f"Results saved to {results_path}")

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    print_results_table(
        pass_at_1,
        difficulty_breakdown,
        len(problems),
        args.model_path,
        date_range,
    )


if __name__ == "__main__":
    main()
