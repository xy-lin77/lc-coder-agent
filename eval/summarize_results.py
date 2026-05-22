#!/usr/bin/env python3
"""
Aggregate and compare evaluation results across benchmarks and model checkpoints.
Generates a comparison table: base model vs SFT vs GRPO.

Usage:
    python eval/summarize_results.py
    python eval/summarize_results.py --results-dir eval/results --models base sft grpo
    python eval/summarize_results.py --results-dir eval/results --output eval/results/summary
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Rich helpers
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box as rich_box
    from rich.text import Text

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
# Data structures
# ---------------------------------------------------------------------------

# Each slot is either a float (0-1) or None (not evaluated)
class BenchmarkRow:
    """Holds the evaluation metrics for one model checkpoint."""

    def __init__(self, label: str, model_path: str | None = None) -> None:
        self.label = label
        self.model_path: str | None = model_path
        self.humaneval_pass1: float | None = None
        self.humaneval_pass10: float | None = None
        self.humaneval_pass5: float | None = None
        self.mbpp_pass1: float | None = None
        self.lcb_pass1: float | None = None
        self.lcb_easy: float | None = None
        self.lcb_medium: float | None = None
        self.lcb_hard: float | None = None
        self.timestamps: dict[str, str] = {}

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "model_path": self.model_path,
            "humaneval": {
                "pass@1": self.humaneval_pass1,
                "pass@5": self.humaneval_pass5,
                "pass@10": self.humaneval_pass10,
            },
            "mbpp": {
                "pass@1": self.mbpp_pass1,
            },
            "livecodebench": {
                "pass@1": self.lcb_pass1,
                "easy": self.lcb_easy,
                "medium": self.lcb_medium,
                "hard": self.lcb_hard,
            },
            "timestamps": self.timestamps,
        }


# ---------------------------------------------------------------------------
# Results file discovery
# ---------------------------------------------------------------------------

def _find_results_files(results_dir: Path) -> dict[str, list[Path]]:
    """
    Scan results_dir for benchmark result files.

    Expected layout:
        results_dir/
            humaneval/<model_label>/results.json   (preferred)
            humaneval/results.json                 (flat layout fallback)
            mbpp/<model_label>/results.json
            mbpp/results.json
            livecodebench/<model_label>/results.json
            livecodebench/results.json

    Returns a dict:
        {
            "humaneval": [path, ...],
            "mbpp":       [path, ...],
            "livecodebench": [path, ...],
        }
    """
    bench_dirs = {
        "humaneval": ["humaneval", "human_eval", "humaneval"],
        "mbpp": ["mbpp"],
        "livecodebench": ["livecodebench", "livecode_bench", "lcb"],
    }
    found: dict[str, list[Path]] = {"humaneval": [], "mbpp": [], "livecodebench": []}

    for bench, dir_names in bench_dirs.items():
        for dir_name in dir_names:
            bench_root = results_dir / dir_name
            if not bench_root.is_dir():
                continue
            # Recursive glob: pick up nested model dirs and flat layout
            for p in bench_root.rglob("results.json"):
                if p not in found[bench]:
                    found[bench].append(p)

    return found


def _infer_label(path: Path, explicit_models: list[str] | None) -> str:
    """
    Derive a human-readable model label from the results.json path.

    Priority:
    1. Match against explicit --models labels (substring match on parent dirs).
    2. Use the immediate parent directory name if it is not a benchmark keyword.
    3. Fall back to the model_path field inside results.json.
    """
    benchmark_keywords = {"humaneval", "mbpp", "livecodebench", "lcb", "results"}
    parts = [p.lower() for p in path.parts]

    # Check if any explicit model label appears in the path
    if explicit_models:
        for label in explicit_models:
            if label.lower() in parts:
                return label

    # Walk from the file upward, skip benchmark-keyword directories
    for part in reversed(path.parent.parts):
        if part.lower() not in benchmark_keywords:
            return part

    # Last resort: read model_path from the file
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        mp = data.get("model_path", "")
        return Path(mp).name if mp else path.parent.name
    except Exception:  # noqa: BLE001
        return path.parent.name


# ---------------------------------------------------------------------------
# Results loading
# ---------------------------------------------------------------------------

def load_humaneval(path: Path) -> tuple[str | None, dict]:
    """Return (model_path, metric_dict) for a HumanEval results.json."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        _warn(f"Could not read {path}: {exc}")
        return None, {}

    metrics = {
        "pass@1": data.get("pass@1"),
        "pass@5": data.get("pass@5"),
        "pass@10": data.get("pass@10"),
        "timestamp": data.get("timestamp", ""),
        "model_path": data.get("model_path", ""),
    }
    return data.get("model_path"), metrics


def load_mbpp(path: Path) -> tuple[str | None, dict]:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        _warn(f"Could not read {path}: {exc}")
        return None, {}

    metrics = {
        "pass@1": data.get("pass@1"),
        "timestamp": data.get("timestamp", ""),
        "model_path": data.get("model_path", ""),
    }
    return data.get("model_path"), metrics


def load_livecodebench(path: Path) -> tuple[str | None, dict]:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        _warn(f"Could not read {path}: {exc}")
        return None, {}

    db = data.get("difficulty_breakdown", {})
    metrics = {
        "pass@1": data.get("pass@1"),
        "easy": db.get("easy", {}).get("pass@1") if isinstance(db.get("easy"), dict) else db.get("easy"),
        "medium": db.get("medium", {}).get("pass@1") if isinstance(db.get("medium"), dict) else db.get("medium"),
        "hard": db.get("hard", {}).get("pass@1") if isinstance(db.get("hard"), dict) else db.get("hard"),
        "timestamp": data.get("timestamp", ""),
        "model_path": data.get("model_path", ""),
        "date_range": f"{data.get('start_date', '')} → {data.get('end_date', '')}",
    }
    return data.get("model_path"), metrics


# ---------------------------------------------------------------------------
# Table building
# ---------------------------------------------------------------------------

def _pct(value: float | None, baseline: float | None = None) -> str:
    """
    Format a metric value as a percentage string.
    If baseline is provided, append a delta annotation.
    """
    if value is None:
        return "N/A"
    pct = f"{value * 100:.1f}%"
    if baseline is not None and value is not None:
        delta = (value - baseline) * 100
        sign = "+" if delta >= 0 else ""
        pct += f" ({sign}{delta:.1f})"
    return pct


def build_rich_table(
    rows: list[BenchmarkRow],
    baseline_label: str | None,
) -> Table | None:
    if not HAS_RICH:
        return None

    # Find baseline row
    baseline: BenchmarkRow | None = None
    if baseline_label:
        for r in rows:
            if r.label.lower() == baseline_label.lower():
                baseline = r
                break
    if baseline is None and rows:
        baseline = rows[0]

    table = Table(
        title="Benchmark Comparison — Qwen2.5-7B-Instruct",
        box=rich_box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        show_lines=True,
    )
    table.add_column("Model", style="bold white", min_width=18)
    table.add_column("HE pass@1",  justify="right", min_width=12)
    table.add_column("HE pass@10", justify="right", min_width=12)
    table.add_column("MBPP pass@1", justify="right", min_width=12)
    table.add_column("LCB pass@1", justify="right", min_width=12)
    table.add_column("LCB easy",   justify="right", min_width=10)
    table.add_column("LCB medium", justify="right", min_width=10)
    table.add_column("LCB hard",   justify="right", min_width=10)

    for row in rows:
        is_base = baseline is not None and row.label == baseline.label
        bl = baseline if not is_base else None  # don't delta-compare to self

        table.add_row(
            Text(row.label, style="bold yellow" if is_base else ""),
            _pct(row.humaneval_pass1,  bl.humaneval_pass1  if bl else None),
            _pct(row.humaneval_pass10, bl.humaneval_pass10 if bl else None),
            _pct(row.mbpp_pass1,       bl.mbpp_pass1       if bl else None),
            _pct(row.lcb_pass1,        bl.lcb_pass1        if bl else None),
            _pct(row.lcb_easy,         bl.lcb_easy         if bl else None),
            _pct(row.lcb_medium,       bl.lcb_medium       if bl else None),
            _pct(row.lcb_hard,         bl.lcb_hard         if bl else None),
        )

    return table


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------

def _md_pct(value: float | None, baseline: float | None = None) -> str:
    if value is None:
        return "N/A"
    pct = f"{value * 100:.1f}%"
    if baseline is not None:
        delta = (value - baseline) * 100
        sign = "+" if delta >= 0 else ""
        pct += f" ({sign}{delta:.1f})"
    return pct


def build_markdown_table(
    rows: list[BenchmarkRow],
    baseline_label: str | None,
) -> str:
    baseline: BenchmarkRow | None = None
    if baseline_label:
        for r in rows:
            if r.label.lower() == baseline_label.lower():
                baseline = r
                break
    if baseline is None and rows:
        baseline = rows[0]

    headers = [
        "Model",
        "HE pass@1",
        "HE pass@10",
        "MBPP pass@1",
        "LCB pass@1",
        "LCB easy",
        "LCB medium",
        "LCB hard",
    ]
    separator = ["---"] * len(headers)

    lines: list[str] = [
        "# Benchmark Summary — Qwen2.5-7B-Instruct Fine-Tuning",
        "",
        f"Generated: {datetime.utcnow().isoformat(timespec='seconds')}Z",
        "",
        "## Results Table",
        "",
        "Delta values in parentheses are relative to the baseline (first row).",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]

    for row in rows:
        is_base = baseline is not None and row.label == baseline.label
        bl = baseline if not is_base else None

        cells = [
            f"**{row.label}**" if is_base else row.label,
            _md_pct(row.humaneval_pass1,  bl.humaneval_pass1  if bl else None),
            _md_pct(row.humaneval_pass10, bl.humaneval_pass10 if bl else None),
            _md_pct(row.mbpp_pass1,       bl.mbpp_pass1       if bl else None),
            _md_pct(row.lcb_pass1,        bl.lcb_pass1        if bl else None),
            _md_pct(row.lcb_easy,         bl.lcb_easy         if bl else None),
            _md_pct(row.lcb_medium,       bl.lcb_medium       if bl else None),
            _md_pct(row.lcb_hard,         bl.lcb_hard         if bl else None),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    # Difficulty breakdown section
    lcb_rows_with_breakdown = [
        r for r in rows
        if any(v is not None for v in [r.lcb_easy, r.lcb_medium, r.lcb_hard])
    ]
    if lcb_rows_with_breakdown:
        lines += [
            "",
            "## LiveCodeBench Difficulty Breakdown",
            "",
            "| Model | Easy | Medium | Hard |",
            "| --- | --- | --- | --- |",
        ]
        for row in lcb_rows_with_breakdown:
            bl = baseline if (baseline and row.label != baseline.label) else None
            lines.append(
                f"| {row.label} "
                f"| {_md_pct(row.lcb_easy, bl.lcb_easy if bl else None)} "
                f"| {_md_pct(row.lcb_medium, bl.lcb_medium if bl else None)} "
                f"| {_md_pct(row.lcb_hard, bl.lcb_hard if bl else None)} |"
            )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Core aggregation logic
# ---------------------------------------------------------------------------

def aggregate(
    results_dir: Path,
    explicit_models: list[str] | None,
) -> list[BenchmarkRow]:
    """
    Scan results_dir, load all benchmark results, and return a list of
    BenchmarkRow objects — one per unique model label.
    """
    found = _find_results_files(results_dir)

    # Collect by label
    rows_by_label: dict[str, BenchmarkRow] = {}

    def _get_row(label: str, model_path: str | None) -> BenchmarkRow:
        if label not in rows_by_label:
            rows_by_label[label] = BenchmarkRow(label, model_path)
        elif model_path and rows_by_label[label].model_path is None:
            rows_by_label[label].model_path = model_path
        return rows_by_label[label]

    # --- HumanEval ---
    for path in found["humaneval"]:
        label = _infer_label(path, explicit_models)
        model_path, metrics = load_humaneval(path)
        if not metrics:
            continue
        row = _get_row(label, model_path)
        row.humaneval_pass1  = metrics.get("pass@1")
        row.humaneval_pass5  = metrics.get("pass@5")
        row.humaneval_pass10 = metrics.get("pass@10")
        if metrics.get("timestamp"):
            row.timestamps["humaneval"] = metrics["timestamp"]

    # --- MBPP ---
    for path in found["mbpp"]:
        label = _infer_label(path, explicit_models)
        model_path, metrics = load_mbpp(path)
        if not metrics:
            continue
        row = _get_row(label, model_path)
        row.mbpp_pass1 = metrics.get("pass@1")
        if metrics.get("timestamp"):
            row.timestamps["mbpp"] = metrics["timestamp"]

    # --- LiveCodeBench ---
    for path in found["livecodebench"]:
        label = _infer_label(path, explicit_models)
        model_path, metrics = load_livecodebench(path)
        if not metrics:
            continue
        row = _get_row(label, model_path)
        row.lcb_pass1  = metrics.get("pass@1")
        row.lcb_easy   = metrics.get("easy")
        row.lcb_medium = metrics.get("medium")
        row.lcb_hard   = metrics.get("hard")
        if metrics.get("timestamp"):
            row.timestamps["livecodebench"] = metrics["timestamp"]

    if not rows_by_label:
        return []

    # ------------------------------------------------------------------
    # Order rows: respect --models order if given, otherwise sort by label
    # ------------------------------------------------------------------
    if explicit_models:
        ordered: list[BenchmarkRow] = []
        for label in explicit_models:
            # Case-insensitive match
            match = next(
                (r for r in rows_by_label.values() if r.label.lower() == label.lower()),
                None,
            )
            if match:
                ordered.append(match)
        # Append any remaining rows not in explicit list
        explicit_lower = {m.lower() for m in explicit_models}
        for r in rows_by_label.values():
            if r.label.lower() not in explicit_lower:
                ordered.append(r)
        return ordered

    # Default: sort alphabetically, but try to put "base" first
    base_priority = ["base", "qwen", "baseline", "sft", "grpo", "rl"]

    def sort_key(r: BenchmarkRow) -> tuple[int, str]:
        label_lower = r.label.lower()
        for i, keyword in enumerate(base_priority):
            if keyword in label_lower:
                return (i, r.label)
        return (len(base_priority), r.label)

    return sorted(rows_by_label.values(), key=sort_key)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate benchmark results and generate a comparison table "
            "across model checkpoints."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        default="eval/results",
        help="Root directory containing benchmark result subdirectories.",
    )
    parser.add_argument(
        "--output",
        default="eval/results/summary",
        help=(
            "Output file stem (without extension).  "
            "Will write <output>.md and <output>.json."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help=(
            "Ordered list of model labels to include.  "
            "Labels are matched against parent directory names of results.json files.  "
            "Example: --models base sft grpo"
        ),
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help=(
            "Label of the baseline model (used for delta computation).  "
            "Defaults to the first row."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)

    if not results_dir.is_dir():
        _error(
            f"Results directory not found: {results_dir}\n"
            "Run the evaluation scripts first to generate results.json files."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    _info(f"Scanning {results_dir} for benchmark results ...")
    rows = aggregate(results_dir, args.models)

    if not rows:
        _warn(
            "No results found.  Make sure results.json files exist under "
            f"{results_dir}/humaneval/, {results_dir}/mbpp/, "
            f"and {results_dir}/livecodebench/"
        )
        # Exit cleanly — don't crash, just report
        sys.exit(0)

    _success(f"Found results for {len(rows)} model(s): {[r.label for r in rows]}")

    # ------------------------------------------------------------------
    # Determine baseline
    # ------------------------------------------------------------------
    baseline_label = args.baseline or (rows[0].label if rows else None)

    # ------------------------------------------------------------------
    # Print rich table
    # ------------------------------------------------------------------
    if HAS_RICH:
        table = build_rich_table(rows, baseline_label)
        if table:
            console.print(table)
    else:
        # Plain-text fallback
        print("\n=== Benchmark Comparison ===")
        header = f"{'Model':<20} {'HE@1':>8} {'HE@10':>8} {'MBPP@1':>8} {'LCB@1':>8} {'LCB-E':>7} {'LCB-M':>7} {'LCB-H':>7}"
        print(header)
        print("-" * len(header))
        for row in rows:
            print(
                f"{row.label:<20} "
                f"{_pct(row.humaneval_pass1):>8} "
                f"{_pct(row.humaneval_pass10):>8} "
                f"{_pct(row.mbpp_pass1):>8} "
                f"{_pct(row.lcb_pass1):>8} "
                f"{_pct(row.lcb_easy):>7} "
                f"{_pct(row.lcb_medium):>7} "
                f"{_pct(row.lcb_hard):>7}"
            )
        print()

    # ------------------------------------------------------------------
    # Write Markdown
    # ------------------------------------------------------------------
    output_stem = Path(args.output)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    md_path = output_stem.with_suffix(".md")
    md_content = build_markdown_table(rows, baseline_label)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md_content)
    _success(f"Markdown summary saved to {md_path}")

    # ------------------------------------------------------------------
    # Write JSON
    # ------------------------------------------------------------------
    json_path = output_stem.with_suffix(".json")
    summary_json: dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "baseline": baseline_label,
        "models": [r.as_dict() for r in rows],
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary_json, fh, indent=2, ensure_ascii=False)
    _success(f"JSON summary saved to {json_path}")


if __name__ == "__main__":
    main()
