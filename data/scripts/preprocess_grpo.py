#!/usr/bin/env python3
"""Preprocess problems with test cases into verl GRPO parquet format.

Reads APPS and/or CodeContests JSONL files from data/raw/, builds
chat-format prompts, selects the hardest problems for training and
a diverse subset for validation, and saves the results as parquet
files consumable by the verl GRPO trainer.

Usage:
    python preprocess_grpo.py \\
        --input-dir data/raw \\
        --output-dir data/processed/grpo \\
        --n-train 70 \\
        --n-val 20 \\
        --seed 42
"""

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="DEBUG",
)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE = (
    "You are an expert programmer. Solve the problem step-by-step inside "
    "<think>...</think> tags, then provide your final Python solution in a "
    "code block."
)

USER_MESSAGE_TEMPLATE = """\
Solve the following problem:

{title}

{description}

Your solution should pass all test cases."""

# ---------------------------------------------------------------------------
# Difficulty ordering for "hardest first" selection
# ---------------------------------------------------------------------------

DIFFICULTY_RANK: dict[str, int] = {
    "expert": 6,
    "hard": 5,
    "medium": 4,
    "easy": 3,
    "unknown": 2,
    "introductory": 1,
    "interview": 3,
    "competition": 5,
}


def difficulty_rank(d: str) -> int:
    return DIFFICULTY_RANK.get((d or "unknown").lower(), 2)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning(f"{path}:{lineno} — JSON parse error: {exc}")
    return records


def load_all_problems(input_dir: Path) -> list[dict]:
    """Load all JSONL problem files from input_dir (recursively)."""
    all_records: list[dict] = []
    jsonl_files = sorted(input_dir.rglob("*.jsonl"))
    if not jsonl_files:
        logger.warning(f"No JSONL files found under {input_dir}")
        return all_records
    for path in jsonl_files:
        records = load_jsonl(path)
        logger.info(f"Loaded {len(records):,} records from {path}")
        all_records.extend(records)
    return all_records


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def deduplicate(records: list[dict]) -> list[dict]:
    """Deduplicate by problem_id, keeping the first occurrence."""
    seen: set[str] = set()
    unique: list[dict] = []
    for rec in records:
        pid = rec.get("problem_id", "")
        if not pid or pid in seen:
            continue
        seen.add(pid)
        unique.append(rec)
    logger.info(f"Deduplication: {len(records)} → {len(unique)} unique problems.")
    return unique


# ---------------------------------------------------------------------------
# Problem selection strategy
# ---------------------------------------------------------------------------


def select_problems(
    records: list[dict],
    n_train: int,
    n_val: int,
    rng: random.Random,
) -> tuple[list[dict], list[dict]]:
    """Select train (hardest) and val (diverse) problem subsets.

    Strategy:
    - Sort by difficulty descending for train to maximise challenge.
    - For val, sample proportionally from each difficulty bucket to
      ensure diversity.
    - Ensure train and val are disjoint.
    """
    if len(records) < n_train + n_val:
        logger.warning(
            f"Only {len(records)} problems available, but n_train={n_train} + "
            f"n_val={n_val} = {n_train + n_val} requested. "
            "Adjusting to available data."
        )
        n_total = len(records)
        n_val = min(n_val, max(1, int(n_total * 0.2)))
        n_train = n_total - n_val

    # Sort descending by difficulty rank
    sorted_recs = sorted(
        records,
        key=lambda r: difficulty_rank(r.get("difficulty", "unknown")),
        reverse=True,
    )

    # Train: take the top n_train hardest problems
    train_pool = sorted_recs[:n_train]

    # Val: sample from the remaining records to get diversity
    remaining = sorted_recs[n_train:]
    rng.shuffle(remaining)

    if len(remaining) >= n_val:
        # Stratified sampling from remaining by difficulty
        buckets: dict[str, list[dict]] = {}
        for rec in remaining:
            d = (rec.get("difficulty") or "unknown").lower()
            buckets.setdefault(d, []).append(rec)

        val_pool: list[dict] = []
        bucket_names = list(buckets.keys())
        rng.shuffle(bucket_names)

        # Round-robin fill val_pool from buckets
        while len(val_pool) < n_val and any(buckets[b] for b in bucket_names):
            for b in bucket_names:
                if len(val_pool) >= n_val:
                    break
                if buckets[b]:
                    val_pool.append(buckets[b].pop(0))
    else:
        logger.warning(
            f"Remaining pool ({len(remaining)}) < n_val ({n_val}). "
            "Using all remaining for val."
        )
        val_pool = remaining

    logger.info(
        f"Selected {len(train_pool)} train problems (hardest), "
        f"{len(val_pool)} val problems (diverse)."
    )
    return train_pool, val_pool


# ---------------------------------------------------------------------------
# Build verl-format rows
# ---------------------------------------------------------------------------


def infer_function_signature(description: str, title: str) -> str:
    """Heuristically extract a function signature from the description.

    Returns an empty string if no signature can be found.
    """
    import re

    # Look for "def <name>(" patterns in the problem description
    match = re.search(r"(def\s+\w+\s*\(.*?\)\s*(?:->\s*\w[\w\[\], ]*)?:?)", description)
    if match:
        return match.group(1).strip().rstrip(":")
    # Fallback: derive from title (e.g. "Two Sum" → "def two_sum(...)")
    if title:
        name = re.sub(r"[^a-zA-Z0-9 ]", "", title)
        name = "_".join(name.lower().split())
        return f"def {name}(...):"
    return ""


def build_chat_prompt(problem: dict) -> list[dict]:
    """Build a chat-format prompt list for one problem."""
    title = problem.get("title", "Untitled Problem").strip()
    description = problem.get("description", "").strip()
    user_content = USER_MESSAGE_TEMPLATE.format(
        title=title,
        description=description,
    )
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]


def build_verl_row(problem: dict) -> dict:
    """Convert a raw problem dict into a verl-format dict row."""
    problem_id = problem.get("problem_id", "")
    title = problem.get("title", "Untitled Problem").strip()
    description = problem.get("description", "").strip()
    difficulty = (problem.get("difficulty") or "unknown").lower()
    source = problem.get("source", "unknown")
    test_cases = problem.get("test_cases", [])

    chat_prompt = build_chat_prompt(problem)
    func_sig = infer_function_signature(description, title)

    return {
        "data_source": "leetcode_grpo",
        # verl expects prompt as JSON string of the chat list
        "prompt": json.dumps(chat_prompt, ensure_ascii=False),
        "ability": "code",
        # reward_model is a JSON string consumed by the verl reward function
        "reward_model": json.dumps(
            {
                "style": "rule",
                "ground_truth": {
                    "test_cases": test_cases,
                    "problem_id": problem_id,
                    "function_signature": func_sig,
                },
            },
            ensure_ascii=False,
        ),
        # extra_info is a JSON string for convenience / logging
        "extra_info": json.dumps(
            {
                "problem_id": problem_id,
                "difficulty": difficulty,
                "source": source,
                "title": title,
                "n_test_cases": len(test_cases),
            },
            ensure_ascii=False,
        ),
    }


# ---------------------------------------------------------------------------
# DataFrame construction and parquet I/O
# ---------------------------------------------------------------------------

PARQUET_SCHEMA = pa.schema(
    [
        pa.field("data_source", pa.string()),
        pa.field("prompt", pa.string()),
        pa.field("ability", pa.string()),
        pa.field("reward_model", pa.string()),
        pa.field("extra_info", pa.string()),
    ]
)


def build_dataframe(problems: list[dict]) -> pd.DataFrame:
    rows = [build_verl_row(p) for p in problems]
    return pd.DataFrame(rows)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to parquet using pyarrow with the canonical schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, schema=PARQUET_SCHEMA, preserve_index=False)
    pq.write_table(table, str(path), compression="snappy")
    logger.info(f"Saved {len(df):,} rows → {path}  ({path.stat().st_size / 1024:.1f} KB)")


# ---------------------------------------------------------------------------
# Print schema + sample row
# ---------------------------------------------------------------------------


def print_schema_and_sample(df: pd.DataFrame, split_name: str) -> None:
    """Log the DataFrame schema and a single representative row."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"{split_name.upper()} SPLIT — schema:")
    for col in df.columns:
        dtype = df[col].dtype
        logger.info(f"  {col:<20} {dtype}")

    if len(df) == 0:
        logger.warning("  (empty DataFrame)")
        return

    sample = df.iloc[0].to_dict()
    logger.info(f"\n{split_name.upper()} SPLIT — sample row (index 0):")
    for key, val in sample.items():
        val_str = str(val)
        if len(val_str) > 200:
            val_str = val_str[:200] + "…"
        logger.info(f"  {key:<20} {val_str!r}")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess problems with test cases into verl GRPO parquet format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Root directory containing raw JSONL files (searched recursively).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/grpo"),
        help="Directory where train.parquet and val.parquet are written.",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=70,
        help="Number of (hardest) problems to include in the train split.",
    )
    parser.add_argument(
        "--n-val",
        type=int,
        default=20,
        help="Number of (diverse) problems to include in the val split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffling and selection.",
    )
    parser.add_argument(
        "--min-test-cases",
        type=int,
        default=2,
        help="Minimum number of test cases required to include a problem.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_dir: Path = args.input_dir.resolve()
    output_dir: Path = args.output_dir.resolve()

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    # 1. Load all raw problems
    all_problems = load_all_problems(input_dir)
    logger.info(f"Total records loaded: {len(all_problems):,}")

    if not all_problems:
        logger.error("No problems loaded. Check --input-dir.")
        sys.exit(1)

    # 2. Filter: must have at least --min-test-cases test cases
    before_filter = len(all_problems)
    all_problems = [
        p for p in all_problems
        if len(p.get("test_cases", [])) >= args.min_test_cases
    ]
    logger.info(
        f"After test-case filter (min={args.min_test_cases}): "
        f"{len(all_problems):,} / {before_filter:,} problems remain."
    )

    # 3. Deduplicate by problem_id
    all_problems = deduplicate(all_problems)

    # 4. Shuffle with seed (before selection)
    rng = random.Random(args.seed)
    rng.shuffle(all_problems)

    # 5. Select train and val subsets
    train_problems, val_problems = select_problems(
        all_problems,
        n_train=args.n_train,
        n_val=args.n_val,
        rng=rng,
    )

    # 6. Build DataFrames
    train_df = build_dataframe(train_problems)
    val_df = build_dataframe(val_problems)

    # 7. Save as parquet
    output_dir.mkdir(parents=True, exist_ok=True)
    save_parquet(train_df, output_dir / "train.parquet")
    save_parquet(val_df, output_dir / "val.parquet")

    # 8. Print schema and sample
    print_schema_and_sample(train_df, "train")
    print_schema_and_sample(val_df, "val")

    logger.info("GRPO preprocessing complete.")
    logger.info(f"  Train: {output_dir / 'train.parquet'}")
    logger.info(f"  Val:   {output_dir / 'val.parquet'}")


if __name__ == "__main__":
    main()
