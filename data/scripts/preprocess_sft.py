#!/usr/bin/env python3
"""Preprocess raw CoT data into LLaMA-Factory SFT format.

Reads the generated_cot.jsonl file produced by generate_sft_data.py,
validates each entry, builds instruction/output pairs, performs a
stratified train/val split, and writes LLaMA-Factory–compatible JSON
files plus a dataset_info.json manifest.

Usage:
    python preprocess_sft.py \\
        --input data/raw/generated_cot.jsonl \\
        --output-dir data/processed/sft \\
        --val-ratio 0.05 \\
        --seed 42
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

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
# Validation patterns
# ---------------------------------------------------------------------------

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
CODE_BLOCK_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL)

# ---------------------------------------------------------------------------
# Instruction template
# ---------------------------------------------------------------------------

INSTRUCTION_TEMPLATE = """\
Solve the following coding problem. Think step-by-step inside <think>...</think> tags, then provide your final Python solution.

**Problem:** {title}

{description}"""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_entry(entry: dict) -> tuple[bool, str]:
    """Validate a generated_cot.jsonl record.

    Returns (is_valid, reason).
    """
    response = entry.get("response", "")
    if not response:
        return False, "empty response"

    think_match = THINK_RE.search(response)
    if not think_match:
        return False, "no <think>...</think> tags found"
    think_content = think_match.group(1).strip()
    if len(think_content) < 30:
        return False, f"<think> block too short ({len(think_content)} chars)"

    code_match = CODE_BLOCK_RE.search(response)
    if not code_match:
        return False, "no ```python code block found"
    code_content = code_match.group(1).strip()
    if len(code_content) < 10:
        return False, f"python code block too short ({len(code_content)} chars)"

    prompt = entry.get("prompt", "")
    if not prompt:
        return False, "empty prompt"

    return True, "ok"


def extract_title_from_prompt(prompt: str) -> str:
    """Try to extract the problem title from a generated prompt string."""
    # The generate_sft_data.py template wraps the title with **...**
    match = re.search(r"\*\*(.+?)\*\*", prompt)
    if match:
        return match.group(1).strip()
    # Fallback: first non-empty line
    for line in prompt.splitlines():
        line = line.strip()
        if line:
            return line[:120]
    return "Untitled"


# ---------------------------------------------------------------------------
# Build LLaMA-Factory record
# ---------------------------------------------------------------------------


def build_llama_factory_record(entry: dict) -> dict:
    """Convert a generated_cot record to a LLaMA-Factory {instruction, output} dict."""
    problem_id: str = entry.get("problem_id", "")
    prompt: str = entry.get("prompt", "")
    response: str = entry.get("response", "")

    # The prompt from generate_sft_data.py already includes the title and
    # description formatted via USER_PROMPT_TEMPLATE.  Re-wrap it using
    # the canonical SFT instruction template so the model sees a consistent
    # format.  We parse title + description out of the existing prompt.
    title = extract_title_from_prompt(prompt)

    # Strip the bold title line and leading/trailing whitespace to get
    # a clean description for the instruction.
    description = re.sub(r"\*\*.+?\*\*\s*\n?", "", prompt, count=1).strip()

    instruction = INSTRUCTION_TEMPLATE.format(
        title=title,
        description=description,
    )

    return {
        "instruction": instruction,
        "output": response.strip(),
        # Extra metadata kept for debugging; LLaMA-Factory ignores unknown keys
        "_problem_id": problem_id,
        "_difficulty": entry.get("difficulty", "unknown"),
    }


# ---------------------------------------------------------------------------
# Stratified train/val split
# ---------------------------------------------------------------------------


def stratified_split(
    records: list[dict],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Split records into train/val, stratified by _difficulty if present.

    Falls back to a random split if difficulty info is missing.
    """
    rng = random.Random(seed)

    # Group by difficulty
    buckets: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        difficulty = rec.get("_difficulty", "unknown") or "unknown"
        buckets[difficulty].append(rec)

    train_records: list[dict] = []
    val_records: list[dict] = []

    for difficulty, bucket in buckets.items():
        rng.shuffle(bucket)
        n_val = max(1, round(len(bucket) * val_ratio))
        val_records.extend(bucket[:n_val])
        train_records.extend(bucket[n_val:])
        logger.debug(
            f"  Difficulty '{difficulty}': {len(bucket)} total → "
            f"{len(bucket) - n_val} train / {n_val} val"
        )

    # Shuffle final splits
    rng.shuffle(train_records)
    rng.shuffle(val_records)
    return train_records, val_records


# ---------------------------------------------------------------------------
# Strip internal metadata before saving
# ---------------------------------------------------------------------------


def strip_metadata(records: list[dict]) -> list[dict]:
    """Remove keys prefixed with '_' before writing to disk."""
    clean = []
    for rec in records:
        clean.append({k: v for k, v in rec.items() if not k.startswith("_")})
    return clean


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def compute_statistics(
    total_loaded: int,
    total_valid: int,
    train_records: list[dict],
    val_records: list[dict],
) -> None:
    """Print dataset statistics to the logger."""
    all_records = train_records + val_records

    output_lengths = [len(r.get("output", "")) for r in all_records]
    think_lengths = []
    code_lengths = []
    for r in all_records:
        resp = r.get("output", "")
        tm = THINK_RE.search(resp)
        cm = CODE_BLOCK_RE.search(resp)
        if tm:
            think_lengths.append(len(tm.group(1).strip()))
        if cm:
            code_lengths.append(len(cm.group(1).strip()))

    pct_valid = 100 * total_valid / total_loaded if total_loaded else 0

    logger.info("=" * 60)
    logger.info("Dataset Statistics")
    logger.info(f"  Total loaded:        {total_loaded:>8,}")
    logger.info(f"  Valid format:        {total_valid:>8,}  ({pct_valid:.1f}%)")
    logger.info(f"  Train examples:      {len(train_records):>8,}")
    logger.info(f"  Val examples:        {len(val_records):>8,}")
    if output_lengths:
        avg_out = sum(output_lengths) / len(output_lengths)
        max_out = max(output_lengths)
        logger.info(f"  Avg response len:    {avg_out:>8,.0f} chars")
        logger.info(f"  Max response len:    {max_out:>8,} chars")
    if think_lengths:
        avg_think = sum(think_lengths) / len(think_lengths)
        logger.info(f"  Avg <think> len:     {avg_think:>8,.0f} chars")
    if code_lengths:
        avg_code = sum(code_lengths) / len(code_lengths)
        logger.info(f"  Avg code block len:  {avg_code:>8,.0f} chars")
    pct_fmt = 100 * len(all_records) / total_loaded if total_loaded else 0
    logger.info(f"  % with valid format: {pct_fmt:>7.1f}%")
    logger.info("=" * 60)


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
                logger.warning(f"Line {lineno}: JSON parse error — {exc}")
    return records


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    logger.info(f"Wrote {len(data) if isinstance(data, list) else 1} records → {path}")


# ---------------------------------------------------------------------------
# dataset_info.json
# ---------------------------------------------------------------------------


def build_dataset_info(output_dir: Path) -> dict:
    """Build the dataset_info.json manifest for LLaMA-Factory."""
    return {
        "leetcode_cot": {
            "file_name": str((output_dir / "train.json").resolve()),
            "columns": {
                "prompt": "instruction",
                "response": "output",
            },
        },
        "leetcode_cot_val": {
            "file_name": str((output_dir / "val.json").resolve()),
            "columns": {
                "prompt": "instruction",
                "response": "output",
            },
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess raw CoT JSONL data into LLaMA-Factory SFT format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/generated_cot.jsonl"),
        help="Path to generated_cot.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/sft"),
        help="Directory where train.json, val.json, and dataset_info.json are written.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Fraction of data to use for validation (stratified by difficulty).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits.",
    )
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help=(
            "Include entries that fail format validation. "
            "By default they are dropped."
        ),
    )
    parser.add_argument(
        "--min-think-chars",
        type=int,
        default=30,
        help="Minimum number of characters required in the <think> block.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # 1. Load
    logger.info(f"Loading {args.input} …")
    raw_entries = load_jsonl(args.input)
    total_loaded = len(raw_entries)
    logger.info(f"Loaded {total_loaded:,} entries.")

    # 2. Validate
    valid_entries: list[dict] = []
    invalid_count = 0
    for entry in raw_entries:
        ok, reason = validate_entry(entry)
        if not ok:
            invalid_count += 1
            logger.debug(
                f"  [{entry.get('problem_id', '?')}] INVALID: {reason}"
            )
            if args.include_invalid:
                valid_entries.append(entry)
        else:
            valid_entries.append(entry)

    total_valid = total_loaded - invalid_count
    logger.info(
        f"Validation: {total_valid}/{total_loaded} valid "
        f"({invalid_count} dropped{' (--include-invalid overrides)' if args.include_invalid else ''})."
    )

    if not valid_entries:
        logger.error("No valid entries remain after validation. Aborting.")
        sys.exit(1)

    # 3. Build LLaMA-Factory records
    lf_records: list[dict] = []
    for entry in valid_entries:
        rec = build_llama_factory_record(entry)
        lf_records.append(rec)

    # 4. Stratified train/val split
    logger.info(f"Splitting {len(lf_records)} records (val_ratio={args.val_ratio}, seed={args.seed}) …")
    train_records, val_records = stratified_split(lf_records, args.val_ratio, args.seed)

    # 5. Write files
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    write_json(strip_metadata(train_records), output_dir / "train.json")
    write_json(strip_metadata(val_records), output_dir / "val.json")

    # 6. dataset_info.json
    dataset_info = build_dataset_info(output_dir)
    write_json(dataset_info, output_dir / "dataset_info.json")

    # 7. Print statistics
    compute_statistics(total_loaded, total_valid, train_records, val_records)


if __name__ == "__main__":
    main()
