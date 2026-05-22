#!/usr/bin/env python3
"""Download raw datasets to data/raw/.

Fetches APPS and CodeContests from HuggingFace, filters for problems with
at least 2 test cases, and saves them as JSONL files.

Usage:
    python download_datasets.py [--output-dir data/raw] [--max-samples 800]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="DEBUG",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def configure_hf_auth() -> None:
    """Set HuggingFace token from environment if available."""
    token = os.environ.get("HF_TOKEN", "")
    if token:
        try:
            from huggingface_hub import login

            login(token=token, add_to_git_credential=False)
            logger.info("Authenticated with HuggingFace using HF_TOKEN.")
        except Exception as exc:
            logger.warning(f"HF auth failed (continuing without auth): {exc}")
    else:
        logger.warning(
            "HF_TOKEN not set. Some datasets may require authentication. "
            "Export HF_TOKEN=<your_token> to enable authenticated access."
        )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(records: list[dict], filepath: Path) -> None:
    """Write a list of dicts to a JSONL file (overwrites)."""
    with open(filepath, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(records):,} records → {filepath}")


# ---------------------------------------------------------------------------
# APPS dataset
# ---------------------------------------------------------------------------

APPS_DIFFICULTY_MAP = {
    "introductory": "easy",
    "interview": "medium",
    "competition": "hard",
}

TARGET_APPS_DIFFICULTIES = {"introductory", "interview"}


def parse_apps_test_cases(raw: Any) -> list[dict[str, str]]:
    """Extract test cases from APPS input/output fields.

    APPS stores inputs as a JSON string of a list, similarly for outputs.
    Returns a list of {input, expected_output} dicts.
    """
    test_cases: list[dict[str, str]] = []
    try:
        inputs_raw = raw.get("inputs") or raw.get("input_output", {})
        # APPS uses a nested JSON structure in `input_output`
        io_data = raw.get("input_output")
        if isinstance(io_data, str):
            try:
                io_data = json.loads(io_data)
            except json.JSONDecodeError:
                return []
        if not isinstance(io_data, dict):
            return []

        inputs = io_data.get("inputs", [])
        outputs = io_data.get("outputs", [])
        if not isinstance(inputs, list) or not isinstance(outputs, list):
            return []

        for inp, out in zip(inputs, outputs):
            inp_str = inp if isinstance(inp, str) else json.dumps(inp)
            out_str = out if isinstance(out, str) else json.dumps(out)
            test_cases.append({"input": inp_str.strip(), "expected_output": out_str.strip()})
    except Exception as exc:
        logger.debug(f"parse_apps_test_cases error: {exc}")
    return test_cases


def download_apps(output_dir: Path, max_samples: int) -> int:
    """Download APPS dataset and save to output_dir/apps/train.jsonl."""
    logger.info(f"Loading APPS dataset (max_samples={max_samples}) …")
    out_path = output_dir / "apps"
    ensure_dir(out_path)

    try:
        ds = load_dataset(
            "codeparrot/apps",
            split="train",
            trust_remote_code=True,
        )
    except Exception as exc:
        logger.error(f"Failed to load APPS: {exc}")
        return 0

    logger.info(f"APPS train split has {len(ds):,} total examples.")

    records: list[dict] = []
    skipped_difficulty = 0
    skipped_test_cases = 0

    for idx, example in enumerate(tqdm(ds, desc="Processing APPS", unit="ex")):
        if len(records) >= max_samples:
            break

        difficulty_raw = example.get("difficulty", "")
        if difficulty_raw not in TARGET_APPS_DIFFICULTIES:
            skipped_difficulty += 1
            continue

        test_cases = parse_apps_test_cases(example)
        if len(test_cases) < 2:
            skipped_test_cases += 1
            continue

        problem_id = f"apps_{example.get('problem_id', idx)}"
        title = (example.get("question", "")[:80] or f"Problem {idx}").split("\n")[0].strip()
        description = example.get("question", "").strip()

        records.append(
            {
                "problem_id": problem_id,
                "title": title,
                "description": description,
                "difficulty": APPS_DIFFICULTY_MAP.get(difficulty_raw, difficulty_raw),
                "test_cases": test_cases,
                "source": "apps",
            }
        )

    logger.info(
        f"APPS — kept: {len(records)}, "
        f"skipped (difficulty): {skipped_difficulty}, "
        f"skipped (<2 test cases): {skipped_test_cases}"
    )
    write_jsonl(records, out_path / "train.jsonl")
    return len(records)


# ---------------------------------------------------------------------------
# CodeContests dataset
# ---------------------------------------------------------------------------


def parse_code_contests_test_cases(example: dict) -> list[dict[str, str]]:
    """Extract public test cases from a CodeContests example."""
    test_cases: list[dict[str, str]] = []
    try:
        # CodeContests uses separate lists for public and private tests
        for split_name in ("public_tests", "private_tests", "generated_tests"):
            split = example.get(split_name, {})
            if not isinstance(split, dict):
                continue
            inputs = split.get("input", [])
            outputs = split.get("output", [])
            if not isinstance(inputs, list) or not isinstance(outputs, list):
                continue
            for inp, out in zip(inputs, outputs):
                inp_str = inp if isinstance(inp, str) else json.dumps(inp)
                out_str = out if isinstance(out, str) else json.dumps(out)
                test_cases.append(
                    {"input": inp_str.strip(), "expected_output": out_str.strip()}
                )
            # Prefer public tests; stop if we already have some
            if test_cases and split_name == "public_tests":
                break
    except Exception as exc:
        logger.debug(f"parse_code_contests_test_cases error: {exc}")
    return test_cases


CC_DIFFICULTY_MAP = {
    0: "unknown",
    1: "easy",
    2: "easy",
    3: "medium",
    4: "medium",
    5: "hard",
    6: "hard",
    7: "hard",
    8: "expert",
    9: "expert",
}


def download_code_contests(output_dir: Path, max_samples: int) -> int:
    """Download CodeContests dataset and save to output_dir/code_contests/train.jsonl."""
    logger.info(f"Loading CodeContests dataset (max_samples={max_samples}) …")
    out_path = output_dir / "code_contests"
    ensure_dir(out_path)

    try:
        ds = load_dataset(
            "deepmind/code_contests",
            split="train",
            trust_remote_code=True,
        )
    except Exception as exc:
        logger.error(f"Failed to load CodeContests: {exc}")
        return 0

    logger.info(f"CodeContests train split has {len(ds):,} total examples.")

    records: list[dict] = []
    skipped_test_cases = 0

    for idx, example in enumerate(tqdm(ds, desc="Processing CodeContests", unit="ex")):
        if len(records) >= max_samples:
            break

        test_cases = parse_code_contests_test_cases(example)
        if len(test_cases) < 2:
            skipped_test_cases += 1
            continue

        raw_difficulty = example.get("difficulty", 0)
        if not isinstance(raw_difficulty, int):
            try:
                raw_difficulty = int(raw_difficulty)
            except (TypeError, ValueError):
                raw_difficulty = 0

        problem_id = f"cc_{example.get('name', str(idx)).replace(' ', '_')}"
        title = example.get("name", f"Problem {idx}").strip()
        description = example.get("description", "").strip()

        records.append(
            {
                "problem_id": problem_id,
                "title": title,
                "description": description,
                "difficulty": CC_DIFFICULTY_MAP.get(raw_difficulty, "unknown"),
                "test_cases": test_cases,
                "source": "code_contests",
            }
        )

    logger.info(
        f"CodeContests — kept: {len(records)}, "
        f"skipped (<2 test cases): {skipped_test_cases}"
    )
    write_jsonl(records, out_path / "train.jsonl")
    return len(records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download APPS and CodeContests datasets from HuggingFace.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Root directory for raw data output.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=800,
        help="Maximum number of samples to keep per dataset.",
    )
    parser.add_argument(
        "--apps-max",
        type=int,
        default=None,
        help="Override max samples specifically for APPS (defaults to --max-samples).",
    )
    parser.add_argument(
        "--cc-max",
        type=int,
        default=None,
        help="Override max samples specifically for CodeContests (defaults to --max-samples).",
    )
    parser.add_argument(
        "--skip-apps",
        action="store_true",
        help="Skip downloading APPS.",
    )
    parser.add_argument(
        "--skip-code-contests",
        action="store_true",
        help="Skip downloading CodeContests.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    configure_hf_auth()

    output_dir: Path = args.output_dir.resolve()
    ensure_dir(output_dir)
    logger.info(f"Output directory: {output_dir}")

    apps_max = args.apps_max if args.apps_max is not None else args.max_samples
    cc_max = args.cc_max if args.cc_max is not None else args.max_samples

    total = 0

    if not args.skip_apps:
        n = download_apps(output_dir, max_samples=apps_max)
        total += n
    else:
        logger.info("Skipping APPS download.")

    if not args.skip_code_contests:
        n = download_code_contests(output_dir, max_samples=cc_max)
        total += n
    else:
        logger.info("Skipping CodeContests download.")

    logger.info(f"Download complete. Total problems saved: {total:,}")


if __name__ == "__main__":
    main()
