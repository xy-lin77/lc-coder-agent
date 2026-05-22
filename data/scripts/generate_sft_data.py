#!/usr/bin/env python3
"""Generate SFT training data with <think> CoT format using LLM APIs.

Loads raw problems from a JSONL file and calls an LLM (OpenAI GPT-4o,
Anthropic Claude, or DeepSeek) to generate chain-of-thought solutions
wrapped in <think>...</think> tags followed by a clean Python code block.

Supports checkpointing, async rate-limited batching, and output validation.

Usage:
    python generate_sft_data.py \\
        --input data/raw/apps/train.jsonl \\
        --output data/raw/generated_cot.jsonl \\
        --provider openai \\
        --model gpt-4o \\
        --max-problems 800 \\
        --max-workers 10
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger
from tqdm import tqdm

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
# Constants / Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert competitive programmer and software engineer.
When solving a coding problem, you MUST follow this exact format:

<think>
[Your detailed step-by-step reasoning here. Include:
 - Problem analysis and understanding
 - Edge cases to consider
 - Algorithm choice and complexity analysis
 - Step-by-step derivation of the solution]
</think>

```python
[Your clean, correct Python solution here — no extra commentary]
```

Rules:
- The <think> block must appear BEFORE the code block.
- The code block must use ```python fenced syntax.
- Do NOT include any text after the closing ``` of the code block.
- The code must be self-contained and runnable.
"""

USER_PROMPT_TEMPLATE = """\
Solve the following coding problem:

**{title}**

{description}

Provide your reasoning inside <think>...</think> tags, then your Python solution.
"""

# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "openai": {
        "model": "gpt-4o",
        "base_url": None,
        "env_key": "OPENAI_API_KEY",
    },
    "anthropic": {
        "model": "claude-3-5-sonnet-20241022",
        "base_url": None,
        "env_key": "ANTHROPIC_API_KEY",
    },
    "deepseek": {
        "model": "deepseek-reasoner",
        "base_url": "https://api.deepseek.com/v1",
        "env_key": "DEEPSEEK_API_KEY",
    },
}

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
CODE_BLOCK_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL)


def validate_response(text: str) -> tuple[bool, str]:
    """Check that a response contains both <think> tags and a python code block.

    Returns (is_valid, reason).
    """
    if not THINK_RE.search(text):
        return False, "missing <think>...</think> tags"
    think_match = THINK_RE.search(text)
    if think_match and len(think_match.group(1).strip()) < 20:
        return False, "<think> block is too short (< 20 chars)"
    if not CODE_BLOCK_RE.search(text):
        return False, "missing ```python code block"
    code_match = CODE_BLOCK_RE.search(text)
    if code_match and len(code_match.group(1).strip()) < 5:
        return False, "python code block is empty or too short"
    return True, "ok"


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------


class CheckpointManager:
    """Track which problem_ids have already been processed.

    The checkpoint file is the same as the output file — we read existing
    records on startup to build the seen-set, and append new records as
    they complete.
    """

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self._seen: set[str] = set()
        self._fh = None
        self._lock = asyncio.Lock()

    def load_existing(self) -> int:
        """Load already-processed problem_ids from the output file."""
        if not self.output_path.exists():
            return 0
        count = 0
        with open(self.output_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if "problem_id" in rec:
                        self._seen.add(rec["problem_id"])
                        count += 1
                except json.JSONDecodeError:
                    pass
        logger.info(f"Checkpoint: {count} already-processed records loaded from {self.output_path}")
        return count

    def is_done(self, problem_id: str) -> bool:
        return problem_id in self._seen

    def open_for_append(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.output_path, "a", encoding="utf-8")

    async def write_record(self, record: dict) -> None:
        async with self._lock:
            if self._fh is None:
                raise RuntimeError("Call open_for_append() before write_record().")
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._fh.flush()
            self._seen.add(record["problem_id"])

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


# ---------------------------------------------------------------------------
# API callers
# ---------------------------------------------------------------------------


async def call_openai(
    client: Any,
    model: str,
    problem: dict,
    semaphore: asyncio.Semaphore,
) -> tuple[str | None, str]:
    """Call OpenAI chat completion API. Returns (response_text, error_msg)."""
    user_content = USER_PROMPT_TEMPLATE.format(
        title=problem.get("title", "Untitled"),
        description=problem.get("description", ""),
    )
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.7,
                max_tokens=4096,
            )
            text = response.choices[0].message.content or ""
            return text, ""
        except Exception as exc:
            return None, str(exc)


async def call_anthropic(
    client: Any,
    model: str,
    problem: dict,
    semaphore: asyncio.Semaphore,
) -> tuple[str | None, str]:
    """Call Anthropic Messages API. Returns (response_text, error_msg)."""
    user_content = USER_PROMPT_TEMPLATE.format(
        title=problem.get("title", "Untitled"),
        description=problem.get("description", ""),
    )
    async with semaphore:
        try:
            response = await client.messages.create(
                model=model,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
                max_tokens=4096,
                temperature=0.7,
            )
            text = response.content[0].text if response.content else ""
            return text, ""
        except Exception as exc:
            return None, str(exc)


async def call_deepseek(
    client: Any,
    model: str,
    problem: dict,
    semaphore: asyncio.Semaphore,
) -> tuple[str | None, str]:
    """Call DeepSeek via OpenAI-compatible API. Returns (response_text, error_msg)."""
    # DeepSeek-R1 exposes reasoning_content separately; we concatenate it.
    user_content = USER_PROMPT_TEMPLATE.format(
        title=problem.get("title", "Untitled"),
        description=problem.get("description", ""),
    )
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.6,
                max_tokens=8192,
            )
            choice = response.choices[0]
            text = choice.message.content or ""
            # DeepSeek-R1 may put CoT in reasoning_content
            reasoning = getattr(choice.message, "reasoning_content", None)
            if reasoning and "<think>" not in text:
                text = f"<think>\n{reasoning}\n</think>\n\n{text}"
            return text, ""
        except Exception as exc:
            return None, str(exc)


# ---------------------------------------------------------------------------
# Per-problem task
# ---------------------------------------------------------------------------


async def process_problem(
    problem: dict,
    provider: str,
    model: str,
    client: Any,
    semaphore: asyncio.Semaphore,
    checkpoint: CheckpointManager,
    pbar: tqdm,
    stats: dict,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> None:
    """Generate CoT for one problem, validate, and write to checkpoint."""
    problem_id = problem["problem_id"]

    caller_map = {
        "openai": call_openai,
        "anthropic": call_anthropic,
        "deepseek": call_deepseek,
    }
    caller = caller_map[provider]

    response_text = None
    last_error = ""

    for attempt in range(1, max_retries + 1):
        text, err = await caller(client, model, problem, semaphore)
        if text is not None:
            response_text = text
            last_error = ""
            break
        last_error = err
        logger.warning(
            f"[{problem_id}] API error (attempt {attempt}/{max_retries}): {err}"
        )
        if attempt < max_retries:
            await asyncio.sleep(retry_delay * attempt)

    pbar.update(1)

    if response_text is None:
        logger.error(f"[{problem_id}] All retries exhausted: {last_error}")
        stats["failed"] += 1
        return

    valid, reason = validate_response(response_text)
    if not valid:
        logger.warning(f"[{problem_id}] Validation failed: {reason}")
        stats["invalid"] += 1
        # Still save it — downstream preprocessing will filter
        stats["saved_invalid"] += 1

    record = {
        "problem_id": problem_id,
        "prompt": USER_PROMPT_TEMPLATE.format(
            title=problem.get("title", "Untitled"),
            description=problem.get("description", ""),
        ),
        "response": response_text,
        "provider": provider,
        "model": model,
        "valid_format": valid,
        "validation_reason": reason,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    await checkpoint.write_record(record)
    stats["success"] += 1


# ---------------------------------------------------------------------------
# Main async driver
# ---------------------------------------------------------------------------


async def run_generation(
    problems: list[dict],
    provider: str,
    model: str,
    max_workers: int,
    checkpoint: CheckpointManager,
) -> dict:
    """Run async generation over all problems."""
    # Build client
    provider_cfg = PROVIDER_DEFAULTS[provider]
    api_key = os.environ.get(provider_cfg["env_key"], "")
    if not api_key:
        raise ValueError(
            f"API key not found. Set the {provider_cfg['env_key']} environment variable."
        )

    if provider in ("openai", "deepseek"):
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("Run: pip install openai")

        kwargs: dict[str, Any] = {"api_key": api_key}
        if provider_cfg["base_url"]:
            kwargs["base_url"] = provider_cfg["base_url"]
        client = AsyncOpenAI(**kwargs)
    elif provider == "anthropic":
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError("Run: pip install anthropic")
        client = AsyncAnthropic(api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    semaphore = asyncio.Semaphore(max_workers)
    stats: dict[str, int] = {
        "success": 0,
        "failed": 0,
        "invalid": 0,
        "saved_invalid": 0,
        "skipped": 0,
    }

    # Filter already-done problems
    pending = [p for p in problems if not checkpoint.is_done(p["problem_id"])]
    already_done = len(problems) - len(pending)
    stats["skipped"] = already_done
    logger.info(
        f"Problems: {len(problems)} total, {already_done} already done, {len(pending)} pending."
    )

    if not pending:
        logger.info("Nothing to do — all problems already processed.")
        return stats

    checkpoint.open_for_append()

    with tqdm(total=len(pending), desc="Generating CoT", unit="prob") as pbar:
        tasks = [
            process_problem(
                problem=p,
                provider=provider,
                model=model,
                client=client,
                semaphore=semaphore,
                checkpoint=checkpoint,
                pbar=pbar,
                stats=stats,
            )
            for p in pending
        ]
        await asyncio.gather(*tasks)

    checkpoint.close()
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def load_problems(input_path: Path, max_problems: int) -> list[dict]:
    """Load problems from a JSONL file."""
    problems: list[dict] = []
    with open(input_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "problem_id" not in rec:
                    continue
                problems.append(rec)
            except json.JSONDecodeError as exc:
                logger.warning(f"Skipping malformed JSONL line: {exc}")
            if len(problems) >= max_problems:
                break
    return problems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate SFT training data with <think> CoT format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input JSONL file (output of download_datasets.py).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/generated_cot.jsonl"),
        help="Path to output JSONL file (checkpointed).",
    )
    parser.add_argument(
        "--provider",
        choices=list(PROVIDER_DEFAULTS.keys()),
        default="openai",
        help="LLM API provider.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (defaults to provider default if not specified).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Max concurrent API calls (semaphore size).",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=800,
        help="Maximum number of problems to process.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Number of API retries per problem before giving up.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=5.0,
        help="Base delay (seconds) between retries (multiplied by attempt number).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    provider: str = args.provider
    model: str = args.model or PROVIDER_DEFAULTS[provider]["model"]

    logger.info(f"Provider: {provider}  |  Model: {model}")
    logger.info(f"Input: {args.input}  |  Output: {args.output}")

    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    problems = load_problems(args.input, args.max_problems)
    logger.info(f"Loaded {len(problems):,} problems from {args.input}")

    if not problems:
        logger.error("No problems loaded. Check the input file.")
        sys.exit(1)

    checkpoint = CheckpointManager(args.output)
    checkpoint.load_existing()

    start = time.perf_counter()
    stats = asyncio.run(
        run_generation(
            problems=problems,
            provider=provider,
            model=model,
            max_workers=args.max_workers,
            checkpoint=checkpoint,
        )
    )
    elapsed = time.perf_counter() - start

    logger.info("=" * 60)
    logger.info("Generation complete.")
    logger.info(f"  Elapsed:       {elapsed:.1f}s")
    logger.info(f"  Success:       {stats['success']}")
    logger.info(f"  Failed:        {stats['failed']}")
    logger.info(f"  Invalid fmt:   {stats['invalid']}")
    logger.info(f"  Skipped:       {stats['skipped']}")
    logger.info(f"  Output:        {args.output}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
