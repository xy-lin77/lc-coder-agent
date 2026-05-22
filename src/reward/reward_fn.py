#!/usr/bin/env python3
"""
GRPO reward function for code reasoning.
Rewards = format_reward + execution_reward.
Integrates with verl's RewardManager interface.
"""

from __future__ import annotations

import traceback
from typing import Any, Optional

from loguru import logger

from .code_executor import execute_with_timeout, extract_code_block, extract_function_name
from .hardcode_detector import compute_format_reward, detect_hardcoded_output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_and_normalize_output(raw_output: Any) -> str:
    """
    Convert the raw return value from code execution to a canonical string.

    - ``None`` → ``"None"``
    - Exceptions (passed as strings from the worker) are returned unchanged.
    - Everything else is ``str(raw_output).strip()``.
    """
    if raw_output is None:
        return "None"
    try:
        return str(raw_output).strip()
    except Exception as exc:
        return f"<repr error: {exc}>"


def _outputs_equal(actual: Any, expected: Any) -> bool:
    """
    Compare two values for semantic equality.

    Try, in order:
    1. Direct Python equality  (handles int, float, list, dict, …)
    2. String equality of ``str()`` representations
    3. ``eval(str(actual)) == expected``  (handles e.g. "[1, 2]" vs [1, 2])
    4. ``actual == eval(str(expected))``  (handles expected stored as string)
    """
    # 1. Direct equality
    if actual == expected:
        return True

    # 2. String equality
    try:
        if str(actual).strip() == str(expected).strip():
            return True
    except Exception:
        pass

    # 3. eval(str(actual))
    try:
        parsed_actual = eval(str(actual))  # noqa: S307
        if parsed_actual == expected:
            return True
    except Exception:
        pass

    # 4. eval(str(expected))
    try:
        parsed_expected = eval(str(expected))  # noqa: S307
        if actual == parsed_expected:
            return True
    except Exception:
        pass

    return False


# ---------------------------------------------------------------------------
# Core reward computation
# ---------------------------------------------------------------------------

def compute_reward(
    response: str,
    test_cases: list[dict],
    timeout: int = 5,
) -> dict:
    """
    Compute the GRPO reward for a single model *response* given *test_cases*.

    Each element of *test_cases* must be a dict with at least:
        ``args``            – positional arguments (list)
        ``kwargs``          – keyword arguments (dict, optional)
        ``expected_output`` – expected return value

    Returns
    -------
    dict with keys:
        total_reward       float in [0, 1]
        format_reward      float
        execution_reward   float
        pass_rate          float in [0, 1]
        passed             int
        total              int
        hardcode_detected  bool
        error_details      list[str]
    """
    error_details: list[str] = []

    # ------------------------------------------------------------------
    # Format reward (always computed)
    # ------------------------------------------------------------------
    format_reward = compute_format_reward(response)

    # ------------------------------------------------------------------
    # Extract code
    # ------------------------------------------------------------------
    code = extract_code_block(response)
    if not code:
        logger.debug("No code block found in response; awarding format reward only.")
        return {
            "total_reward": round(max(0.0, min(1.0, format_reward)), 6),
            "format_reward": format_reward,
            "execution_reward": 0.0,
            "pass_rate": 0.0,
            "passed": 0,
            "total": len(test_cases),
            "hardcode_detected": False,
            "error_details": ["NoCodeBlock: no Python code block found in response"],
        }

    # ------------------------------------------------------------------
    # Extract function name
    # ------------------------------------------------------------------
    fn_name = extract_function_name(code)
    if not fn_name:
        logger.debug("Could not determine function name from code.")
        return {
            "total_reward": round(max(0.0, min(1.0, format_reward)), 6),
            "format_reward": format_reward,
            "execution_reward": 0.0,
            "pass_rate": 0.0,
            "passed": 0,
            "total": len(test_cases),
            "hardcode_detected": False,
            "error_details": ["NoFunctionDef: no top-level function definition found"],
        }

    # ------------------------------------------------------------------
    # Hardcode detection
    # ------------------------------------------------------------------
    hardcode_penalty = 0.0
    hardcode_detected = False
    if test_cases:
        try:
            hardcode_detected = detect_hardcoded_output(code, test_cases)
        except Exception as exc:
            logger.warning(f"Hardcode detector raised: {exc}")
            hardcode_detected = False

    if hardcode_detected:
        logger.debug(f"Hardcoded output detected; applying penalty -0.2.")
        hardcode_penalty = -0.2

    # ------------------------------------------------------------------
    # Execute each test case
    # ------------------------------------------------------------------
    passed = 0
    total = len(test_cases)

    for idx, tc in enumerate(test_cases):
        args = tuple(tc.get("args", []))
        kwargs = tc.get("kwargs", {}) or {}
        expected = tc.get("expected_output")

        status, payload = execute_with_timeout(
            code=code,
            fn_name=fn_name,
            args=args,
            kwargs=kwargs,
            timeout=timeout,
        )

        if status == "err":
            error_details.append(f"test_case[{idx}]: {payload}")
            continue

        if _outputs_equal(payload, expected):
            passed += 1
        else:
            actual_str = extract_and_normalize_output(payload)
            expected_str = extract_and_normalize_output(expected)
            error_details.append(
                f"test_case[{idx}]: wrong answer – got {actual_str!r}, expected {expected_str!r}"
            )

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    pass_rate = passed / total if total > 0 else 0.0
    execution_reward = pass_rate * 0.9

    total_reward = format_reward + execution_reward + hardcode_penalty
    total_reward = round(max(0.0, min(1.0, total_reward)), 6)

    return {
        "total_reward": total_reward,
        "format_reward": format_reward,
        "execution_reward": execution_reward,
        "pass_rate": pass_rate,
        "passed": passed,
        "total": total,
        "hardcode_detected": hardcode_detected,
        "error_details": error_details,
    }


# ---------------------------------------------------------------------------
# verl-compatible RewardManager
# ---------------------------------------------------------------------------

class CodeRewardManager:
    """
    verl-compatible reward manager for code reasoning with GRPO.

    verl calls this object as a callable with a ``DataProto`` batch.  For each
    sample the manager:

    1. Decodes the response token IDs.
    2. Retrieves test cases from ``data.non_tensor_batch["reward_model"]["ground_truth"]["test_cases"]``.
    3. Calls ``compute_reward`` to get a scalar reward.
    4. Writes the reward into a tensor that is returned via the ``DataProto``.

    If any sample raises an unexpected exception the reward defaults to 0.0 and
    a warning is logged so training can continue.

    Parameters
    ----------
    tokenizer:
        HuggingFace tokenizer used to decode response token IDs.
    num_examine:
        Number of samples to log at DEBUG level for inspection (0 = disabled).
    """

    def __init__(self, tokenizer, num_examine: int = 0) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _decode_response(self, token_ids) -> str:
        """Decode a 1-D tensor / list of token IDs to a string."""
        try:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        except Exception as exc:
            logger.warning(f"Tokenizer decode failed: {exc}")
            return ""

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(self, data):  # data: DataProto
        """
        Compute rewards for all samples in *data* and return a ``DataProto``
        with ``batch["token_level_scores"]`` or ``batch["rewards"]`` filled.

        verl typically passes either ``responses`` (token IDs) or
        ``input_ids`` + ``attention_mask``.  We look for ``responses`` first.
        """
        import torch  # noqa: PLC0415  (lazy import to avoid hard dep at module level)

        try:
            from verl import DataProto  # noqa: PLC0415
        except ImportError:
            # Allow the module to be imported without verl installed (e.g. for
            # unit tests) by using a minimal duck-type check instead.
            DataProto = None  # type: ignore[assignment]

        batch_size = data.batch.batch_size[0] if hasattr(data.batch, "batch_size") else len(data.batch["responses"])

        rewards = torch.zeros(batch_size, dtype=torch.float32)

        for i in range(batch_size):
            try:
                # ----------------------------------------------------------
                # Decode response
                # ----------------------------------------------------------
                if "responses" in data.batch:
                    token_ids = data.batch["responses"][i]
                else:
                    # Fallback: slice from input_ids using attention_mask
                    token_ids = data.batch["input_ids"][i]

                response_str = self._decode_response(token_ids)

                # ----------------------------------------------------------
                # Retrieve test cases
                # ----------------------------------------------------------
                reward_model_data = data.non_tensor_batch["reward_model"][i]
                ground_truth = reward_model_data.get("ground_truth", {})

                if isinstance(ground_truth, str):
                    import json  # noqa: PLC0415
                    try:
                        ground_truth = json.loads(ground_truth)
                    except json.JSONDecodeError:
                        ground_truth = {}

                test_cases: list[dict] = ground_truth.get("test_cases", [])

                if not test_cases:
                    logger.warning(f"Sample {i}: no test cases found in ground_truth; reward = 0.0")
                    rewards[i] = 0.0
                    continue

                # ----------------------------------------------------------
                # Compute reward
                # ----------------------------------------------------------
                result = compute_reward(response_str, test_cases)
                rewards[i] = result["total_reward"]

                if i < self.num_examine:
                    logger.debug(
                        f"[examine {i}] pass_rate={result['pass_rate']:.2f} "
                        f"fmt={result['format_reward']:.3f} "
                        f"exec={result['execution_reward']:.3f} "
                        f"hardcode={result['hardcode_detected']} "
                        f"reward={result['total_reward']:.4f}"
                    )
                    if result["error_details"]:
                        for err in result["error_details"][:3]:
                            logger.debug(f"  error: {err}")

            except Exception as exc:
                logger.warning(
                    f"CodeRewardManager: unexpected error on sample {i}: "
                    f"{type(exc).__name__}: {exc}\n"
                    f"{traceback.format_exc()}"
                )
                rewards[i] = 0.0

        # ------------------------------------------------------------------
        # Write rewards back into the DataProto
        # ------------------------------------------------------------------
        data.batch["token_level_scores"] = rewards.unsqueeze(-1)
        # Also provide a flat rewards tensor for compatibility
        data.batch["rewards"] = rewards

        return data
