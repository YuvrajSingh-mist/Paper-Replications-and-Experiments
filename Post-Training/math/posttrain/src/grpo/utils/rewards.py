"""GSM8K reward for TRL GRPO — correctness + light format shaping."""

from __future__ import annotations

import math
import re
from typing import Any


def parse_answer(text: str) -> float:
    """Pull a float from <answer>...</answer>, else from the last number in the text."""
    tagged = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    candidate = tagged.group(1) if tagged else text
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", candidate.replace(",", ""))
    if not nums:
        return float("nan")
    try:
        return float(nums[-1])
    except ValueError:
        return float("nan")


def _think_ok(text: str) -> bool:
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    return bool(m and m.group(1).strip())


def _format_ok(text: str) -> bool:
    if not _think_ok(text):
        return False
    if not re.search(r"<answer>\s*.*?\s*</answer>", text, re.DOTALL | re.IGNORECASE):
        return False
    return math.isfinite(parse_answer(text))


def gsm8k_reward(
    prompts: list[Any],
    completions: list[Any],
    answer: list[str],
    **kwargs: Any,
) -> list[float]:
    """TRL reward_funcs signature — extra dataset columns arrive as kwargs.

    +1.0 correct numeric answer
    +0.1 non-empty <think>
    +0.1 full <think>+<answer> format with a parseable number
    """
    rewards: list[float] = []
    for completion, gold in zip(completions, answer, strict=True):
        if isinstance(completion, list):
            # conversational assistant message(s)
            text = completion[-1]["content"] if completion else ""
        else:
            text = str(completion)
        try:
            gold_f = float(str(gold).replace(",", ""))
        except ValueError:
            gold_f = float("nan")
        pred = parse_answer(text)
        correct = (
            1.0
            if math.isfinite(pred) and math.isfinite(gold_f) and math.isclose(pred, gold_f, abs_tol=1e-6)
            else 0.0
        )
        rewards.append(float(correct + 0.1 * _think_ok(text) + 0.1 * _format_ok(text)))
    return rewards
