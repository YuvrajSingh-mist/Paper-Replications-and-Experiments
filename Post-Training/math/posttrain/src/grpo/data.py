"""GSM8K prompts for TRL GRPO — conversational messages + numeric gold answers."""

from __future__ import annotations

import re
from typing import Any

from datasets import Dataset, load_dataset

SYSTEM_PROMPT = (
    "You are an assistant expert at solving math problems. "
    "Think step by step inside <think>...</think>, then put the final numeric "
    "answer inside <answer>...</answer>."
)


def extract_gsm8k_answer(answer: str) -> str | None:
    match = re.search(r"####\s*(.+)\s*$", answer.strip())
    if not match:
        match = re.search(r"###\s*(.+)\s*$", answer.strip())
    if not match:
        return None
    return match.group(1).strip().replace(",", "")


def build_gsm8k_dataset(
    dataset_name: str = "openai/gsm8k",
    subset: str = "main",
    split: str = "train",
    max_samples: int | None = None,
) -> Dataset:
    raw = load_dataset(dataset_name, subset, split=split)
    rows: list[dict[str, Any]] = []
    for ex in raw:
        gold = extract_gsm8k_answer(ex["answer"])
        if gold is None:
            continue
        rows.append(
            {
                # TRL conversational format — chat template applied inside rollout_func.
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": ex["question"]},
                ],
                "answer": gold,
            }
        )
        if max_samples is not None and len(rows) >= max_samples:
            break
    return Dataset.from_list(rows)
