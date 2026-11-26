"""GSM8K → TRL conversational SFT messages."""

from __future__ import annotations

import re
from typing import Any

from datasets import Dataset, load_dataset

SYSTEM_PROMPT = (
    "You are an assistant expert at solving math problems. "
    "Think step by step inside <think>...</think>, then put the final numeric "
    "answer inside <answer>...</answer>."
)


def _split_gsm8k_answer(answer: str) -> tuple[str, str] | None:
    """Return (reasoning, final_answer) from a GSM8K gold string."""
    match = re.search(r"####\s*(.+)\s*$", answer.strip())
    if not match:
        match = re.search(r"###\s*(.+)\s*$", answer.strip())
    if not match:
        return None
    final = match.group(1).strip().replace(",", "")
    reasoning = answer[: match.start()].strip()
    return reasoning, final


def _to_messages(question: str, reasoning: str, final: str) -> list[dict[str, str]]:
    assistant = f"<think>\n{reasoning}\n</think>\n<answer>{final}</answer>"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": assistant},
    ]


def build_gsm8k_sft_dataset(
    dataset_name: str = "openai/gsm8k",
    subset: str = "main",
    split: str = "train",
    max_samples: int | None = None,
) -> Dataset:
    raw = load_dataset(dataset_name, subset, split=split)
    rows: list[dict[str, Any]] = []
    for ex in raw:
        parsed = _split_gsm8k_answer(ex["answer"])
        if parsed is None:
            continue
        reasoning, final = parsed
        rows.append({"messages": _to_messages(ex["question"], reasoning, final)})
        if max_samples is not None and len(rows) >= max_samples:
            break
    return Dataset.from_list(rows)
