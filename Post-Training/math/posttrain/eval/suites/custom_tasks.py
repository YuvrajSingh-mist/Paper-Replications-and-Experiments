"""Extra LightEval tasks: PIQA via parquet mirror (`lighteval/piqa`) for datasets≥4."""

from __future__ import annotations

from string import ascii_uppercase

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def piqa_prompt(line: dict, task_name: str | None = None) -> Doc:
    """Format one PIQA row for LightEval.

    Args:
        line: Dataset row with ``goal``, ``sol1``, ``sol2``, ``label``.
        task_name: Task id (set by harness).

    Returns:
        ``Doc`` with multiple-choice query and gold index.
    """
    letters = list(ascii_uppercase)[:2]
    query = "The following are multiple choice questions (with answers) about common sense.\n"
    query += f"Question: {line['goal']}\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(letters, [line["sol1"], line["sol2"]])])
    query += "Answer: "

    gold_ix = int(line["label"])
    is_few_shots = line.get("__few_shots", False)
    return Doc(
        task_name=task_name,
        query=query,
        choices=letters if not is_few_shots else [line["sol1"], line["sol2"]],
        gold_index=gold_ix,
        instruction="The following are multiple choice questions (with answers) about common sense.\n",
    )


# Built-in name ``piqa`` points at script-based ``ybisk/piqa`` (broken on datasets≥4).
piqa_hf = LightevalTaskConfig(
    name="piqa_hf",
    prompt_function=piqa_prompt,
    hf_repo="lighteval/piqa",
    hf_subset="plain_text",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=1,
)

TASKS_TABLE = [piqa_hf]
