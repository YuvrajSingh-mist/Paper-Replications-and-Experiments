#!/usr/bin/env python3
"""CLI wrapper: build LightEval model_args + task list, then call `lighteval accelerate`."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("math_eval")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "eval.yaml"
SUITE_FILES = {
    "instruct": PROJECT_ROOT / "eval" / "suites" / "instruct_tasks.txt",
    "instruct_full": PROJECT_ROOT / "eval" / "suites" / "instruct_full_tasks.txt",
    "base": PROJECT_ROOT / "eval" / "suites" / "base_tasks.txt",
}
MATH_TASKS = "gsm8k|5"


def load_config(path: Path) -> dict:
    """Load YAML config.

    Args:
        path: Absolute or project-relative YAML path.

    Returns:
        Parsed mapping (empty dict if file is empty).
    """
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _resolve_path(p: str | Path) -> Path:
    """Resolve path relative to project root if not absolute.

    Args:
        p: Path string or Path.

    Returns:
        Absolute Path.
    """
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def pick_device() -> str:
    """Pick inference device.

    Returns:
        ``mps``, ``cuda``, or ``cpu``.
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def pick_dtype() -> str:
    """Pick default dtype for the current device.

    Returns:
        ``float16`` (MPS), ``bfloat16`` (CUDA), or ``float32``.
    """
    if torch.backends.mps.is_available():
        return "float16"
    if torch.cuda.is_available():
        return "bfloat16"
    return "float32"


def _patch_hf_hub_list_models() -> None:
    """Shim: map deprecated ``model_name=`` to ``search=`` for huggingface_hub ≥0.24.

    LightEval 0.13 still calls ``HfApi.list_models(model_name=...)`` when loading
    some optional task packs. We do not modify LightEval; this only patches Hub
    in-process so registry import succeeds.
    """
    from huggingface_hub import HfApi

    if getattr(HfApi.list_models, "_math_posttrain_patched", False):
        return
    orig = HfApi.list_models

    def _list_models(self, *args, model_name=None, **kwargs):  # noqa: ANN001
        if model_name is not None and "search" not in kwargs:
            kwargs["search"] = model_name
        return orig(self, *args, **kwargs)

    _list_models._math_posttrain_patched = True  # type: ignore[attr-defined]
    HfApi.list_models = _list_models  # type: ignore[method-assign]


def _read_task_file(path: Path) -> str:
    """Parse a LightEval task list file into a CLI task string.

    Args:
        path: Text file with one ``name|fewshot`` per line (# comments ok).

    Returns:
        Comma-separated task string, e.g. ``\"gsm8k|5,ifeval|0\"``.
    """
    tasks: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tasks.append(line)
    return ",".join(tasks)


def resolve_tasks(cfg: dict, tasks_arg: str | None, suite_arg: str | None) -> str:
    """Resolve which LightEval tasks to run.

    Args:
        cfg: Loaded eval YAML.
        tasks_arg: CLI override — comma list or path to ``.txt``.
        suite_arg: ``instruct`` | ``instruct_full`` | ``base`` | ``math`` (if no tasks).

    Returns:
        Comma-separated ``name|fewshot`` string for LightEval.
    """
    if tasks_arg:
        p = Path(tasks_arg)
        resolved = _resolve_path(p)
        if p.suffix == ".txt" and resolved.exists():
            return _read_task_file(resolved)
        return tasks_arg
    if cfg.get("tasks"):
        raw = cfg["tasks"]
        if isinstance(raw, list):
            return ",".join(str(t) for t in raw)
        return str(raw)
    suite = suite_arg or cfg.get("suite") or "instruct"
    if suite == "math":
        return MATH_TASKS
    path = SUITE_FILES.get(suite)
    if path is None or not path.exists():
        raise SystemExit(f"unknown suite={suite!r}; use instruct|instruct_full|base|math or --tasks")
    return _read_task_file(path)


def resolve_model_name(model_path: str) -> str:
    """Resolve HF id / local dir for LightEval ``model_name=``.

    Args:
        model_path: HF hub id, checkpoint dir, or PEFT adapter dir.

    Returns:
        Model id/path string. PEFT dirs fall back to ``base_model_name_or_path``.
    """
    model_path_p = Path(model_path)
    adapter_cfg = model_path_p / "adapter_config.json"
    if adapter_cfg.exists():
        base = json.loads(adapter_cfg.read_text()).get("base_model_name_or_path")
        if not base:
            raise SystemExit(f"PEFT adapter at {model_path_p} missing base_model_name_or_path")
        logger.warning(
            "PEFT adapter detected; evaluating base=%s only (merge adapter before eval for LoRA scores)",
            base,
        )
        return base
    return model_path


def main() -> None:
    """Parse CLI/config and run LightEval accelerate in-process.

    Inputs (CLI or ``configs/eval.yaml``):
        --model, --suite/--tasks, --output-path, device/dtype/batch/limit.

    Side effects:
        Writes LightEval results under ``output_path`` (no return value).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=str, default=None, help="HF id / local ckpt")
    parser.add_argument(
        "--suite",
        type=str,
        default=None,
        choices=["instruct", "instruct_full", "base", "math"],
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated LightEval tasks (name|fewshot) or path to .txt task list",
    )
    parser.add_argument("--limit", type=int, default=None, help="--max-samples smoke cap")
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--save-details", action="store_true")
    args = parser.parse_args()

    _patch_hf_hub_list_models()

    cfg = load_config(args.config) if args.config.exists() else {}
    model_path = args.model or cfg.get("model")
    if not model_path:
        raise SystemExit("pass --model or set model: in configs/eval.yaml")

    tasks = resolve_tasks(cfg, args.tasks, args.suite)
    out_path = _resolve_path(args.output_path or cfg.get("output_path", "outputs/lighteval"))
    out_path.mkdir(parents=True, exist_ok=True)

    device = args.device or cfg.get("device") or pick_device()
    dtype = args.dtype or cfg.get("dtype") or pick_dtype()
    batch_size = args.batch_size if args.batch_size is not None else cfg.get("batch_size")
    max_samples = args.limit if args.limit is not None else cfg.get("max_samples")
    use_chat = not args.no_chat_template and bool(cfg.get("override_chat_template", True))
    save_details = args.save_details or bool(cfg.get("save_details", False))

    model_name = resolve_model_name(model_path)
    # Official LightEval knobs — needed on MPS (auto batch probe OOMs at 8k ctx).
    # Note: logprobs batch_size is flattened (samples × max_choices); set in eval.yaml.
    max_length = cfg.get("max_length") or (2048 if device == "mps" else None)
    if batch_size is None:
        batch_size = 32 if device == "mps" else batch_size
    model_args = (
        f"model_name={model_name},dtype={dtype},device={device},"
        f"trust_remote_code=True,override_chat_template={str(use_chat)}"
    )
    if batch_size is not None:
        model_args += f",batch_size={int(batch_size)}"
    if max_length is not None:
        model_args += f",max_length={int(max_length)}"

    logger.info("LightEval model_args=%s", model_args)
    logger.info("tasks=%s", tasks)
    logger.info("output_dir=%s", out_path)

    custom_tasks = PROJECT_ROOT / "eval" / "suites" / "custom_tasks.py"

    from lighteval.main_accelerate import accelerate

    accelerate(
        model_args=model_args,
        tasks=tasks,
        output_dir=str(out_path),
        max_samples=max_samples,
        save_details=save_details,
        custom_tasks=str(custom_tasks) if custom_tasks.exists() else None,
    )


if __name__ == "__main__":
    main()
