#!/usr/bin/env python3
"""CLI wrapper around EleutherAI lm-eval (HF backend)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("posttrain.eval")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "eval.yaml"
SUITE_FILES = {
    "instruct": EVAL_DIR / "suites" / "instruct_tasks.txt",
}
MATH_TASKS = "gsm8k"


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


def _read_task_file(path: Path) -> str:
    """Parse a task list file into a comma-separated lm-eval task string.

    Args:
        path: Text file with one task name per line (# comments ok).

    Returns:
        Comma-separated task names.
    """
    tasks: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Allow leftover LightEval-style ``name|shots`` lines; ignore shots.
        tasks.append(line.split("|", 1)[0])
    return ",".join(tasks)


def resolve_tasks(cfg: dict, tasks_arg: str | None, suite_arg: str | None) -> str:
    """Resolve which lm-eval tasks to run.

    Args:
        cfg: Loaded eval YAML.
        tasks_arg: CLI override — comma list or path to ``.txt``.
        suite_arg: ``instruct`` | ``math`` (if no tasks).

    Returns:
        Comma-separated task names for lm-eval.
    """
    if tasks_arg:
        p = Path(tasks_arg)
        resolved = _resolve_path(p)
        if p.suffix == ".txt" and resolved.exists():
            return _read_task_file(resolved)
        return ",".join(t.split("|", 1)[0] for t in tasks_arg.split(",") if t.strip())
    if cfg.get("tasks"):
        raw = cfg["tasks"]
        if isinstance(raw, list):
            return ",".join(str(t).split("|", 1)[0] for t in raw)
        return ",".join(t.split("|", 1)[0] for t in str(raw).split(",") if t.strip())
    suite = suite_arg or cfg.get("suite") or "instruct"
    if suite == "math":
        return MATH_TASKS
    path = SUITE_FILES.get(suite)
    if path is None or not path.exists():
        raise SystemExit(f"unknown suite={suite!r}; use instruct|math or --tasks")
    return _read_task_file(path)


def resolve_model_name(model_path: str) -> str:
    """Resolve HF id / local dir for lm-eval ``pretrained=``.

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


def _first_response(sample: dict) -> str:
    """Pull the primary model string from an lm-eval sample dict.

    Prefers raw ``resps`` over ``filtered_resps`` (filters may be ``[invalid]``).
    """
    for key in ("resps", "filtered_resps"):
        val = sample.get(key)
        if not val:
            continue
        cur = val
        while isinstance(cur, (list, tuple)) and cur:
            cur = cur[0]
        if isinstance(cur, str) and cur and cur != "[invalid]":
            return cur
    return ""


def format_readable_samples(task_name: str, samples: list[dict]) -> list[dict]:
    """Collapse raw lm-eval sample rows into readable per-example JSON objects.

    Args:
        task_name: lm-eval task id (``ifeval``, ``gsm8k``, …).
        samples: Raw sample dicts (may include multiple filter rows per doc).

    Returns:
        Sorted list of slim records with prompt/response/target/metrics.
    """
    skip = {
        "doc_id",
        "doc",
        "target",
        "arguments",
        "resps",
        "filtered_resps",
        "filter",
        "metrics",
        "doc_hash",
        "prompt_hash",
        "target_hash",
    }
    by_id: dict[int, dict] = {}
    for sample in samples:
        doc_id = int(sample.get("doc_id", len(by_id)))
        doc = sample.get("doc") or {}
        filt = sample.get("filter") or "none"

        score_keys = {
            k: v
            for k, v in sample.items()
            if k not in skip and not str(k).endswith("_stderr")
        }
        if isinstance(sample.get("metrics"), dict):
            score_keys.update(sample["metrics"])

        if doc_id not in by_id:
            if task_name == "ifeval":
                prompt = doc.get("prompt", "")
            elif "question" in doc:
                prompt = doc["question"]
            else:
                prompt = doc.get("prompt") or doc.get("query") or ""
            entry = {
                "id": doc_id,
                "prompt": prompt,
                "response": _first_response(sample),
                "target": sample.get("target", ""),
                "metrics": {},
            }
            if task_name == "ifeval" and doc.get("instruction_id_list") is not None:
                entry["instruction_ids"] = doc["instruction_id_list"]
            by_id[doc_id] = entry

        entry = by_id[doc_id]
        if not entry["response"]:
            entry["response"] = _first_response(sample)

        # gsm8k emits one row per filter (strict-match / flexible-extract).
        extracted = sample.get("filtered_resps")
        while isinstance(extracted, (list, tuple)) and extracted:
            extracted = extracted[0]
        # Only keep extracted answers when they differ from the full response
        # (e.g. gsm8k number extract); skip ifeval-style full-string filters.
        if (
            isinstance(extracted, str)
            and extracted
            and extracted != "[invalid]"
            and extracted != entry["response"]
        ):
            entry["metrics"][f"extracted:{filt}" if filt != "none" else "extracted"] = extracted

        if filt and filt != "none":
            for k, v in score_keys.items():
                key = k if ":" in str(k) else f"{k}:{filt}"
                entry["metrics"][key] = v
        else:
            entry["metrics"].update(score_keys)

    return [by_id[i] for i in sorted(by_id)]


def write_readable_samples(out_path: Path, task_name: str, samples: list[dict]) -> Path:
    """Write pretty-printed ``answers_<task>.json`` next to lm-eval outputs.

    Args:
        out_path: Eval output directory (same as ``--output-path``).
        task_name: Task id.
        samples: Raw lm-eval samples for that task.

    Returns:
        Path of the written JSON file.
    """
    # EvaluationTracker nests under sanitized model name.
    model_dirs = [p for p in out_path.iterdir() if p.is_dir()] if out_path.is_dir() else []
    dest_dir = model_dirs[0] if len(model_dirs) == 1 else out_path
    dest = dest_dir / f"answers_{task_name}.json"
    payload = format_readable_samples(task_name, samples)
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")
    return dest


def main() -> None:
    """Parse CLI/config and run lm-eval.

    Side effects:
        Writes results under ``output_path``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=str, default=None, help="HF id / local ckpt")
    parser.add_argument("--suite", type=str, default=None, choices=["instruct", "math"])
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated lm-eval tasks or path to .txt task list",
    )
    parser.add_argument("--limit", type=int, default=None, help="Smoke cap (max samples)")
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument("--max-gen-toks", type=int, default=None)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--log-samples", action="store_true", help="Save per-sample outputs")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config.exists() else {}
    model_path = args.model or cfg.get("model")
    if not model_path:
        raise SystemExit("pass --model or set model: in configs/eval.yaml")

    tasks = resolve_tasks(cfg, args.tasks, args.suite)
    out_path = _resolve_path(args.output_path or cfg.get("output_path", "outputs/lm_eval"))
    out_path.mkdir(parents=True, exist_ok=True)

    device = args.device or cfg.get("device") or pick_device()
    dtype = args.dtype or cfg.get("dtype") or pick_dtype()
    batch_size = args.batch_size if args.batch_size is not None else cfg.get("batch_size", 8)
    limit = args.limit if args.limit is not None else cfg.get("max_samples")
    num_fewshot = args.num_fewshot if args.num_fewshot is not None else cfg.get("num_fewshot")
    max_gen_toks = args.max_gen_toks if args.max_gen_toks is not None else cfg.get("max_gen_toks", 256)
    apply_chat = not args.no_chat_template and bool(cfg.get("apply_chat_template", True))
    log_samples = args.log_samples or bool(cfg.get("log_samples", False))

    model_name = resolve_model_name(model_path)
    model_args = f"pretrained={model_name},dtype={dtype},trust_remote_code=True"

    # Don't pass num_fewshot=None into lm-eval — that can overwrite task defaults oddly.
    fewshot_kw: dict = {}
    if num_fewshot is not None:
        fewshot_kw["num_fewshot"] = int(num_fewshot)

    # Ensure ifeval extras are importable before lm_eval loads task YAML.
    try:
        import immutabledict  # noqa: F401
        import nltk

        for pkg in ("punkt", "punkt_tab"):
            try:
                nltk.data.find(f"tokenizers/{pkg}")
            except LookupError:
                nltk.download(pkg, quiet=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ifeval optional deps: %s", exc)

    logger.info("lm-eval model_args=%s device=%s batch_size=%s", model_args, device, batch_size)
    logger.info("tasks=%s", tasks)
    logger.info("output_dir=%s", out_path)

    from lm_eval import simple_evaluate
    from lm_eval.loggers import EvaluationTracker

    tracker = EvaluationTracker(output_path=str(out_path))
    results = simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks.split(","),
        device=device,
        batch_size=int(batch_size) if batch_size is not None else "auto",
        limit=limit,
        apply_chat_template=apply_chat,
        fewshot_as_multiturn=apply_chat,
        gen_kwargs=f"max_gen_toks={int(max_gen_toks)},do_sample=False,temperature=0.0",
        log_samples=log_samples,
        evaluation_tracker=tracker,
        confirm_run_unsafe_code=True,
        **fewshot_kw,
    )
    if results is None:
        raise SystemExit("lm-eval returned no results")

    # lm-eval 0.4.x: simple_evaluate returns results but does not write files;
    # mirror the CLI and persist via EvaluationTracker.
    samples = results.pop("samples", None) if log_samples else None
    if not log_samples:
        results.pop("samples", None)
    tracker.save_results_aggregated(
        results=results,
        samples=samples if log_samples else None,
    )
    if log_samples and samples:
        for task_name in results.get("configs", {}):
            if task_name in samples:
                tracker.save_results_samples(
                    task_name=task_name, samples=samples[task_name]
                )
                pretty = write_readable_samples(out_path, task_name, samples[task_name])
                logger.info("Wrote readable samples %s", pretty)
        logger.info("Wrote per-task samples under %s", out_path)
    else:
        logger.info("Wrote aggregated results under %s", out_path)

    # Print compact table-friendly summary.
    table = results.get("results") or {}
    for task_name, metrics in sorted(table.items()):
        flat = {
            k: v
            for k, v in metrics.items()
            if not str(k).endswith("_stderr") and k not in ("alias",)
        }
        logger.info("%s %s", task_name, flat)


if __name__ == "__main__":
    main()
