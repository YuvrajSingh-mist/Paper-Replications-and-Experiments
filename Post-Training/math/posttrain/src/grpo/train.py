#!/usr/bin/env python3
"""TRL GRPO on one Mac Mini; vllm-metal rollouts on the other Minis."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import yaml
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from grpo.data import build_gsm8k_dataset
from grpo.utils import WeightSyncCallback, gsm8k_reward, make_rollout_func, sync_workers

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("grpo.qwen")

# Post-Training/math/posttrain/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "grpo.yaml"


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _resolve_path(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def pick_dtype() -> torch.dtype:
    if torch.backends.mps.is_available():
        return torch.float32  # safest on MPS for training
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-steps", type=int, default=None, help="Override for smoke runs")
    parser.add_argument("--skip-initial-sync", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = _resolve_path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    cfg["sync"]["local_adapter_dir"] = str(_resolve_path(cfg["sync"]["local_adapter_dir"]))

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = pick_dtype()
    device_map = "mps" if torch.backends.mps.is_available() else "auto"
    logger.info("loading %s dtype=%s device_map=%s", cfg["model_name"], dtype, device_map)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=device_map,
    )

    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg.get("dropout", 0.0)),
        target_modules=list(lora_cfg["target_modules"]),
        task_type="CAUSAL_LM",
        bias="none",
    )

    data_cfg = cfg["data"]
    train_ds = build_gsm8k_dataset(
        dataset_name=data_cfg["dataset_name"],
        subset=data_cfg.get("subset", "main"),
        split=data_cfg.get("split", "train"),
        max_samples=data_cfg.get("max_train_samples"),
    )
    logger.info("train examples: %d", len(train_ds))

    g = cfg["grpo"]
    grpo_args = GRPOConfig(
        output_dir=str(out),
        run_name=out.name,
        seed=int(cfg.get("seed", 42)),
        learning_rate=float(g["learning_rate"]),
        per_device_train_batch_size=int(g["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(g["gradient_accumulation_steps"]),
        num_train_epochs=float(g["num_train_epochs"]),
        logging_steps=int(g.get("logging_steps", 1)),
        save_steps=int(g.get("save_steps", 100)),
        bf16=bool(g.get("bf16", False)),
        fp16=bool(g.get("fp16", False)),
        num_generations=int(g["num_generations"]),
        max_completion_length=int(g["max_completion_length"]),
        temperature=float(g.get("temperature", 0.7)),
        top_p=float(g.get("top_p", 0.95)),
        beta=float(g.get("beta", 0.0)),
        loss_type=str(g.get("loss_type", "grpo")),
        use_vllm=False,  # custom HTTP rollout_func → vllm-metal workers
        report_to=["none"],
        remove_unused_columns=False,  # keep `answer` for the reward fn
        max_steps=int(args.max_steps) if args.max_steps is not None else -1,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=gsm8k_reward,
        args=grpo_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        rollout_func=make_rollout_func(cfg),
    )

    # Seed workers with the initial (random) adapter so model=lora.name resolves.
    if not args.skip_initial_sync:
        local = Path(cfg["sync"]["local_adapter_dir"])
        local.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(local)
        try:
            sync_workers(local, cfg)
        except Exception:  # noqa: BLE001
            logger.exception(
                "initial weight sync failed — start workers with --enable-lora "
                "and VLLM_ALLOW_RUNTIME_LORA_UPDATING=1, then retry"
            )
            raise

    trainer.add_callback(WeightSyncCallback(cfg, trainer.model))
    trainer.train()
    trainer.save_model(str(out / "final"))
    logger.info("done → %s", out / "final")


if __name__ == "__main__":
    main()
