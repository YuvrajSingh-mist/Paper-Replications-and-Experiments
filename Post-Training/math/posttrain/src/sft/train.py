#!/usr/bin/env python3
"""TRL SFT on GSM8K for Qwen (Mac Mini / MPS)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import yaml
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from sft.data import build_gsm8k_sft_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("sft.qwen")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "sft.yaml"


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _resolve_path(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def pick_dtype() -> torch.dtype:
    if torch.backends.mps.is_available():
        return torch.float32
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = _resolve_path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

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

    lora_cfg = cfg.get("lora") or {}
    peft_config = None
    if lora_cfg.get("enabled", True):
        peft_config = LoraConfig(
            r=int(lora_cfg.get("r", 16)),
            lora_alpha=int(lora_cfg.get("alpha", 32)),
            lora_dropout=float(lora_cfg.get("dropout", 0.05)),
            target_modules=list(
                lora_cfg.get(
                    "target_modules",
                    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                )
            ),
            task_type="CAUSAL_LM",
            bias="none",
        )

    data_cfg = cfg["data"]
    train_ds = build_gsm8k_sft_dataset(
        dataset_name=data_cfg.get("dataset_name", "openai/gsm8k"),
        subset=data_cfg.get("subset", "main"),
        split=data_cfg.get("train_split", "train"),
        max_samples=data_cfg.get("max_train_samples"),
    )
    eval_ds = None
    if data_cfg.get("eval_split"):
        eval_ds = build_gsm8k_sft_dataset(
            dataset_name=data_cfg.get("dataset_name", "openai/gsm8k"),
            subset=data_cfg.get("subset", "main"),
            split=data_cfg["eval_split"],
            max_samples=data_cfg.get("max_eval_samples", 256),
        )
    logger.info("train=%d eval=%s", len(train_ds), len(eval_ds) if eval_ds else None)

    t = cfg["train"]
    sft_args = SFTConfig(
        output_dir=str(out),
        run_name=out.name,
        seed=int(cfg.get("seed", 42)),
        learning_rate=float(t["learning_rate"]),
        per_device_train_batch_size=int(t["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(t.get("per_device_eval_batch_size", t["per_device_train_batch_size"])),
        gradient_accumulation_steps=int(t["gradient_accumulation_steps"]),
        num_train_epochs=float(t["num_train_epochs"]),
        logging_steps=int(t.get("logging_steps", 10)),
        save_steps=int(t.get("save_steps", 200)),
        eval_strategy=str(t.get("eval_strategy", "no" if eval_ds is None else "steps")),
        eval_steps=int(t.get("eval_steps", 200)) if eval_ds is not None else None,
        bf16=bool(t.get("bf16", False)),
        fp16=bool(t.get("fp16", False)),
        max_length=int(t.get("max_length", 1024)),
        packing=bool(t.get("packing", False)),
        assistant_only_loss=bool(t.get("assistant_only_loss", True)),
        report_to=["none"],
        max_steps=int(args.max_steps) if args.max_steps is not None else -1,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    final_dir = out / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info("done → %s", final_dir)


if __name__ == "__main__":
    main()
