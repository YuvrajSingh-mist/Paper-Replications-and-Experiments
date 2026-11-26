"""Push LoRA adapters to vllm-metal workers (rsync + /v1/load_lora_adapter)."""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import requests
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


def _expand_remote(path: str) -> str:
    # Keep ~ for the remote shell; only normalize double slashes.
    return path.replace("//", "/")


def rsync_adapter(local_dir: Path, user: str, host: str, remote_dir: str) -> None:
    remote = f"{user}@{host}:{_expand_remote(remote_dir)}/"
    # Ensure remote parent exists.
    subprocess.run(
        ["ssh", f"{user}@{host}", f"mkdir -p {_expand_remote(remote_dir)}"],
        check=True,
        timeout=60,
    )
    cmd = [
        "rsync",
        "-az",
        "--delete",
        f"{str(local_dir).rstrip('/')}/",
        remote,
    ]
    logger.info("rsync %s → %s", local_dir, remote)
    subprocess.run(cmd, check=True, timeout=300)


def wait_healthy(url: str, path: str, retries: int, interval_s: float) -> None:
    health = f"{url.rstrip('/')}{path}"
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(health, timeout=5)
            if r.status_code < 500:
                return
        except Exception as e:  # noqa: BLE001 — poll until up
            last_err = e
        time.sleep(interval_s)
        logger.info("health %s attempt %d/%d", health, attempt, retries)
    raise RuntimeError(f"worker not healthy: {health} ({last_err})")


def load_lora_adapter(url: str, lora_name: str, remote_adapter_dir: str) -> None:
    """vLLM / vllm-metal runtime LoRA reload (needs VLLM_ALLOW_RUNTIME_LORA_UPDATING=1)."""
    endpoint = f"{url.rstrip('/')}/v1/load_lora_adapter"
    payload = {
        "lora_name": lora_name,
        "lora_path": _expand_remote(remote_adapter_dir),
        "load_exists": True,
    }
    logger.info("load_lora_adapter %s %s", endpoint, payload)
    r = requests.post(endpoint, json=payload, timeout=120)
    r.raise_for_status()


def maybe_restart_worker(user: str, host: str, restart_cmd: str | None) -> None:
    if not restart_cmd:
        return
    logger.warning("falling back to ssh restart on %s", host)
    subprocess.run(["ssh", f"{user}@{host}", restart_cmd], check=True, timeout=180)


def sync_workers(local_adapter_dir: Path, cfg: dict[str, Any]) -> None:
    sync_cfg = cfg["sync"]
    lora_name = cfg["lora"]["name"]
    for worker in cfg["workers"]:
        rsync_adapter(
            local_adapter_dir,
            worker["user"],
            worker["host"],
            worker["remote_adapter_dir"],
        )
        try:
            load_lora_adapter(worker["url"], lora_name, worker["remote_adapter_dir"])
        except Exception as e:  # noqa: BLE001
            logger.exception("load_lora_adapter failed on %s: %s", worker["url"], e)
            maybe_restart_worker(worker["user"], worker["host"], sync_cfg.get("restart_cmd"))
            wait_healthy(
                worker["url"],
                sync_cfg.get("health_path", "/health"),
                int(sync_cfg.get("health_retries", 30)),
                float(sync_cfg.get("health_interval_s", 2)),
            )
            load_lora_adapter(worker["url"], lora_name, worker["remote_adapter_dir"])


class WeightSyncCallback(TrainerCallback):
    """Save PEFT adapter every N steps, rsync, reload on vllm-metal workers."""

    def __init__(self, cfg: dict[str, Any], model: Any):
        self.cfg = cfg
        self.model = model
        self.every = int(cfg["sync"].get("every_steps", 0))
        self.local_dir = Path(cfg["sync"]["local_adapter_dir"])

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        if self.every <= 0 or state.global_step <= 0:
            return control
        if state.global_step % self.every != 0:
            return control
        self.local_dir.mkdir(parents=True, exist_ok=True)
        # PEFT / transformers both expose save_pretrained on the wrapped model.
        unwrapped = self.model
        if hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        logger.info("weight sync at step %d → %s", state.global_step, self.local_dir)
        unwrapped.save_pretrained(self.local_dir)
        sync_workers(self.local_dir, self.cfg)
        return control
