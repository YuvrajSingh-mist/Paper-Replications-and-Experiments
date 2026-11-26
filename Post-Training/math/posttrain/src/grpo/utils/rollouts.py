"""HTTP rollouts against vllm-metal OpenAI /v1/completions (multi-worker fan-out)."""

from __future__ import annotations

import itertools
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests

logger = logging.getLogger(__name__)


def _render_prompt(prompt: Any, tokenizer: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list) and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    raise TypeError(f"Unsupported prompt type: {type(prompt)!r}")


def _sampled_logprobs(choice: dict[str, Any]) -> list[float]:
    """Extract per-token logprob of the sampled token from an OpenAI-style choice."""
    lp = choice.get("logprobs") or {}
    token_lps = lp.get("token_logprobs")
    if token_lps is not None:
        return [float(x) if x is not None else 0.0 for x in token_lps]
    content = lp.get("content") or []
    out: list[float] = []
    for step in content:
        if isinstance(step, dict) and "logprob" in step:
            out.append(float(step["logprob"]))
    return out


def _completion_token_ids(choice: dict[str, Any], tokenizer: Any, text: str) -> list[int]:
    if choice.get("token_ids") is not None:
        return [int(t) for t in choice["token_ids"]]
    lp = choice.get("logprobs") or {}
    tokens = lp.get("tokens")
    if tokens:
        ids: list[int] = []
        for tok in tokens:
            # vLLM may prefix token-id strings as "token_id:1234"
            if isinstance(tok, str) and tok.startswith("token_id:"):
                ids.append(int(tok.split(":", 1)[1]))
                continue
            enc = tokenizer.encode(tok, add_special_tokens=False)
            ids.extend(enc)
        if ids:
            return ids
    return tokenizer.encode(text, add_special_tokens=False)


def _post_completion(url: str, path: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    resp = requests.post(f"{url.rstrip('/')}{path}", json=payload, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def generate_rollouts(
    prompts: list[Any],
    trainer: Any,
    *,
    worker_urls: list[str],
    lora_name: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    logprobs: int,
    completion_path: str,
    request_timeout_s: float,
    max_workers: int,
) -> dict[str, Any]:
    """TRL `rollout_func` body.

    GRPOTrainer's RepeatSampler already repeats each unique prompt
    ``num_generations`` times in ``prompts``. Emit **one** completion per
    entry so lengths line up with ``inputs`` / reward columns and with
    ``rewards.view(-1, num_generations)``.
    """
    tokenizer = trainer.processing_class
    rendered = [_render_prompt(p, tokenizer) for p in prompts]
    prompt_ids = [tokenizer.encode(t, add_special_tokens=False) for t in rendered]

    url_cycle = itertools.cycle(worker_urls)
    jobs = [(idx, text, next(url_cycle)) for idx, text in enumerate(rendered)]
    results: dict[int, tuple[list[int], list[float]]] = {}

    def _one(job: tuple[int, str, str]) -> tuple[int, list[int], list[float]]:
        idx, text, url = job
        payload = {
            "model": lora_name,
            "prompt": text,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": 1,
            "logprobs": logprobs,
            "return_tokens_as_token_ids": True,
        }
        data = _post_completion(url, completion_path, payload, request_timeout_s)
        choice = data["choices"][0]
        text_out = choice.get("text") or ""
        cids = _completion_token_ids(choice, tokenizer, text_out)
        lps = _sampled_logprobs(choice)
        if len(lps) < len(cids):
            lps = lps + [0.0] * (len(cids) - len(lps))
        elif len(lps) > len(cids):
            lps = lps[: len(cids)]
        return idx, cids, lps

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_one, job) for job in jobs]
        for fut in as_completed(futs):
            idx, cids, lps = fut.result()
            results[idx] = (cids, lps)

    out_completion_ids = [results[i][0] for i in range(len(prompts))]
    out_logprobs = [results[i][1] for i in range(len(prompts))]

    logger.info("rollouts: %d completions via %d workers", len(prompts), len(worker_urls))
    return {
        "prompt_ids": prompt_ids,
        "completion_ids": out_completion_ids,
        "logprobs": out_logprobs,
    }


def make_rollout_func(cfg: dict[str, Any]):
    """Build a TRL-compatible rollout_func closed over config."""
    urls = [w["url"] for w in cfg["workers"]]
    roll = cfg["rollouts"]
    grpo = cfg["grpo"]
    lora_name = cfg["lora"]["name"]

    def rollout_func(prompts: list[Any], trainer: Any) -> dict[str, Any]:
        return generate_rollouts(
            prompts,
            trainer,
            worker_urls=urls,
            lora_name=lora_name,
            max_tokens=int(grpo["max_completion_length"]),
            temperature=float(grpo.get("temperature", 0.7)),
            top_p=float(grpo.get("top_p", 0.95)),
            logprobs=int(roll.get("logprobs", 1)),
            completion_path=str(roll.get("completion_path", "/v1/completions")),
            request_timeout_s=float(roll.get("request_timeout_s", 120)),
            max_workers=int(roll.get("max_workers", 8)),
        )

    return rollout_func
