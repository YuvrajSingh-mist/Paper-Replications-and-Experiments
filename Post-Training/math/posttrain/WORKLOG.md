# Worklog: Math post-training on Mac Minis (GSM8K)

> Living notes for a [style of write-up like Anshuman’s GSM8K worklog](https://heyyanshuman.com/posts/posttraining_gsm8k_worklog/).  
> Keep this file updated as we go; later we can turn sections into tables/figures for a public post.

**Baseline model (current):** `HuggingFaceTB/SmolLM-360M-Instruct` (set in configs / CLI — code is model-agnostic).  
**Task:** GSM8K grade-school math with explicit reasoning tags.  
**Hardware:** Apple Silicon Mac Minis — trainer on MPS (TRL + PEFT); GRPO rollouts on sibling Minis via **vllm-metal** HTTP + LoRA sync.  
**Code root:** `Post-Training/math/posttrain/`

---

## On this page

- [Problem & format](#problem--format)
- [Stack & layout](#stack--layout)
- [Baselines first](#baselines-first)
- [GSM8K data / splits / lengths](#gsm8k-data--splits--lengths)
- [Eval harness choices](#eval-harness-choices)
- [Getting training to run at all](#getting-training-to-run-at-all) *(planned)*
- [SFT warm-up](#sft-warm-up) *(next)*
- [Naive rewards: correctness and format](#naive-rewards-correctness-and-format) *(planned)*
- [Filtering 0/k and k/k groups](#filtering-0k-and-kk-groups) *(planned)*
- [Presampling / pass@k](#presampling--passk) *(planned)*
- [GRPO on Mac Minis](#grpo-on-mac-minis) *(planned)*
- [Where this leaves us](#where-this-leaves-us)
- [Runs referenced](#runs-referenced)

---

## Problem & format

For a GSM8K question we want the model to answer roughly like:

```text
<think>
…step-by-step working…
</think>
<answer>42</answer>
```

System prompt (SFT / GRPO data path — `src/sft/data.py`):

```text
You are an assistant expert at solving math problems.
Think step by step inside <think>...</think>, then put the final numeric
answer inside <answer>...</answer>.
```

Gold GSM8K strings are split on `#### <number>`; reasoning goes in `<think>`, final in `<answer>`.

| Check (to fill after baselines) | Qwen2.5-0.5B-Instruct | Notes |
| --- | --- | --- |
| right number anywhere | TBD | lenient |
| right number inside `<answer>` | TBD | strict parse |
| exact tag format | TBD | `<think>` + `<answer>` |
| right answer **and** exact format | TBD | train target |

---

## Stack & layout

| Stage | Tooling | Entry |
| --- | --- | --- |
| Env | `uv` + `./setup.sh` | trainer / `--worker` for vllm-metal |
| SFT | TRL `SFTTrainer` + PEFT LoRA | `scripts/train_sft.sh` · `configs/sft.yaml` |
| GRPO | TRL `GRPOTrainer` + HTTP rollouts + LoRA rsync/reload | `scripts/train_grpo.sh` · `configs/grpo.yaml` |
| Eval | Hugging Face **LightEval** | `./evals.sh` / `scripts/eval_instruct_split.sh` · `configs/eval.yaml` |

```
configs/     sft.yaml | grpo.yaml | eval.yaml
src/sft/               TRL SFT
src/grpo/              GRPO + rollouts / sync / rewards
src/eval/              LightEval wrapper (`uv run eval`)
suites/                instruct / base / instruct_full + piqa_hf
evals.sh               sole LightEval shell entry
scripts/eval_instruct_split.sh   hellaswag | gen | mc shards
WORKLOG.md             this file
```

**Decisions locked in so far**

- **Branch = experiment** (e.g. `smollm-gsm8k`); template code stays generic. Model only in `configs/*.yaml` / CLI and `outputs/…/<model-basename>/`.
- Not using smolcluster; trainer ↔ workers = TRL + OpenAI-compatible HTTP + adapter rsync + `/v1/load_lora_adapter`.
- This branch baseline = **SmolLM-360M-Instruct** (v1) + GSM8K.
- LightEval (not lm-eval); default suite drops BBH + MMLU-Pro. Optional `SUITE=instruct_full`. Care most about **GSM8K**.
- Do **not** fork LightEval; thin `src/eval` wrapper + official `model_args` + parquet `piqa_hf`.
- One shell entry `./evals.sh` (no duplicate `scripts/eval.sh`). Split runner for MPS OOM avoidance.
- LightEval logprobs `batch_size` is flattened; slim suite uses `batch_size: 32` in `eval.yaml`.

---

## Baselines first

Goal: baseline **SmolLM v1 360M-Instruct** (older than SmolLM2) before our SFT/GRPO — story is RL / math post-train on a small older instruct model, not chasing SmolLM2 card parity.

### Our LightEval run (in progress)

| Field | Value |
| --- | --- |
| Model | `HuggingFaceTB/SmolLM-360M-Instruct` |
| Suite | instruct (fast): `ifeval`, `hellaswag`, `arc`, `piqa_hf`, `gsm8k\|5` — no BBH / MMLU-Pro |
| Device | MPS, LightEval `batch_size=32` (≈8×4), `max_length=2048` |
| Log | `outputs/lighteval/SmolLM-360M-Instruct/run.log` |
| Status | Clean full split in `tmux` `lighteval-split` |
| Layout | `…/SmolLM-360M-Instruct/{hellaswag,gen,mc}/` → `results/` + `details/` |
| Results | TBD → paste table here when done |

### Reward-format checks (post SFT/GRPO)

Same checks as the reward parser (`src/grpo/utils/rewards.py`):

| Checks | Score | Run / notes |
| --- | --- | --- |
| right number anywhere | TBD | |
| right number in `<answer>` | TBD | |
| exact tag format | TBD | |
| answer **and** format | TBD | |

Also worth a **lenient** vs **strict** column like the reference worklog (format bugs hide real math).

---

## GSM8K data / splits / lengths

| Split | Size | Role |
| --- | --- | --- |
| `train` | 7,473 | SFT / GRPO / few-shot exemplars |
| `test` | 1,319 | Held-out eval (no official val split) |

**Contamination / recipe notes**

- Eval always on **test**; 5-shot draws from **train** (standard).
- SmolLM2-**360M**-Instruct SFT data (`smol-smoltalk`) excludes advanced math — unlikely trained on GSM8K train.
- FineMath pretrain claimed decontam vs GSM8K; **1.7B** SmolTalk path is different (MetaMath etc.).

**Token lengths** (Qwen2.5 tokenizer, train set) for `<think>…</think><answer>…</answer>` assistants:

| | mean | median | p95 | max |
| --- | --- | --- | --- | --- |
| reasoning only | ~117 | ~106 | ~223 | ~445 |
| full assistant | ~132 | ~121 | ~238 | ~463 |
| full chat seq | ~240 | ~227 | ~371 | ~595 |

**Config choice:** `max_new_tokens` / `max_completion_length` = **512** (covers assistant max ~463). SFT `max_length` = **512** for packed chat seq (truncates a few full-seq outliers near ~595).

---

## Eval harness choices

| Attempt | Harness | Outcome |
| --- | --- | --- |
| 1 | lm-eval Open LLM Leaderboard v1 tasks | Too old vs SmolLM2 reporting; cancelled mid-run |
| 2 | LightEval 0.13 + built-in SmolLM2-like tasks | Adopted; `./evals.sh <model>` |
| — | Output dirs | Use model basename only (`outputs/lighteval/SmolLM2-360M-Instruct/`), not `…-instruct` suffix |

Smoke / full:

```bash
./evals.sh HuggingFaceTB/SmolLM2-360M-Instruct --limit 8
./evals.sh HuggingFaceTB/SmolLM2-360M-Instruct
SUITE=math ./evals.sh Qwen/Qwen2.5-0.5B-Instruct   # gsm8k|5
```

---

## Getting training to run at all

*(Fill like the reference worklog: first OOM / MPS quirks, batch size, LoRA targets, LR.)*

- [ ] Smoke SFT: `bash scripts/train_sft.sh --max-steps 2`
- [ ] Full SFT on GSM8K train
- [ ] Smoke GRPO with workers up: `bash scripts/train_grpo.sh --max-steps 1 --skip-initial-sync`
- [ ] Note any MPS / vllm-metal / LoRA reload failures here

---

## SFT warm-up

*(Next major step.)*

| Field | Value |
| --- | --- |
| Config | `configs/sft.yaml` |
| Base | `Qwen/Qwen2.5-0.5B-Instruct` |
| Data | GSM8K train → chat messages w/ `<think>` / `<answer>` |
| LoRA | r=16, α=32, all major linear modules |
| LR / epochs | 2e-4 / 1 |
| `max_length` | 512 |
| Output | `outputs/sft-qwen25-05b-gsm8k/final` |
| Post-SFT GSM8K | TBD (strict / lenient table) |

---

## Naive rewards: correctness and format

Current reward (`gsm8k_reward` in `src/grpo/utils/rewards.py`): correctness from `<answer>` (else last number) + light format shaping for tags.

| Reward | Weight / rule | Status |
| --- | --- | --- |
| correctness | 1 if parsed answer matches gold | implemented |
| format / think | partial credit for tags | implemented (light) |
| lenient correctness | right number anywhere | TBD (reference worklog pattern) |
| tag_check partial | per-tag credit | TBD |
| exact (answer ∧ format) | higher weight | TBD |

*(After first GRPO run: paste train curves + % of rollouts missing `<answer>`.)*

---

## Filtering 0/k and k/k groups

*(Planned — same motivation as reference: GRPO advantage is ~0 when all samples share the same reward.)*

- [ ] Log % of groups that are 0/k and k/k on first GRPO run
- [ ] Filter train questions with no within-group variance
- [ ] Optionally shrink `num_generations` (config: 4) to cover more prompts/step

---

## Presampling / pass@k

*(Planned.)*

- [ ] Generate k samples per train question before RL
- [ ] Drop always-wrong / always-right under **lenient** and **strict** scoring
- [ ] Record pass@k table for base vs SFT checkpoint

---

## GRPO on Mac Minis

| Field | Value |
| --- | --- |
| Config | `configs/grpo.yaml` |
| Rollouts | vllm-metal workers (HTTP `/v1/completions`) |
| Sync | save LoRA → rsync → `/v1/load_lora_adapter` every N steps |
| `max_completion_length` | 512 |
| `num_generations` | 4 |
| Loss | `loss_type: grpo` |

*(Log worker sync bugs, importance-sampling ratios, entropy collapse, etc. here as they appear — mirror reference sections.)*

---

## Where this leaves us

**Done**

- [x] Repo layout: SFT + GRPO + LightEval on Mac Minis
- [x] GSM8K conversational SFT data + reward helpers
- [x] LightEval entrypoint; SmolLM2-style instruct suite chosen
- [x] Token-length study → generation cap **512**
- [x] Worklog started (this file)

**In flight**

- [ ] Finish SmolLM-360M-Instruct LightEval → fill baseline table

**Next**

- [ ] SFT warm-up on GSM8K (SmolLM-360M-Instruct)
- [ ] First GRPO run + rollout autopsy (format vs math)
- [ ] Presample / filter empty-advantage groups
- [ ] Iterate rewards; compare to reference worklog lessons

---

## Runs referenced

| ID / folder | Stage | Model | What | Outcome |
| --- | --- | --- | --- | --- |
| `outputs/lighteval/SmolLM-360M-Instruct/` | eval | SmolLM-360M-Instruct | LightEval instruct (slim) | running |
| — | SFT | SmolLM-360M-Instruct | GSM8K LoRA | not started |
| — | GRPO | post-SFT | first correctness+format | not started |

---

## Changelog (append-only)

| Date | Note |
| --- | --- |
| 2026-11-24 | Scaffolded qwen math post-train (SFT/GRPO/eval); switched eval to LightEval; measured GSM8K lengths; set max gen 512; started SmolLM2 instruct baseline eval; opened this worklog. |
| 2026-11-25 | Dropped BBH+MMLU-Pro from default suite; switched target to SmolLM-360M-Instruct (not SmolLM2); LightEval `batch_size: 32` in yaml (removed flat-batch helper); eval in `tmux` `lighteval-smollm`. |
| 2026-11-25 | Slim eval finished inference (~11h) then died in metrics: `xxhash` 4.x needs bytes. Pin `xxhash>=3.4,<4`; HF response cache kept — restart should score without re-running model. |
| 2026-11-25 | E2E smoke `OUT=…-smoke ./evals.sh … --limit 8` → results JSON + table, `EXIT:0`. Then full `lighteval-smollm` restarted. |
| 2026-11-25 | `save_details: true`; wiped smoke + prior full out; restarted full with `--save-details`. |
| 2026-11-26 | Full suite OOM’d on bootstrap; added `scripts/eval_instruct_split.sh` (no CLI change). Per-task smoke OK; started grouped full split. |
| 2026-11-26 | Renamed `eval/smollm2`→`eval/suites`, `qwen_eval`→`smol_eval` (`smol-eval` entry; `qwen-math-eval` alias). |
| 2026-11-26 | Model-agnostic rename: `qwen/`→`posttrain/`, packages `sft`/`grpo`/`math_eval`, entry `math-eval`; outputs still use model basename. Restarted split eval. |
| 2026-11-26 | `math_eval`→`eval`, task lists→`suites/`, drop `scripts/eval.sh`; branch-per-experiment (`smollm-gsm8k`). |
