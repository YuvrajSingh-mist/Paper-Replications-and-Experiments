# Post-training template (Mac Minis)

One codebase, **one branch per experiment**. Configs + model id change on the branch; package names stay generic (`sft` / `grpo` / `eval`).

| Current branch (example) | Model / focus |
| --- | --- |
| `smollm-gsm8k` | `HuggingFaceTB/SmolLM-360M-Instruct` + GSM8K SFT/GRPO |

| Stage | Tooling | Entry |
| --- | --- | --- |
| **SFT** | TRL `SFTTrainer` + PEFT LoRA | `scripts/train_sft.sh` |
| **GRPO** | TRL `GRPOTrainer` + vllm-metal rollouts | `scripts/train_grpo.sh` |
| **Eval** | EleutherAI lm-eval | `scripts/evals.sh` · `scripts/eval_instruct_split.sh` |

```
configs/          sft.yaml | grpo.yaml | eval.yaml   ← model lives here
src/sft/          TRL SFT
src/grpo/         GRPO + rollouts / sync / rewards
src/eval/         lm-eval wrapper (CLI: `uv run eval`)
src/eval/suites/  task lists
scripts/          evals / eval_instruct_split / train_sft / train_grpo / serve_worker
setup.sh          env bootstrap (stays at repo root)
```

## New experiment branch

```bash
git checkout main   # or master / template tip
git checkout -b <model>-<task>    # e.g. qwen25-gsm8k
# edit configs/*.yaml model_name / model:
./setup.sh
```

Result folders stay model-named: `outputs/lm_eval/<model-basename>/…`.
Per-sample answers (`--log-samples`) are saved only for **ifeval + gsm8k**.

## Setup

```bash
cd Post-Training/math/posttrain
./setup.sh
./setup.sh --worker   # GRPO rollout machines only
```

## Eval

```bash
# full instruct suite, split shards (recommended on MPS)
./scripts/eval_instruct_split.sh HuggingFaceTB/SmolLM-360M-Instruct

./scripts/evals.sh HuggingFaceTB/SmolLM-360M-Instruct
SUITE=math ./scripts/evals.sh HuggingFaceTB/SmolLM-360M-Instruct
./scripts/evals.sh HuggingFaceTB/SmolLM-360M-Instruct --limit 8
```

## Worklog

See [`WORKLOG.md`](./WORKLOG.md).
