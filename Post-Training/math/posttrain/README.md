# Post-training template (Mac Minis)

One codebase, **one branch per experiment**. Configs + model id change on the branch; package names stay generic (`sft` / `grpo` / `eval`).

| Current branch (example) | Model / focus |
| --- | --- |
| `smollm-gsm8k` | `HuggingFaceTB/SmolLM-360M-Instruct` + GSM8K SFT/GRPO |

| Stage | Tooling | Entry |
| --- | --- | --- |
| **SFT** | TRL `SFTTrainer` + PEFT LoRA | `scripts/train_sft.sh` |
| **GRPO** | TRL `GRPOTrainer` + vllm-metal rollouts | `scripts/train_grpo.sh` |
| **Eval** | LightEval | `./evals.sh` · `scripts/eval_instruct_split.sh` |

```
configs/          sft.yaml | grpo.yaml | eval.yaml   ← model lives here
src/sft/          TRL SFT
src/grpo/         GRPO + rollouts / sync / rewards
src/eval/         LightEval wrapper (CLI: `uv run eval`)
suites/           instruct / base / instruct_full task lists
evals.sh          single entry for LightEval
scripts/          train_sft / train_grpo / eval_instruct_split / serve_worker
```

## New experiment branch

```bash
git checkout main   # or master / template tip
git checkout -b <model>-<task>    # e.g. qwen25-gsm8k
# edit configs/*.yaml model_name / model:
./setup.sh
```

Result folders stay model-named: `outputs/lighteval/<model-basename>/…`.

## Setup

```bash
cd Post-Training/math/posttrain
./setup.sh
./setup.sh --worker   # GRPO rollout machines only
```

## Eval

```bash
# full instruct suite, split shards (recommended on MPS)
./scripts/eval_instruct_split.sh HuggingFaceTB/SmolLM-360M-Instruct --save-details

./evals.sh HuggingFaceTB/SmolLM-360M-Instruct
SUITE=math ./evals.sh HuggingFaceTB/SmolLM-360M-Instruct
./evals.sh HuggingFaceTB/SmolLM-360M-Instruct --limit 8
```

## Worklog

See [`WORKLOG.md`](./WORKLOG.md).
