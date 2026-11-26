# Math post-training (Mac Minis)

Model-agnostic SFT / GRPO / LightEval. Set the HF id (or local ckpt) in configs or CLI; result folders use the model basename.

| Stage | Tooling | Entry |
| --- | --- | --- |
| **SFT** on GSM8K | TRL `SFTTrainer` + PEFT LoRA | `scripts/train_sft.sh` |
| **GRPO** | TRL `GRPOTrainer` + vllm-metal HTTP rollouts | `scripts/train_grpo.sh` |
| **Eval** | Hugging Face LightEval | `./evals.sh` / `scripts/eval_instruct_split.sh` |

```
configs/sft.yaml | grpo.yaml | eval.yaml
src/sft/                   TRL SFT
src/grpo/                  GRPO + rollouts / sync / rewards
src/math_eval/             LightEval wrapper
eval/suites/               instruct / base / instruct_full task lists
scripts/                   train_sft / train_grpo / eval / eval_instruct_split
evals.sh                   model-path entrypoint
setup.sh                   uv sync (+ optional vllm-metal worker)
```

## Setup

```bash
cd Post-Training/math/posttrain
./setup.sh
./setup.sh --worker        # GRPO rollout machines only
```

## Eval

```bash
# split instruct suite (hellaswag | ifeval+gsm8k | arc+piqa), saves details
./scripts/eval_instruct_split.sh HuggingFaceTB/SmolLM-360M-Instruct --save-details

./evals.sh HuggingFaceTB/SmolLM-360M-Instruct
SUITE=math ./evals.sh HuggingFaceTB/SmolLM-360M-Instruct
./evals.sh HuggingFaceTB/SmolLM-360M-Instruct --limit 8
```

Outputs land under `outputs/lighteval/<model-basename>/…`.

## Worklog

See [`WORKLOG.md`](./WORKLOG.md).
