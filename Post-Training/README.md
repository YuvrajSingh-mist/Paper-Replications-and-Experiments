# Post-Training Experiments

Fine-tuning and post-training runs: SFT, GRPO, DPO/ORPO variants, and whatever comes next.

## What belongs here, and what doesn't

The rest of this repo keeps one promise: one folder per paper, reimplemented. `DPO/`, `ORPO/` and `SimplePO/` hold up their end of that bargain.

This folder is different. It holds the runs where the point is a hypothesis of my own, not a paper's results:

- `sft/` - supervised fine-tuning recipes, replication-style.
- `grpo/` - GRPO / RL-style post-training recipes, replication-style.
- `runs/` - my own experiments on my own data. Not replications. Don't cite these as a paper's result.

Each run gets its own subfolder with a `config.yaml`, a `results.md` and the metrics it produced. Checkpoints and logs stay out of git (see `.gitignore`).

## Run naming

```
runs/<method>-<model>-<dataset>-<YYYY-MM-DD>/
```

For example `runs/grpo-qwen3-0.6b-gsm8k-2026-11-03/`. The date is the day the run started, so two attempts at the same thing sort chronologically.

## Tagging what ran

A branch per experiment rots. A tag doesn't. Once a run produces numbers worth keeping, tag the commit it ran from:

```bash
git tag -a exp/grpo-qwen3-0.6b-gsm8k-2026-11-03 -m "GRPO, 2 epochs, 3k steps, val acc 0.62"
git push origin exp/grpo-qwen3-0.6b-gsm8k-2026-11-03
```

Then `git log exp/grpo-qwen3-0.6b-gsm8k-2026-11-03` reproduces exactly what the numbers came from.

Branches are for shared code only: a new trainer or loss that more than one run will use, landed on `master` through a PR. If a branch would live longer than two weeks, it should be a folder instead.

## Results format

`results.md` per run, same shape as the replication folders:

| Field | Value |
|-------|-------|
| Base model | |
| Dataset | |
| Method | |
| Hardware | |
| Steps / epochs | |
| Final train loss | |
| Final val loss / metric | |
| Notes | what I was actually testing |
