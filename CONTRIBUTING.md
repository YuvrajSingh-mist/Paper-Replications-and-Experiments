# Contributing

## Before you push

Two things are mandatory, and CI blocks the pull request if either fails:

```bash
ruff format .      # must be applied
ruff check .       # must pass
```

That is the whole requirement. Everything else is optional.

If `ruff` is not installed, either install it (`pip install ruff==0.17.3`) or
run it without installing:

```bash
uvx ruff@0.17.3 format .
uvx ruff@0.17.3 check .
```

The version is pinned in `.github/workflows/ci.yml` and
`.pre-commit-config.yaml`. Formatting output can change between ruff releases,
so the three must stay in step. Bump them together, in a commit that also
reformats the tree — otherwise every open branch conflicts.

## Automate it instead

```bash
pip install pre-commit
pre-commit install
```

From then on `git commit` runs lint and format for you and aborts the commit if
either fails. To check the whole tree without committing:

```bash
pre-commit run --all-files
```

## What CI checks

| Job | Command | What it catches |
| --- | --- | --- |
| Ruff | `ruff check .` | `E9`, `F63`, `F7`, `F82` — see below |
| Format | `ruff format --check .` | anything `ruff format .` would change |
| Parse (3.12, 3.13) | `python tools/check_syntax.py .` | files that do not parse |
| Removed stdlib | `python tools/check_removed_stdlib.py .` | imports of modules Python deleted |

Run all four locally with:

```bash
ruff check . && ruff format --check . && \
  python tools/check_syntax.py . && python tools/check_removed_stdlib.py .
```

## Why these checks exist

Every one of them corresponds to a bug that actually reached `master`:

- **Removed stdlib.** `from uu import encode` sat in a training script for a
  year. It was valid syntax on every version, so nothing caught it — until
  Python 3.13 deleted `uu` and the script died before importing torch. A syntax
  check cannot see this; only resolving the name against the interpreter can.
- **Parse.** One script had an unmatched `)` and had never parsed in any commit,
  including the merge that introduced it.
- **Lint (`F82`).** A module imported a helper from a file that did not exist,
  and six trainer classes referenced helpers that were never in scope.
- **Format.** Not a bug, just consistency: it keeps diffs limited to real
  changes instead of reflowing whatever the previous author's editor produced.

## Scope

`ruff.toml` excludes the following on purpose. Do not "fix" them into scope
without a reason:

- `*/old/**` — superseded copies kept for reference. Still parsed by
  `check_syntax.py`, so a broken file cannot hide there.
- `*.ipynb` — cell order is stateful, which makes undefined-name rules noise.
- `*.md` — reformatting embedded code samples churns READMEs.
- `Replications/CLAP/**` — vendored copy of an upstream repository. Keep it
  byte-close to upstream so it stays diffable; fixes belong upstream.

## Known backlog

These are pre-existing and deliberately *not* enforced yet:

- **Unused imports.** Enough of them that enabling `F401` would fail today.
  Sweep them, then add `F401` to `[lint] select`.

## Running these scripts

Two practical notes, learned the hard way:

- **They need a CUDA device to even import.** `config.py` computes a dataclass
  default from `torch.cuda.device_count()`, so on a CPU-only machine importing
  it raises `ZeroDivisionError` before any training code runs.
- **Launch distributed scripts with `torchrun`.** `setup()` calls
  `init_process_group("nccl")` and the trainers read `os.environ["LOCAL_RANK"]`.
  Plain `python trainer.py` will not work.

## Commits and pull requests

- Keep formatting changes in their own commit, separate from behaviour changes,
  so reviewers can skip the noise.
- Do not add an agent or tool as a co-author on commits.
- Branch protection is not enabled on `master`; CI is still expected to be
  green before merging.
