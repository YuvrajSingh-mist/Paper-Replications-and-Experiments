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

## Installing

```bash
pip install -r requirements.txt       # runtime
pip install -r requirements-dev.txt   # ruff, pytest, pre-commit
```

Pins in `requirements.txt` were resolved on 2026-11-04 and have **not** been
verified together on a CUDA machine. Treat them as a starting point.

On Linux with a specific CUDA build, install torch from the matching index
first, then the rest:

```bash
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Watch the import name. Several distributions differ from the module you import,
and one of them fails dangerously: `from muon import MuonWithAuxAdam` needs
`muon-optimizer`, **not** `muon` — which exists on PyPI and installs an
unrelated omics framework. `tests/test_requirements.py` guards this mapping.

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
| Ruff | `ruff check .` | `E9`, `F63`, `F7`, `F82`, `F401` — see below |
| Format | `ruff format --check .` | anything `ruff format .` would change |
| Import smoke tests | `python -m pytest tests -q` | modules that cannot be imported |

Run all of them locally with:

```bash
ruff check . && ruff format --check . && \
  python -m pytest tests -q
```

## Why these checks exist

Every one of them corresponds to a bug that actually reached `master`:

- **Lint (`F82`).** A module imported a helper from a file that did not exist,
  and six trainer classes referenced helpers that were never in scope.
- **Lint (`F401`).** An unused import of a module that no longer exists still
  raises `ImportError` at runtime, and a wall of unused imports makes the one
  that matters impossible to spot.
- **Import tests.** Every bug above was invisible to parsing and linting. CI was
  green while `import config` raised `ZeroDivisionError` on any machine without a
  GPU, and while `SmolMixtral/model.py` demanded Hugging Face credentials just to
  import. Only importing the module catches that.
- **Format.** Not a bug, just consistency: it keeps diffs limited to real
  changes instead of reflowing whatever the previous author's editor produced.

## Scope

`ruff.toml` excludes the following on purpose. Do not "fix" them into scope
without a reason:

- `*.ipynb` — cell order is stateful, which makes undefined-name rules noise.
- `*.md` — reformatting embedded code samples churns READMEs.
- `Replications/CLAP/**` — vendored copy of an upstream repository. Keep it
  byte-close to upstream so it stays diffable; fixes belong upstream.

## Known backlog

- **Redundant re-imports (`F811`).** The same name imported twice in one scope.
  Harmless, but two of them shadow a module-level name inside a function, so this
  needs a read rather than `--fix`. Sweep them, then add `F811` to
  `[lint] select`.

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
