"""Static guards for module-scope code that only works on a GPU machine.

The Replications scripts cannot be imported in tests: they download weights and
datasets at module scope, so importing them needs a GPU, a token, and gigabytes
of disk. These checks inspect the syntax tree instead, which is enough to catch
the one pattern that reliably broke this repo:

    gradient_accumulation_steps = total_batch_size // (micro_batch_size * block_size * torch.cuda.device_count())

That is a top-level assignment, so it runs at import time. With no visible CUDA
device the divisor is 0 and the module raises ZeroDivisionError before any of its
own code executes -- on any CPU machine, including CI. Wrapping the count in
max(1, ...) makes it defined everywhere without changing GPU behavior.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

EXCLUDED_PARTS = {"old", "__pycache__", ".git", ".ipynb_checkpoints"}
EXCLUDED_DIRS = ("Replications/CLAP",)  # vendored upstream, kept byte-close


def _python_files() -> list[pathlib.Path]:
    files = []
    for path in REPO_ROOT.rglob("*.py"):
        rel = path.relative_to(REPO_ROOT)
        if EXCLUDED_PARTS & set(rel.parts):
            continue
        if any(str(rel).startswith(d) for d in EXCLUDED_DIRS):
            continue
        files.append(path)
    return sorted(files)


def _find_unguarded_device_count(node: ast.AST) -> list[int]:
    """Line numbers where device_count() is called without a max() guard.

    Walks the tree tracking whether the current node sits inside a call to
    max(...), and reports device_count() only when it does not.
    """
    found: list[int] = []

    def walk(current: ast.AST, guarded: bool) -> None:
        if isinstance(current, ast.Call):
            func = current.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name == "device_count" and not guarded:
                found.append(current.lineno)
            inside_max = guarded or (isinstance(func, ast.Name) and func.id == "max")
            for arg in current.args:
                walk(arg, inside_max)
            return
        for child in ast.iter_child_nodes(current):
            walk(child, guarded)

    walk(node, False)
    return found


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_module_scope_device_count_is_guarded(path: pathlib.Path) -> None:
    """No import-time division by torch.cuda.device_count() without max(1, ...)."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines()

    offenders: list[str] = []
    for statement in tree.body:
        # Only module-scope statements run on import.
        if not isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Expr, ast.If)):
            continue
        for lineno in _find_unguarded_device_count(statement):
            line = lines[lineno - 1]
            # A bare call, e.g. `world_size = torch.cuda.device_count()`, is fine:
            # it returns 0, it does not divide by it.
            if "/" in line:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")

    assert not offenders, "module-scope division by device_count() without max(1, ...):\n" + "\n".join(offenders)


def test_no_placeholder_hf_token_sent_to_the_hub() -> None:
    """The literal '...' was passed as an access token, so importing did a bad request."""
    # Built from parts so this file does not match its own check.
    placeholder = "hf_token=" + "'" + "." * 3 + "'"
    offenders = []
    for path in _python_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if placeholder in line:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}")
    assert not offenders, "placeholder token passed to from_pretrained():\n" + "\n".join(offenders)


def test_guard_would_have_caught_the_original_bug() -> None:
    """The detector must actually fire on the pattern it exists to catch."""
    buggy = ast.parse(
        "gradient_accumulation_steps = total_batch_size // (micro_batch_size * (block_size * torch.cuda.device_count()))"
    )
    fixed = ast.parse(
        "gradient_accumulation_steps = total_batch_size // (micro_batch_size * (block_size * max(1, torch.cuda.device_count())))"
    )
    assert _find_unguarded_device_count(buggy.body[0])
    assert not _find_unguarded_device_count(fixed.body[0])
