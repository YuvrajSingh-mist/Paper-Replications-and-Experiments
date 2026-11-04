#!/usr/bin/env python3
"""Parse every Python source file and report all syntax errors at once.

`python -m compileall` stops being useful here because it writes bytecode, and
because a single bad file hides the rest. This walks the tree with `ast.parse`,
collects every failure, and exits non-zero if there are any.

The repo has carried a file with an unmatched closing paren for its entire
existence (SmolHub/SmolMixtral/old/mixtral_final_v2.py), because nothing ever
parsed it. This is the check that would have caught it.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints"}


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else ".").resolve()
    failures: list[tuple[Path, SyntaxError]] = []
    checked = 0

    for path in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        checked += 1
        try:
            ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
        except SyntaxError as exc:
            failures.append((path.relative_to(root), exc))

    if failures:
        print(f"\n{len(failures)} file(s) failed to parse:\n")
        for rel, exc in failures:
            print(f"  {rel}:{exc.lineno}: {exc.msg}")
        return 1

    print(f"All {checked} Python files parsed cleanly on {sys.version.split()[0]}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
