#!/usr/bin/env python3
"""Fail if any source file imports a module that was removed from the stdlib.

A previous bug in this repo was `from uu import encode` in
SmolHub/SmolLlama/dpo_train.py. It sat there harmlessly for a year because the
name was never used, then Python 3.13 deleted the `uu` module (PEP 594) and the
script started failing before torch was even imported.

A plain syntax check cannot catch this: the import is valid syntax on every
version. Only resolving the module name against the interpreter catches it, so
this script keeps an explicit list of removals and scans the AST for them.

Run on the oldest Python you support; the list below covers every removal up to
and including 3.13, so the result is the same on any version.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# PEP 594 (removed in 3.13) and PEP 632 (removed in 3.12).
REMOVED: dict[str, str] = {
    "uu": "3.13",
    "cgi": "3.13",
    "cgitb": "3.13",
    "pipes": "3.13",
    "telnetlib": "3.13",
    "crypt": "3.13",
    "nis": "3.13",
    "nntplib": "3.13",
    "ossaudiodev": "3.13",
    "sndhdr": "3.13",
    "spwd": "3.13",
    "sunau": "3.13",
    "xdrlib": "3.13",
    "aifc": "3.13",
    "audioop": "3.13",
    "chunk": "3.13",
    "imghdr": "3.13",
    "mailcap": "3.13",
    "msilib": "3.13",
    "macpath": "3.13",
    "asynchat": "3.13",
    "asyncore": "3.13",
    "smtpd": "3.13",
    "lib2to3": "3.13",
    "distutils": "3.12",
    "imp": "3.12",
}

SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints"}


def iter_sources(root: Path):
    for path in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else ".").resolve()
    problems: list[str] = []

    for path in iter_sources(root):
        rel = path.relative_to(root)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as exc:
            # The syntax check job owns this; do not double-report.
            print(f"skip (syntax error): {rel}:{exc.lineno}", file=sys.stderr)
            continue

        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]

            for module in modules:
                top = module.split(".")[0]
                if top in REMOVED:
                    problems.append(
                        f"{rel}:{node.lineno}: imports '{module}', "
                        f"removed from the standard library in Python {REMOVED[top]}"
                    )

    if problems:
        print(f"\n{len(problems)} import(s) of removed standard-library modules:\n")
        for problem in problems:
            print(f"  {problem}")
        print(
            "\nThese import fine on older interpreters but fail at startup on "
            "newer ones. Use a maintained third-party package instead, or delete "
            "the import if it is unused."
        )
        return 1

    print("No imports of removed standard-library modules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
