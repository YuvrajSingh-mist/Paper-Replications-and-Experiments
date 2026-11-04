"""Keep requirements.txt in step with what the tree actually imports.

A requirements file rots silently: someone adds `import pandas`, the file is not
updated, and the next person to install from it gets an ImportError that looks
like a code bug. This checks both directions.

It also pins down the import-name to distribution-name mapping, because several
of them differ and one of them is actively dangerous:

    from muon import MuonWithAuxAdam   ->  muon-optimizer

`muon` exists on PyPI but is a multimodal omics framework with no such class, so
`pip install muon` installs something that fails later with a confusing
AttributeError rather than a ModuleNotFoundError at import.
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
REQUIREMENTS = REPO_ROOT / "requirements.txt"

EXCLUDED_PARTS = {"old", "__pycache__", ".git", ".ipynb_checkpoints"}
EXCLUDED_DIRS = ("Replications/CLAP",)  # vendored; see its own requirements.txt

# Import name -> distribution name, only where they differ.
IMPORT_TO_DISTRIBUTION = {
    "PIL": "Pillow",
    "google": "protobuf",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "cv2": "opencv-python",
    # The dangerous one. See the module docstring.
    "muon": "muon-optimizer",
}

# Imported only inside tests or dev tooling, so they belong in
# requirements-dev.txt rather than the runtime manifest.
DEV_ONLY = {"pytest", "_pytest"}

# Provided by the interpreter, or by a file in this repository.
STDLIB = set(sys.stdlib_module_names)


def _local_module_names() -> set[str]:
    names = set()
    for path in REPO_ROOT.rglob("*.py"):
        rel = path.relative_to(REPO_ROOT)
        if EXCLUDED_PARTS & set(rel.parts):
            continue
        names.add(path.stem)
    return names


def _third_party_imports() -> dict[str, set[str]]:
    """Map top-level import name -> files that import it."""
    local = _local_module_names()
    found: dict[str, set[str]] = {}
    for path in REPO_ROOT.rglob("*.py"):
        rel = path.relative_to(REPO_ROOT)
        if EXCLUDED_PARTS & set(rel.parts):
            continue
        if any(str(rel).startswith(d) for d in EXCLUDED_DIRS):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            for name in names:
                if name in STDLIB or name in local or name in DEV_ONLY:
                    continue
                if name in {"tools", "tests"}:
                    continue
                found.setdefault(name, set()).add(str(rel))
    return found


def _declared_distributions() -> set[str]:
    """Distribution names declared in requirements.txt, normalised."""
    declared = set()
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name = re.split(r"[=<>!~\[ ]", line, maxsplit=1)[0]
        declared.add(name.lower().replace("_", "-"))
    return declared


def test_requirements_file_exists() -> None:
    assert REQUIREMENTS.exists()


def test_every_imported_distribution_is_declared() -> None:
    declared = _declared_distributions()
    missing = []
    for module, files in sorted(_third_party_imports().items()):
        distribution = IMPORT_TO_DISTRIBUTION.get(module, module)
        if distribution.lower().replace("_", "-") not in declared:
            missing.append(f"  {module} -> {distribution}  (imported in {sorted(files)[0]} +{len(files) - 1} more)")
    assert not missing, (
        "these third-party imports are not declared in requirements.txt:\n"
        + "\n".join(missing)
        + "\n\nAdd the distribution, or map the import name in IMPORT_TO_DISTRIBUTION."
    )


def test_no_unused_declarations() -> None:
    """Every declared distribution should correspond to a real import."""
    imported = {IMPORT_TO_DISTRIBUTION.get(m, m).lower().replace("_", "-") for m in _third_party_imports()}
    # Installed for a CUDA-index build or as a transitive requirement, or used by
    # a script that imports it dynamically, so not necessarily a direct import.
    ALLOWED_EXTRA = {
        "torch",  # imported everywhere, but listed explicitly for the CUDA index
        "tokenizers",  # pulled in by transformers
        "safetensors",  # pulled in by transformers
    }
    unused = sorted(_declared_distributions() - imported - ALLOWED_EXTRA)
    assert not unused, f"declared in requirements.txt but never imported: {unused}"
