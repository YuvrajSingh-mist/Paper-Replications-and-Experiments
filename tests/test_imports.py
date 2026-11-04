"""Smoke tests: every SmolHub library module must import cleanly.

These run with torch replaced by stubs (see _stub_deps.py), so they need no GPU,
no network, no weights, and finish in a couple of seconds.

What they catch, all of which have actually broken this repo:

  * a dataclass default dividing by torch.cuda.device_count(), which raises
    ZeroDivisionError on any machine without a GPU -- before user code runs
  * module-scope code that builds a tokenizer or downloads a dataset, so merely
    importing the file performs network I/O
  * a `from <module> import <name>` where the name does not exist, which is an
    ImportError at import time and which the linter cannot see across modules

Each case runs in its own subprocess. Every package defines modules with the same
names (config, data, model, trainer), so importing them in one process would let
the first one win and hide the rest.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SMOLHUB = REPO_ROOT / "SmolHub"
TESTS_DIR = REPO_ROOT / "tests"

# Library modules used by the trainers. Standalone scripts (model_summary.py,
# test_tokenizer.py, download_model_weight.py) are deliberately excluded: they do
# work at import by design, e.g. model_summary prints a parameter table.
MODULES = ("config", "tokenizer", "data", "model", "trainer", "inference")

PACKAGES = (
    "SmolLlama",
    "StoryLlama",
    "SmolMixtral",
    "StoryMixtral",
    "SmolTransformer",
    "StoryKimi",
)


def _cases() -> list[tuple[str, str]]:
    return [
        (package, module) for package in PACKAGES for module in MODULES if (SMOLHUB / package / f"{module}.py").exists()
    ]


def _import_in_subprocess(package: str, module: str, gpus: int = 0) -> subprocess.CompletedProcess:
    code = (
        "import sys\n"
        "import _stub_deps\n"
        "_stub_deps.install(device_count=%d)\n"
        "import importlib\n"
        "importlib.import_module(%r)\n"
        "if _stub_deps.NETWORK_CALLS:\n"
        "    sys.exit('network calls during import: %%r' %% (_stub_deps.NETWORK_CALLS,))\n"
        "print('ok')\n" % (gpus, module)
    )
    env = {
        "PYTHONPATH": f"{TESTS_DIR}:{SMOLHUB}",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
        # Make sure a token in the developer's environment cannot change behavior.
        "HOME": "/tmp",
    }
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=SMOLHUB / package,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


@pytest.mark.parametrize(("package", "module"), _cases())
def test_module_imports_without_gpu_or_network(package: str, module: str) -> None:
    """Import must succeed on a 0-GPU machine and must not touch the network."""
    result = _import_in_subprocess(package, module)
    assert result.returncode == 0, (
        f"importing {package}/{module}.py failed with no GPU available\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


def test_gradient_accumulation_is_defined_without_gpu() -> None:
    """Regression: this was a ZeroDivisionError at import on any CPU-only box."""
    code = (
        "import _stub_deps\n"
        "_stub_deps.install(device_count=0)\n"
        "import config\n"
        "value = config.ModelArgs.gradient_accumulation_steps\n"
        "assert isinstance(value, int) and value > 0, value\n"
        "print('ok', value)\n"
    )
    env = {
        "PYTHONPATH": f"{TESTS_DIR}:{SMOLHUB}",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
        "HOME": "/tmp",
    }
    for package in ("SmolLlama", "StoryLlama"):
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=SMOLHUB / package,
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        assert result.returncode == 0, f"{package}: {result.stdout}\n{result.stderr}"


def test_dataset_loading_is_lazy() -> None:
    """Regression: importing data.py used to download TinyStories."""
    code = (
        "import _stub_deps\n"
        "_stub_deps.install(device_count=0)\n"
        "import data\n"
        "assert not _stub_deps.NETWORK_CALLS, ('network at import', _stub_deps.NETWORK_CALLS)\n"
        "data._ensure_datasets()\n"
        "names = [call[0] for call in _stub_deps.NETWORK_CALLS]\n"
        "assert 'load_dataset' in names, names\n"
        "print('ok')\n"
    )
    env = {
        "PYTHONPATH": f"{TESTS_DIR}:{SMOLHUB}",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
        "HOME": "/tmp",
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=SMOLHUB / "SmolLlama",
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


def test_dpo_loader_yields_chosen_rejected_pairs() -> None:
    """Regression: train_dpo() used the plain-LM loader, so DPOloss got no 'chosen' key.

    Two things must hold: the DPO dataset is the one that gets loaded, and the
    collate function emits the {chosen, rejected} shape DPOloss consumes.
    """
    code = (
        "import _stub_deps\n"
        "_stub_deps.install(device_count=0)\n"
        "import data\n"
        "data.prepare_dataset_dpo('train', 0, 4)\n"
        "loaded = [call[1][0] for call in _stub_deps.NETWORK_CALLS if call[0] == 'load_dataset']\n"
        "assert loaded, 'no dataset was loaded'\n"
        "assert all('ultrafeedback' in str(name) for name in loaded), loaded\n"
        "print('ok')\n"
    )
    env = {
        "PYTHONPATH": f"{TESTS_DIR}:{SMOLHUB}",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
        "HOME": "/tmp",
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=SMOLHUB / "SmolLlama",
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


def test_train_dpo_uses_the_dpo_loader() -> None:
    """Regression: dpo_train_dpo() used the plain-LM loader, so DPOloss got no 'chosen' key.

    Checked on the AST rather than the test passing, because the bug was a call
    site: prepare_dataset_dpo() existed and worked the whole time, it was simply
    never called.
    """
    tree = ast.parse((SMOLHUB / "SmolLlama" / "trainer.py").read_text())

    train_dpo = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "train_dpo")
    called = {
        node.func.id for node in ast.walk(train_dpo) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "prepare_dataset_dpo" in called, f"train_dpo calls {sorted(called)}"
    assert "prepare_dataset" not in called, "train_dpo still uses the plain-LM loader"


def test_dpo_trainer_entry_point_still_reachable() -> None:
    """dpo_trainer.py was removed as a duplicate; --train D must still reach DPO."""
    source = (SMOLHUB / "SmolLlama" / "trainer.py").read_text()
    assert "def train_dpo(" in source
    assert 'args.train == "D"' in source
    assert not (SMOLHUB / "SmolLlama" / "dpo_trainer.py").exists()
