"""Device resolution must keep CUDA behaviour and add MPS/CPU.

The whole point of `device_utils` is that it is additive: on a CUDA machine every
helper returns what the old hardcoded code returned. These tests pin that down so
a later edit cannot silently turn a CUDA run into a CPU one.

Each scenario runs in its own subprocess because the torch stub is a module-level
singleton, and `install()` is deliberately idempotent.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SMOLHUB = REPO_ROOT / "SmolHub"
TESTS_DIR = REPO_ROOT / "tests"

PACKAGES = (
    "SmolLlama",
    "StoryLlama",
    "SmolMixtral",
    "StoryMixtral",
    "SmolTransformer",
    "StoryKimi",
)


def _run(package: str, code: str, gpus: int = 0) -> subprocess.CompletedProcess:
    """Run `code` in `package`'s directory with the torch stub installed."""
    prelude = (
        "import sys\n"
        f"sys.path.insert(0, {str(TESTS_DIR)!r})\n"
        "import _stub_deps\n"
        f"_stub_deps.install(device_count={gpus})\n"
    )
    return subprocess.run(
        [sys.executable, "-c", prelude + code],
        cwd=SMOLHUB / package,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_all_packages_ship_the_same_helper():
    """The helper is duplicated per package (like tokenizer.py); keep it in sync."""
    copies = {p: (SMOLHUB / p / "device_utils.py").read_text() for p in PACKAGES}
    reference = copies["SmolLlama"]
    for package, text in copies.items():
        assert text == reference, f"{package}/device_utils.py has drifted from SmolLlama's copy"


NO_ACCELERATOR = """
import device_utils as du

assert du.cuda_available() is False
assert du.mps_available() is False
assert du.default_device() == "cpu"
assert du.default_dtype() == "float32"
assert du.is_distributed() is False
assert du.distributed_world_size() == 1
assert du.grad_scaler_enabled() is False

model = object()
assert du.wrap_ddp(model) is model
print("ok")
"""

CUDA_MACHINE = """
import os
import torch
import device_utils as du

backends = []
torch.distributed.init_process_group = lambda backend, *a, **k: backends.append(backend)

os.environ["LOCAL_RANK"] = "3"
os.environ["WORLD_SIZE"] = "8"

assert du.default_device() == "cuda"
assert du.default_dtype() == "bfloat16"
assert du.distributed_world_size() == 8
assert du.local_rank() == 3
assert du.train_device() == 3, "CUDA must keep returning the bare LOCAL_RANK int"
assert du.grad_scaler_enabled("cuda", "float16") is True
assert du.grad_scaler_enabled("cuda", "bfloat16") is False

du.setup_process_group("cuda")
assert backends == ["nccl"], backends
print("ok")
"""

MPS_MACHINE = """
import sys
import torch
import device_utils as du

torch.cuda.is_available = lambda: False
sys.modules["torch.backends.mps"].is_available = lambda: True

assert du.mps_available() is True
assert du.default_device() == "mps"
assert du.default_dtype() == "float16"
assert du.grad_scaler_enabled() is False

assert du.distributed_world_size() == 1
print("ok")
"""


@pytest.mark.parametrize("package", PACKAGES)
def test_no_accelerator_falls_back_to_cpu(package):
    result = _run(package, NO_ACCELERATOR)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.parametrize("package", PACKAGES)
def test_cuda_behaviour_is_unchanged(package):
    result = _run(package, CUDA_MACHINE, gpus=8)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.parametrize("package", PACKAGES)
def test_mps_is_used_when_available(package):
    result = _run(package, MPS_MACHINE)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.parametrize("package", PACKAGES)
def test_config_defaults_follow_the_machine(package):
    """A dataclass default must not hardcode 'cuda' any more."""
    code = """
import sys
sys.argv = ["trainer"]
from config import ModelArgs
print(ModelArgs().device if callable(ModelArgs) else ModelArgs.device)
"""
    result = _run(package, code)
    assert result.returncode == 0, result.stderr
    assert "cpu" in result.stdout
