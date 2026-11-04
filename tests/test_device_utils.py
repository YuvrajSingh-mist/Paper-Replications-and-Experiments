"""Device resolution: CUDA keeps its distributed path, MPS stays single-device.

Two things must stay true, and both are easy to break by accident:

  1. On CUDA, every helper returns what the original hardcoded code returned:
     `nccl`, DDP with `device_ids=[rank]`, `DistributedSampler`, the bare
     LOCAL_RANK int as the training device, and a GradScaler for float16.

  2. On MPS/CPU nothing touches torch.distributed at all. Not a process group,
     not a collective, not a DistributedSampler. These trainers run on one Apple
     GPU, so the distributed machinery is CUDA-only.

Each scenario runs in its own subprocess because the torch stub is a module-level
singleton.
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


# --------------------------------------------------------------------------
# No accelerator: single process, no distributed machinery
# --------------------------------------------------------------------------

NO_ACCELERATOR = """
import torch
import device_utils as du

assert du.cuda_available() is False
assert du.mps_available() is False
assert du.default_device() == "cpu"
assert du.default_dtype() == "float32"

# The whole point: no process group is created off CUDA.
du.setup_process_group()
assert torch.distributed.is_initialized() is False, "process group must stay uninitialised"
du.teardown_process_group()  # must not raise

assert du.is_distributed() is False
assert du.distributed_world_size() == 1
assert du.is_main_process() is True
assert du.grad_scaler_enabled() is False

model = object()
assert du.wrap_ddp(model) is model, "no DDP wrap off CUDA"
assert type(du.distributed_sampler(list(range(8)))).__name__ == "RandomSampler"
assert type(du.distributed_sampler(list(range(8)), shuffle=False)).__name__ == "SequentialSampler"

t = torch.zeros(1)
assert du.reduce_sum_to_main(t) is t, "no collective off CUDA"
print("ok")
"""


@pytest.mark.parametrize("package", PACKAGES)
def test_no_accelerator_is_single_process(package):
    result = _run(package, NO_ACCELERATOR)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


# --------------------------------------------------------------------------
# MPS: single device, still no distributed machinery
# --------------------------------------------------------------------------

MPS_MACHINE = """
import sys
import torch
import device_utils as du

torch.cuda.is_available = lambda: False
sys.modules["torch.backends.mps"].is_available = lambda: True

assert du.default_device() == "mps"
assert du.default_dtype() == "float16"
assert du.is_cuda() is False

du.setup_process_group()
assert torch.distributed.is_initialized() is False, "MPS must not join a process group"
assert du.is_distributed() is False
assert du.distributed_world_size() == 1
assert du.is_main_process() is True

assert type(du.distributed_sampler(list(range(8)))).__name__ == "RandomSampler"
print("ok")
"""


@pytest.mark.parametrize("package", PACKAGES)
def test_mps_never_joins_a_process_group(package):
    result = _run(package, MPS_MACHINE)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


# --------------------------------------------------------------------------
# CUDA: the original distributed path, unchanged
# --------------------------------------------------------------------------

CUDA_MACHINE = """
import os
import torch
import device_utils as du

recorded = {}
torch.distributed.init_process_group = lambda backend, *a, **k: recorded.setdefault("backend", backend)
torch.distributed.reduce = lambda *a, **k: recorded.setdefault("reduce", True)
torch.cuda.set_device = lambda idx: recorded.setdefault("set_device", idx)
torch.distributed.is_initialized = lambda: True

os.environ["LOCAL_RANK"] = "3"
os.environ["WORLD_SIZE"] = "8"
os.environ["RANK"] = "3"

assert du.default_device() == "cuda"
assert du.default_dtype() == "bfloat16"
assert du.distributed_world_size() == 8
assert du.local_rank() == 3

# CUDA must keep the bare LOCAL_RANK int, not a torch.device.
assert du.train_device() == 3
assert isinstance(du.train_device(), int)

du.setup_process_group()
assert recorded["backend"] == "nccl", recorded
du.select_device()
assert recorded["set_device"] == 3, recorded

assert du.grad_scaler_enabled("cuda", "float16") is True
assert du.grad_scaler_enabled("cuda", "bfloat16") is False
assert du.is_main_process() is False

du.reduce_sum_to_main(torch.zeros(1))
assert recorded.get("reduce") is True
print("ok")
"""


@pytest.mark.parametrize("package", PACKAGES)
def test_cuda_behaviour_is_unchanged(package):
    result = _run(package, CUDA_MACHINE, gpus=8)
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


def test_cuda_dry_run_script_passes():
    """`scripts/cuda_dry_run.py` simulates an 8-GPU node and must stay green.

    It covers the whole dispatch layer under CUDA plus a static check that no
    ported file has gone back to calling nccl/DDP/set_device directly.
    """
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "cuda_dry_run.py")],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-2000:]
    assert "CUDA DRY RUN: ALL PASS" in result.stdout
