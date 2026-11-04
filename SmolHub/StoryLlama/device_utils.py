"""Device, dtype and launch helpers.

This module exists so the same scripts can run on CUDA, on Apple Silicon (MPS),
and on CPU. It is deliberately additive: every function returns exactly what the
code used to return when CUDA is available, so an existing CUDA run behaves
identically.

The old code hardcoded three things that do not exist on a Mac:

  device = 'cuda'             -- no CUDA on Apple Silicon
  init_process_group('nccl')  -- NCCL is only built with CUDA support
  torch.cuda.set_device(rank) -- CUDA-only API

Each has a replacement here that resolves the right value at runtime.
"""

import importlib.util
import os

import torch


def cuda_available():
    return torch.cuda.is_available()


def mps_available():
    """True on Apple Silicon with a Metal-capable build."""
    try:
        return torch.backends.mps.is_available()
    except AttributeError:  # torch built without MPS
        return False


def default_device():
    """CUDA if present (unchanged), else MPS, else CPU."""
    if cuda_available():
        return "cuda"
    if mps_available():
        return "mps"
    return "cpu"


def default_dtype():
    """bfloat16 on CUDA where supported; float16 on MPS; float32 on CPU."""
    if cuda_available():
        return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    if mps_available():
        # MPS supports float16. bfloat16 support is uneven across macOS versions,
        # so float16 is the safe choice here.
        return "float16"
    return "float32"


def is_launched_distributed():
    """True when torchrun (or equivalent) launched this process."""
    return "RANK" in os.environ and "LOCAL_RANK" in os.environ


def world_size():
    """Number of processes, or 1 for a plain `python trainer.py` run."""
    return int(os.environ.get("WORLD_SIZE", "1"))


def local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_distributed():
    """True when more than one process is participating."""
    return torch.distributed.is_available() and torch.distributed.is_initialized() and world_size() > 1


def setup_process_group(device=None):
    """Initialise the process group.

    CUDA: `nccl`, exactly as before, initialised unconditionally.

    MPS/CPU: stock PyTorch ships no distributed backend for MPS, and `nccl` is not
    compiled into a macOS build at all. So this uses `gloo` and, when the script
    was started with plain `python`, initialises a single-process group. That keeps
    every `torch.distributed` call in the training loops working unchanged while
    world size is 1.
    """
    device = device or default_device()

    if device.startswith("cuda"):
        # Unchanged CUDA path.
        torch.distributed.init_process_group("nccl")
        return

    # gloo has no rendezvous information for a plain python invocation.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    torch.distributed.init_process_group("gloo")


def train_device(device=None):
    """The device this process trains on.

    CUDA keeps returning the bare `LOCAL_RANK` integer, because that is what the
    existing code does and every downstream `.to(rank)` and `device_ids=[rank]`
    relies on it. MPS and CPU get a torch.device, which `.to()` also accepts.
    """
    device = device or default_device()
    if str(device).startswith("cuda"):
        return local_rank()
    return torch.device(device)


def select_device(device=None):
    """Apply the process's device to the runtime.

    CUDA: sets the current device from LOCAL_RANK, exactly as before.
    MPS/CPU: nothing to select, so this is a no-op.
    """
    device = device or default_device()
    if str(device).startswith("cuda"):
        torch.cuda.set_device(local_rank())


def distributed_world_size(device=None):
    """World size for loss averaging.

    CUDA keeps `torch.cuda.device_count()`, matching the original code. Elsewhere
    the process group's world size is the only meaningful answer, and it is 1 for
    a single-process run.
    """
    device = device or default_device()
    if str(device).startswith("cuda"):
        return torch.cuda.device_count() or 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def wrap_ddp(model, device=None):
    """Wrap in DDP only when there is more than one process.

    On CUDA with multiple GPUs this is the same DDP wrap as before, with the same
    `device_ids`. Single-process runs skip DDP, which is what makes MPS work: DDP
    over MPS has no stock backend.
    """
    device = device or default_device()
    if is_distributed() and str(device).startswith("cuda"):
        return torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank()])
    if is_distributed():
        return torch.nn.parallel.DistributedDataParallel(model)
    return model


def unwrap(model):
    """The underlying module, whether or not it is DDP-wrapped."""
    return model.module if hasattr(model, "module") else model


def autocast_context(device=None, dtype=torch.bfloat16):
    """Autocast on CUDA, no-op elsewhere.

    Passing device_type='mps' to torch.autocast is not supported on every PyTorch
    version, and a failure here would abort training rather than degrade, so MPS
    runs in the dtype the weights were cast to.
    """
    import contextlib

    device = device or default_device()
    if str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def grad_scaler_enabled(device=None, dtype=None):
    """Gradient scaling is a CUDA float16 concern only."""
    device = device or default_device()
    dtype = dtype or default_dtype()
    return str(device).startswith("cuda") and str(dtype) == "float16"


def synchronize(device=None):
    """Block until the current device's queued work finishes."""
    device = device or default_device()
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elif str(device) == "mps":
        torch.mps.synchronize()


def empty_cache(device=None):
    """Release cached memory for the current device."""
    device = device or default_device()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    elif str(device) == "mps":
        torch.mps.empty_cache()


_LIGER_AVAILABLE = None


def liger_available():
    """True when liger-kernel is installed and usable on this machine.

    Liger ships Triton kernels built for CUDA. Without a CUDA device the projects
    fall back to plain PyTorch ops, so every `use_liger` flag is gated on this.
    """
    global _LIGER_AVAILABLE
    if _LIGER_AVAILABLE is None:
        try:
            _LIGER_AVAILABLE = cuda_available() and importlib.util.find_spec("liger_kernel") is not None
        except (ImportError, ValueError):
            _LIGER_AVAILABLE = False
    return _LIGER_AVAILABLE
