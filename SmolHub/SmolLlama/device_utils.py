"""Device and launch helpers.

One rule decides everything here:

    CUDA  -> distributed training, exactly as this repository has always done it.
    MPS   -> one device, one process. No process group, no DDP, no collectives.
    CPU   -> same as MPS.

MPS is deliberately *not* wired into the distributed machinery. Stock PyTorch
ships no distributed backend for MPS, and these trainers are meant to run on a
single Apple GPU, so on MPS the code takes the plain single-process path instead
of pretending to be rank 0 of a one-rank world.

On a CUDA machine every helper below returns what the original hardcoded code
returned, so an existing multi-GPU run is unaffected:

    device = 'cuda'                  -> train_device()      -> LOCAL_RANK (int)
    init_process_group('nccl')       -> setup_process_group() -> 'nccl'
    torch.cuda.set_device(rank)      -> select_device()      -> set_device(rank)
    DDP(model, device_ids=[rank])    -> wrap_ddp()           -> same DDP wrap
    DistributedSampler(ds)           -> distributed_sampler()-> same sampler
    torch.cuda.device_count()        -> world_size()         -> device_count()
"""

import importlib.util
import os

import torch

# --------------------------------------------------------------------------
# Hardware detection
# --------------------------------------------------------------------------


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


def is_cuda(device=None):
    """True when this run should use the CUDA/distributed path."""
    device = device or default_device()
    return str(device).startswith("cuda")


# --------------------------------------------------------------------------
# Launch topology
# --------------------------------------------------------------------------


def local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


def distributed_world_size(device=None):
    """Number of cooperating processes.

    CUDA keeps `torch.cuda.device_count()`, matching the original code, so the
    loss averaging in the training loops divides by the same number as before.
    MPS and CPU are single-process by definition.
    """
    if not is_cuda(device):
        return 1
    return torch.cuda.device_count() or 1


def is_distributed(device=None):
    """True only when a real multi-process CUDA job is running."""
    if not is_cuda(device) or distributed_world_size(device) <= 1:
        return False
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def setup_process_group(device=None):
    """Initialise the process group on CUDA; do nothing anywhere else.

    CUDA: `nccl`, initialised unconditionally, exactly as before.

    MPS/CPU: returns immediately. There is no backend to initialise, and adding a
    one-process gloo group would drag MPS into the distributed path for no
    benefit. Nothing downstream calls a collective on these devices (see
    `reduce_sum_to_main` and `distributed_sampler`).
    """
    if not is_cuda(device):
        return
    torch.distributed.init_process_group("nccl")


def train_device(device=None):
    """The device this process trains on.

    CUDA keeps returning the bare `LOCAL_RANK` integer, because that is what the
    original code does and every downstream `.to(rank)` and `device_ids=[rank]`
    relies on it. MPS and CPU get a torch.device, which `.to()` also accepts.
    """
    if is_cuda(device):
        return local_rank()
    return torch.device(device or default_device())


def select_device(device=None):
    """Apply the process's device to the runtime.

    CUDA: sets the current device from LOCAL_RANK, exactly as before.
    MPS/CPU: nothing to select, so this is a no-op.
    """
    if is_cuda(device):
        torch.cuda.set_device(local_rank())


def is_main_process(device=None):
    """True on rank 0.

    The trainers gate logging, wandb and checkpointing on `device == 0`, which
    works only because `train_device()` returns the bare rank on CUDA. On MPS and
    CPU the device is a torch.device, so that comparison is always False. This is
    the equivalent check that holds on every backend.
    """
    if is_cuda(device):
        return local_rank() == 0
    return True


# --------------------------------------------------------------------------
# Distributed wrappers (all no-ops off CUDA)
# --------------------------------------------------------------------------


def wrap_ddp(model, device=None):
    """Wrap in DDP on CUDA, return the model unchanged otherwise.

    On CUDA this is the same DDP wrap as before, with the same `device_ids`.
    MPS has no DDP backend, so the model is used directly.
    """
    if is_cuda(device):
        return torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank()])
    return model


def unwrap(model):
    """The underlying module, whether or not it is DDP-wrapped."""
    return model.module if hasattr(model, "module") else model


def distributed_sampler(dataset, shuffle=True, device=None):
    """Partitioned sampler on CUDA, plain sampler otherwise.

    `DistributedSampler` requires an initialised process group, which no longer
    exists off CUDA. Returning `RandomSampler`/`SequentialSampler` keeps the
    DataLoader call sites unchanged: they already pass `shuffle=False` alongside
    an explicit sampler.
    """
    if is_cuda(device):
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        return torch.utils.data.RandomSampler(dataset)
    return torch.utils.data.SequentialSampler(dataset)


def reduce_sum_to_main(tensor, dst=0, device=None):
    """Sum `tensor` across processes onto `dst`.

    A no-op off CUDA: with one process the reduction is the identity, and gloo
    has no MPS kernel for `reduce_` at all. Multi-process CUDA keeps the original
    collective.
    """
    if is_distributed(device):
        torch.distributed.reduce(tensor, dst=dst, op=torch.distributed.ReduceOp.SUM)
    return tensor


# --------------------------------------------------------------------------
# Precision and memory
# --------------------------------------------------------------------------


def autocast_context(device=None, dtype=torch.bfloat16):
    """Autocast on CUDA, no-op elsewhere.

    Passing device_type='mps' to torch.autocast is not supported on every PyTorch
    version, and a failure here would abort training rather than degrade, so MPS
    runs in the dtype the weights were cast to.
    """
    import contextlib

    if is_cuda(device):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def grad_scaler_enabled(device=None, dtype=None):
    """Gradient scaling is a CUDA float16 concern only."""
    dtype = dtype or default_dtype()
    return is_cuda(device) and str(dtype) == "float16"


def synchronize(device=None):
    """Block until the current device's queued work finishes."""
    device = device or default_device()
    if is_cuda(device):
        torch.cuda.synchronize()
    elif str(device) == "mps":
        torch.mps.synchronize()


def empty_cache(device=None):
    """Release cached memory for the current device."""
    device = device or default_device()
    if is_cuda(device):
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


def teardown_process_group():
    """Destroy the process group if one exists.

    CUDA: destroys it, as before. MPS/CPU: never initialised one, so destroying
    would raise.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
