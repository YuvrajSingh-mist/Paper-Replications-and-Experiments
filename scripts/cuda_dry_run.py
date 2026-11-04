"""Dry run of the CUDA path on a machine with no NVIDIA GPU.

Replaces torch's CUDA/distributed surface with recorders, then exercises the real
`device_utils` module from each SmolHub package and asserts the exact dispatch a
torchrun job on an 8-GPU node would produce.

Uses real torch when it is importable and falls back to `tests/_stub_deps.py`
otherwise, so this runs both locally and in CI (which has no torch installed).

What this proves: with CUDA present the code still selects nccl, DDP with
`device_ids=[rank]`, `DistributedSampler`, a float16 GradScaler and cuda autocast,
and still uses the bare LOCAL_RANK int as the training device.

What it cannot prove: that a real NCCL all-reduce completes. That needs the
hardware. Everything up to the collective is covered here.
"""

import os
import sys
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_torch():
    """Real torch if available, else the repo's import stub."""
    try:
        import torch  # noqa: F401

        return torch, "real torch"
    except ImportError:
        sys.path.insert(0, os.path.join(REPO_ROOT, "tests"))
        import _stub_deps

        _stub_deps.install(device_count=8)
        import torch  # noqa: F401  (now the stub)

        return torch, "stub torch"


class Recorder:
    """Callable that remembers how it was invoked."""

    def __init__(self, name, events, result=None):
        self.name = name
        self.events = events
        self.result = result

    def __call__(self, *args, **kwargs):
        self.events.append((self.name, args, kwargs))
        return self.result


class FakeDDP:
    def __init__(self, module, device_ids=None, **kwargs):
        DDP_CALLS.append({"module": module, "device_ids": device_ids})
        self.module = module


DDP_CALLS = []


class FakeSampler:
    def __init__(self, dataset, shuffle=True, **kwargs):
        SAMPLER_CALLS.append({"dataset": dataset, "shuffle": shuffle})


SAMPLER_CALLS = []


def install_cuda_simulation(torch, gpus=8, rank=3):
    """Make this machine look like GPU `rank` of an `gpus`-card node."""
    events = []

    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = lambda: gpus
    torch.cuda.is_bf16_supported = lambda: True
    torch.cuda.set_device = Recorder("set_device", events)
    torch.cuda.synchronize = Recorder("cuda_synchronize", events)

    torch.distributed.is_available = lambda: True
    torch.distributed.is_initialized = lambda: True
    torch.distributed.get_world_size = lambda: gpus
    torch.distributed.get_rank = lambda: rank
    torch.distributed.init_process_group = Recorder("init_process_group", events)
    torch.distributed.destroy_process_group = Recorder("destroy_process_group", events)
    torch.distributed.reduce = Recorder("reduce", events)
    torch.distributed.ReduceOp = types.SimpleNamespace(SUM="SUM", AVG="AVG")

    torch.nn.parallel.DistributedDataParallel = FakeDDP
    torch.utils.data.distributed.DistributedSampler = FakeSampler
    torch.autocast = Recorder("autocast", events)

    return events


def run(package: str, gpus=8, rank=3) -> bool:
    torch, flavour = _ensure_torch()

    for mod in [m for m in sys.modules if m == "device_utils"]:
        del sys.modules[mod]
    DDP_CALLS.clear()
    SAMPLER_CALLS.clear()
    events = install_cuda_simulation(torch, gpus=gpus, rank=rank)

    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(gpus)

    pkg_dir = os.path.join(REPO_ROOT, "SmolHub", package)
    sys.path.insert(0, pkg_dir)
    results = []
    try:
        import importlib

        du = importlib.import_module("device_utils")

        print(f"\n{'=' * 70}\n{package}  ({flavour}; simulated: {gpus} GPUs, rank {rank})\n{'=' * 70}")

        def check(label, got, want):
            ok = got == want
            results.append(ok)
            print(f"  {'PASS' if ok else 'FAIL'}  {label:<40} {got!r}")
            if not ok:
                print(f"        expected {want!r}")

        def recorded(name):
            return [e for e in events if e[0] == name]

        check("default_device()", du.default_device(), "cuda")
        check("default_dtype()", du.default_dtype(), "bfloat16")
        check("train_device() is int", isinstance(du.train_device(), int), True)
        check("train_device() value", du.train_device(), rank)

        du.setup_process_group()
        r = recorded("init_process_group")
        check("init_process_group backend", r[0][1][0] if r else None, "nccl")

        du.select_device()
        r = recorded("set_device")
        check("torch.cuda.set_device", r[0][1][0] if r else None, rank)

        check("distributed_world_size()", du.distributed_world_size(), gpus)
        check("is_main_process()", du.is_main_process(), rank == 0)
        check("is_distributed()", du.is_distributed(), True)

        model = type("Model", (), {})()
        wrapped = du.wrap_ddp(model)
        check("wrap_ddp -> DDP", len(DDP_CALLS), 1)
        check("DDP device_ids", DDP_CALLS[0]["device_ids"] if DDP_CALLS else None, [rank])
        check("unwrap(DDP) returns module", du.unwrap(wrapped) is model, True)

        du.distributed_sampler([1, 2, 3], shuffle=True)
        check("sampler -> DistributedSampler", len(SAMPLER_CALLS), 1)
        check("sampler shuffle", SAMPLER_CALLS[0]["shuffle"] if SAMPLER_CALLS else None, True)

        check("granular: fp16 grading", du.grad_scaler_enabled("cuda", "float16"), True)
        check("granular: bf16 no grading", du.grad_scaler_enabled("cuda", "bfloat16"), False)

        du.autocast_context("cuda", "bf16")
        r = recorded("autocast")
        check("autocast device_type", r[0][2].get("device_type") if r else None, "cuda")

        du.reduce_sum_to_main(object())
        check("reduce called", len(recorded("reduce")), 1)

        du.teardown_process_group()
        check("destroy_process_group", len(recorded("destroy_process_group")), 1)

    finally:
        sys.path.remove(pkg_dir)

    return all(results)


FORBIDDEN_IN_LOOPS = (
    "init_process_group(",
    "torch.cuda.set_device(",
    "DistributedDataParallel(",
    "DistributedSampler(",
    "torch.cuda.synchronize()",
    "torch.cuda.empty_cache()",
)


def static_wiring() -> bool:
    """A ported file must route CUDA calls through the helpers.

    Only files that already import `device_utils` are checked: those are the ones
    that were ported, so any raw CUDA-only call left in them is an oversight.
    """
    print(f"\n{'=' * 70}\nstatic wiring check\n{'=' * 70}")
    ok = True
    smolhub = os.path.join(REPO_ROOT, "SmolHub")
    checked = 0
    for pkg in sorted(os.listdir(smolhub)):
        pkg_dir = os.path.join(smolhub, pkg)
        if not os.path.isdir(pkg_dir):
            continue
        for name in sorted(os.listdir(pkg_dir)):
            if not name.endswith(".py") or name in ("device_utils.py", "config.py"):
                continue
            path = os.path.join(pkg_dir, name)
            with open(path, encoding="utf-8") as fh:
                src = fh.read()
            if "device_utils" not in src:
                continue
            # Strip comments so commented-out calls do not count.
            code = "\n".join(line.split("#", 1)[0] for line in src.splitlines())
            checked += 1
            bad = [pat for pat in FORBIDDEN_IN_LOOPS if pat in code]
            label = f"{pkg}/{name}"
            if bad:
                ok = False
                print(f"  FAIL  {label:<34} raw CUDA calls: {bad}")
            else:
                print(f"  PASS  {label:<34} routed through helpers")
    print(f"\n  {checked} ported files checked")
    return ok


if __name__ == "__main__":
    packages = ["SmolLlama", "StoryLlama", "SmolMixtral", "StoryMixtral", "SmolTransformer", "StoryKimi"]
    all_ok = True
    for pkg in packages:
        try:
            all_ok &= run(pkg)
        except Exception as exc:  # noqa: BLE001
            all_ok = False
            print(f"  ERROR in {pkg}: {type(exc).__name__}: {exc}")

    all_ok &= static_wiring()

    print(f"\n{'=' * 70}")
    print("CUDA DRY RUN:", "ALL PASS" if all_ok else "FAILURES — see above")
    print("=" * 70)
    sys.exit(0 if all_ok else 1)
