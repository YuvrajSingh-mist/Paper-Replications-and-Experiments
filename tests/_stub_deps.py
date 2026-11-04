"""Import the SmolHub modules with torch and friends replaced by stubs.

Why this exists
---------------
These scripts need a CUDA device, a Hugging Face token and several gigabytes of
weights to do anything real. None of that is needed to catch the bugs that
actually broke them:

  * a dataclass default dividing by torch.cuda.device_count(), which raises
    ZeroDivisionError on any machine without a GPU, before user code runs
  * module-scope code that downloads a dataset or builds a tokenizer, so that
    merely importing the file hits the network
  * a name that does not resolve at import time

All three are import-time failures. This module installs lightweight stubs so a
plain `pytest` run reproduces them in about a second, with no GPU and no network.

What it does NOT do
-------------------
It does not test numerics, training, or anything past import. A module passing
here means "importable and side-effect free", not "correct".
"""

from __future__ import annotations

import importlib.machinery
import sys
import types

THIRD_PARTY = {
    "transformers",
    "datasets",
    "wandb",
    "tokenizers",
    "tiktoken",
    "tqdm",
    "torchinfo",
    "liger_kernel",
    "huggingface_hub",
    "bitsandbytes",
    "safetensors",
    "numpy",
    "PIL",
    "gradio",
    "spaces",
    "dotenv",
    "sklearn",
    "librosa",
    "soundfile",
    "h5py",
    "matplotlib",
    "torchvision",
    "scipy",
    "pandas",
    "sentencepiece",
    "muon",
    "torchtune",
    "jiwer",
    "IPython",
}

# Records attribute calls that would touch the network, so tests can assert that
# importing a module does not call from_pretrained() or load_dataset().
NETWORK_CALLS: list[tuple] = []


class Fake(types.ModuleType):
    """A module whose every attribute is another Fake, and which is callable."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.__path__: list[str] = []

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        child = Fake(f"{self.__name__}.{name}")
        setattr(self, name, child)
        return child

    def __call__(self, *args, **kwargs):
        return Fake(f"{self.__name__}()")

    def __iter__(self):
        return iter(())

    def __len__(self):
        # A tokenizer stub gets asked for len(get_vocab()); any positive int works.
        return 50257


class Module:
    """Stands in for torch.nn.Module so `class X(nn.Module)` is valid."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, *args, **kwargs):
        return Fake("forward()")

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return Fake(name)


class Dataset:
    """Stands in for torch.utils.data.Dataset."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __len__(self) -> int:
        return 0

    def __getitem__(self, index):
        raise IndexError(index)


class DataLoader:
    def __init__(self, dataset=None, *args, **kwargs) -> None:
        self.dataset = dataset

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0


def _record_from_pretrained(*args, **kwargs):
    NETWORK_CALLS.append(("from_pretrained", args, kwargs))
    return Fake("tokenizer")


def _record_load_dataset(*args, **kwargs):
    NETWORK_CALLS.append(("load_dataset", args, kwargs))
    return Fake("dataset")


class _Finder:
    """Fabricates any third-party module that is not installed."""

    def find_spec(self, fullname, path=None, target=None):
        top = fullname.split(".")[0]
        if top not in THIRD_PARTY:
            return None
        return importlib.machinery.ModuleSpec(fullname, self, is_package=True)

    def create_module(self, spec):
        return Fake(spec.name)

    def exec_module(self, module):
        pass


def _module(name: str, **attrs):
    mod = Fake(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    # Attach to the parent so `torch.nn.Module` resolves to this object rather
    # than to a fresh Fake created by the parent's __getattr__.
    parent_name, _, leaf = name.rpartition(".")
    if parent_name:
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, leaf, mod)
    return mod


def install(device_count: int = 0) -> None:
    """Install stubs. Safe to call more than once."""
    if "torch" in sys.modules and getattr(sys.modules["torch"], "_smol_stub", False):
        return

    torch = Fake("torch")
    torch._smol_stub = True  # type: ignore[attr-defined]
    # Register before building submodules: _module() attaches to its parent, and
    # looks the parent up in sys.modules.
    sys.modules["torch"] = torch

    class Cuda:
        @staticmethod
        def device_count():
            return device_count

        @staticmethod
        def is_available():
            return device_count > 0

        @staticmethod
        def is_bf16_supported():
            return device_count > 0

        @staticmethod
        def manual_seed(*args, **kwargs):
            return None

        @staticmethod
        def set_device(*args, **kwargs):
            return None

        @staticmethod
        def synchronize(*args, **kwargs):
            return None

        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def empty_cache():
            return None

        @staticmethod
        def get_device_name(*args, **kwargs):
            return "stub-gpu"

    cuda = Cuda()
    torch.cuda = cuda
    sys.modules["torch.cuda"] = cuda  # type: ignore[assignment]

    nn = _module("torch.nn", Module=Module, Parameter=lambda *a, **k: Fake("param"))
    _module("torch.nn.functional")
    _module("torch.nn.parallel", DistributedDataParallel=type("DDP", (Module,), {}))

    utils = _module("torch.utils")
    utils.path = []  # type: ignore[attr-defined]
    data_mod = _module(
        "torch.utils.data",
        Dataset=Dataset,
        DataLoader=DataLoader,
        TensorDataset=Dataset,
        Subset=Dataset,
        # Real classes so device_utils' non-CUDA fallbacks return recognisable
        # sampler types. A Fake would make every isinstance check pass.
        RandomSampler=type("RandomSampler", (), {"__init__": lambda self, *a, **k: None}),
        SequentialSampler=type("SequentialSampler", (), {"__init__": lambda self, *a, **k: None}),
        random_split=lambda *a, **k: [],
    )
    utils.data = data_mod
    _module(
        "torch.utils.data.distributed",
        DistributedSampler=type("DistributedSampler", (), {"__init__": lambda self, *a, **k: None}),
    )
    _module("torch.utils.checkpoint", checkpoint=lambda fn, *a, **k: fn(*a, **k))
    _module("torch.multiprocessing")
    _module("torch.optim")
    _module("torch.optim.lr_scheduler")

    # Distributed state is explicit rather than a truthy Fake. device_utils asks
    # is_available() and is_initialized(), and a Fake would answer both "yes",
    # which is the opposite of this stub's purpose (model a machine with no GPU).
    _module(
        "torch.distributed",
        is_available=lambda: True,
        is_initialized=lambda: False,
        get_world_size=lambda: 1,
        get_rank=lambda: 0,
        init_process_group=lambda *a, **k: None,
        destroy_process_group=lambda *a, **k: None,
        ReduceOp=Fake("torch.distributed.ReduceOp"),
    )

    # No MPS on a CUDA-only stub machine, and None of the CUDA fast-path flags.
    backends = _module("torch.backends")
    _module("torch.backends.mps", is_available=lambda: False)
    _module("torch.backends.cuda", matmul=Fake("torch.backends.cuda.matmul"), enable_flash_sdp=lambda *a, **k: None)
    _module("torch.backends.cudnn", allow_tf32=False)
    backends.mps = sys.modules["torch.backends.mps"]
    backends.cuda = sys.modules["torch.backends.cuda"]
    backends.cudnn = sys.modules["torch.backends.cudnn"]

    _module("torch.mps", is_available=lambda: False, synchronize=lambda *a, **k: None, empty_cache=lambda *a, **k: None)

    _module("torch.amp", GradScaler=type("GradScaler", (), {"__init__": lambda self, *a, **k: None}))
    _module("torch.amp.grad_scaler")
    _module("torch.cuda.amp", autocast=lambda *a, **k: lambda fn: fn)
    torch.autocast = lambda *a, **k: Fake("torch.autocast()")
    torch.device = lambda spec: Fake(f"torch.device({spec!r})")

    sys.modules["torch"] = torch

    tr = _module(
        "transformers",
        AutoTokenizer=type("AutoTokenizer", (), {"from_pretrained": staticmethod(_record_from_pretrained)}),
        PreTrainedTokenizerFast=type("PreTrainedTokenizerFast", (), {}),
        AutoModelForCausalLM=type(
            "AutoModelForCausalLM", (), {"from_pretrained": staticmethod(_record_from_pretrained)}
        ),
        pipeline=lambda *a, **k: None,
    )
    sys.modules["transformers"] = tr

    ds = _module(
        "datasets",
        load_dataset=_record_load_dataset,
        Dataset=Fake("datasets.Dataset"),
        DatasetDict=Fake("datasets.DatasetDict"),
        concatenate_datasets=lambda *a, **k: None,
    )
    sys.modules["datasets"] = ds

    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


if __name__ == "__main__":
    install()
    import importlib

    module_name = sys.argv[1]
    importlib.import_module(module_name)
