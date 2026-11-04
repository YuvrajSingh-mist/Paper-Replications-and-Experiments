from torch.utils.data.distributed import DistributedSampler


from datasets import load_dataset
from torch.utils.data import DataLoader
from config import ModelArgs
from tokenizer import Tokenizer


# Dataset selection flags. TinyStories is the default because it is small enough
# to actually download and train on.
tinystories = True
fw = False

fw_train = None
fw_test = None

_tokenizer = None


def get_tokenizer():
    """Build the tokenizer on first use.

    This used to run at module scope, so importing this file performed a network
    request and required a valid Hub token just to import.
    """
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = Tokenizer().ready_tokenizer()
        _tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return _tokenizer


def _ensure_datasets():
    """Load the dataset on first use.

    This previously ran at module scope, so importing this file downloaded
    TinyStories, printed it twice, and took a network round trip. Import should
    have no such side effects.
    """
    global fw_train, fw_test
    if fw_train is None:
        if tinystories:
            fw_train = load_dataset("roneneldan/TinyStories", split="train")
            fw_test = load_dataset("roneneldan/TinyStories", split="validation")
        elif fw:
            fw_train = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=False)
            fw_train = fw_train.train_test_split(test_size=0.01)
        else:
            raise ValueError("Set tinystories or fw to True.")
    return fw_train, fw_test


def tokenize_function(examples):
    return get_tokenizer()(
        examples["text"], max_length=ModelArgs.block_size, padding="max_length", truncation=True, return_tensors="pt"
    )


def prepare_dataset(split, device, batch_size):
    print("Device is: ", device)

    _ensure_datasets()
    tokenizer = get_tokenizer()

    def collate_fn(batch):
        # Extract text data
        texts = [item["text"] for item in batch]

        input_encodings = tokenizer(
            texts, max_length=ModelArgs.block_size, padding='max_length', truncation=True, return_tensors="pt"
        )

        input_encodings["labels"] = input_encodings["input_ids"].clone()

        input_encodings["labels"][:, :-1] = input_encodings["input_ids"][:, 1:]
        input_encodings["labels"][:, -1] = tokenizer.eos_token_id

        return input_encodings

    dataloader = None
    if tinystories:
        if split == 'train':
            data_loader = DataLoader(
                fw_train,
                # generator=generator,
                batch_size=batch_size,
                sampler=DistributedSampler(fw_train, shuffle=True),
                collate_fn=collate_fn,
                drop_last=True,
                shuffle=False,
            )
        elif split == 'val':
            data_loader = DataLoader(
                fw_test,
                batch_size=batch_size,
                sampler=DistributedSampler(fw_test, shuffle=True),
                collate_fn=collate_fn,
                drop_last=True,
                shuffle=False,
            )
    elif fw:
        if split == 'train':
            data_loader = DataLoader(
                fw_train['train'],
                batch_size=batch_size,
                sampler=DistributedSampler(fw_train['train'], shuffle=True),
                collate_fn=collate_fn,
                drop_last=True,
                shuffle=False,
            )
        elif split == 'val':
            data_loader = DataLoader(
                fw_train['test'],
                batch_size=batch_size,
                # generator=generator,
                sampler=DistributedSampler(fw_train["test"]),
                collate_fn=collate_fn,
                drop_last=True,
                shuffle=False,
            )
    return data_loader
