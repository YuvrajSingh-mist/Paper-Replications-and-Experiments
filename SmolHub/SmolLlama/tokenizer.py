from transformers import AutoTokenizer
import os


class Tokenizer:
    def __init__(self, hf_token=None) -> None:
        # Read the token from the environment rather than hardcoding it. The
        # previous literal '...' was sent to the Hub as an access token, so
        # importing anything that built a tokenizer both hit the network and
        # presented invalid credentials.
        if hf_token is None:
            hf_token = os.environ.get("HF_TOKEN")
        if not hf_token or hf_token == "...":
            hf_token = None

        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", token=hf_token)

        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def ready_tokenizer(self):

        return self.tokenizer
