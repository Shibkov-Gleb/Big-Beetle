"""Thin GPT-2 BPE wrapper around tiktoken."""

import tiktoken


class BPETokenizer:
    def __init__(self, encoding_name: str = "gpt2"):
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.eos_id = self.encoding.eot_token
        self.vocab_size = self.encoding.n_vocab

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = self.encoding.encode_ordinary(text)
        return ids + [self.eos_id] if add_eos else ids

    def decode(self, token_ids: list[int]) -> str:
        return self.encoding.decode(token_ids)

