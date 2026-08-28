"""JSONL instruction dataset with loss applied only to response tokens."""

import json
from pathlib import Path
import numpy as np
import tensorflow as tf

from data_preprocessing.tokenizer import BPETokenizer


def format_prompt(instruction: str, extra_input: str = "") -> str:
    prompt = f"### Instruction:\n{instruction.strip()}\n"
    if extra_input.strip():
        prompt += f"\n### Input:\n{extra_input.strip()}\n"
    return prompt + "\n### Response:\n"


def create_instruction_dataset(path: str | Path, max_length: int,
                               batch_size: int, shuffle: bool = True):
    tokenizer = BPETokenizer()
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise ValueError("Instruction JSONL is empty")

    def encode(row):
        prompt = tokenizer.encode(format_prompt(row["instruction"], row.get("input", "")))
        answer = tokenizer.encode(row["output"].strip()) + [tokenizer.eos_id]
        total_length = max_length + 1
        prompt = prompt[:max(1, total_length // 2)]
        answer = answer[:total_length - len(prompt)]
        tokens = prompt + answer
        x, y = tokens[:-1], tokens[1:]
        response_start = max(len(prompt) - 1, 0)
        weights = [0.0] * response_start + [1.0] * (len(y) - response_start)
        pad = max_length - len(x)
        return (np.asarray(x + [tokenizer.eos_id] * pad, np.int32),
                np.asarray(y + [tokenizer.eos_id] * pad, np.int32),
                np.asarray(weights + [0.0] * pad, np.float32))

    encoded = [encode(row) for row in rows]
    inputs, labels, weights = (np.stack(items) for items in zip(*encoded))
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels, weights))
    if shuffle:
        dataset = dataset.shuffle(len(encoded), seed=42)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
