"""Memory-efficient next-token datasets backed by uint16 token files."""

from pathlib import Path
import numpy as np
import tensorflow as tf


def create_lm_dataset(path: str | Path, seq_len: int, batch_size: int,
                      shuffle: bool = True, seed: int = 42) -> tf.data.Dataset:
    data = np.memmap(path, dtype=np.uint16, mode="r")
    if len(data) <= seq_len:
        raise ValueError(f"{path} needs more than {seq_len} tokens")

    def examples():
        rng = np.random.default_rng(seed)
        max_start = len(data) - seq_len - 1
        while True:
            starts = (rng.integers(0, max_start + 1, size=1) if shuffle
                      else range(0, max_start + 1, seq_len))
            for start in starts:
                chunk = np.asarray(data[start:start + seq_len + 1], dtype=np.int32)
                yield chunk[:-1], chunk[1:]

    signature = (
        tf.TensorSpec((seq_len,), tf.int32),
        tf.TensorSpec((seq_len,), tf.int32),
    )
    return (tf.data.Dataset.from_generator(examples, output_signature=signature)
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE))

