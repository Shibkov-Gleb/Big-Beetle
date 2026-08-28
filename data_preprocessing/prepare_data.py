"""Convert a UTF-8 text file (or directory of .txt files) to token .bin files."""

import argparse
import json
from pathlib import Path

import numpy as np

from data_preprocessing.tokenizer import BPETokenizer


def text_files(path: Path) -> list[Path]:
    return sorted(path.rglob("*.txt")) if path.is_dir() else [path]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="UTF-8 .txt file or folder")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    args = parser.parse_args()

    tokenizer = BPETokenizer()
    tokens: list[int] = []
    files = text_files(args.input)
    if not files:
        raise ValueError("No .txt files found")
    for path in files:
        tokens.extend(tokenizer.encode(path.read_text(encoding="utf-8"), add_eos=True))

    split = int(len(tokens) * (1.0 - args.validation_fraction))
    train, val = tokens[:split], tokens[split:]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.asarray(train, dtype=np.uint16).tofile(args.output_dir / "train.bin")
    np.asarray(val, dtype=np.uint16).tofile(args.output_dir / "val.bin")
    metadata = {"tokenizer": "gpt2", "vocab_size": tokenizer.vocab_size,
                "eos_id": tokenizer.eos_id, "train_tokens": len(train),
                "validation_tokens": len(val)}
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(metadata)


if __name__ == "__main__":
    main()

