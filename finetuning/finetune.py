"""Instruction fine-tuning from a pretraining checkpoint."""

import argparse
from pathlib import Path
import tensorflow as tf

from config import load_config
from finetuning.dataset import create_instruction_dataset
from training.model import build_model


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="JSONL instruction data")
    parser.add_argument("--config", type=Path, default=ROOT / "models/config.json")
    parser.add_argument("--weights", type=Path,
                        default=ROOT / "models/pretrained_best.weights.h5")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "models/instruction_tuned.weights.h5")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-last-n", type=int, default=2,
                        help="Train last N blocks; use 0 for the entire model")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.max_length > config.max_seq_len:
        raise ValueError("max-length exceeds the model's max_seq_len")
    model = build_model(config)
    model.load_weights(args.weights)

    if args.train_last_n > 0:
        model.embedding.trainable = False
        for block in model.blocks[:-args.train_last_n]:
            block.trainable = False

    dataset = create_instruction_dataset(
        args.data, args.max_length, args.batch_size, shuffle=True)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            args.learning_rate, weight_decay=0.01, global_clipnorm=1.0),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        weighted_metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    model.fit(dataset, epochs=args.epochs, verbose=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(args.output)


if __name__ == "__main__":
    main()
