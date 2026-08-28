"""End-to-end causal language-model pretraining."""

import argparse
from pathlib import Path
import tensorflow as tf

from config import get_config, save_config
from data_preprocessing.dataset import create_lm_dataset
from training.model import build_model


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-bin", type=Path, required=True)
    parser.add_argument("--val-bin", type=Path, required=True)
    parser.add_argument("--preset", choices=["gpt2-small", "tiny"], default="gpt2-small")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=1_000)
    parser.add_argument("--validation-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(42)
    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    config = get_config(args.preset)
    if args.seq_len > config.max_seq_len:
        raise ValueError("seq-len exceeds the model's max_seq_len")

    train_data = create_lm_dataset(args.train_bin, args.seq_len, args.batch_size)
    val_data = create_lm_dataset(args.val_bin, args.seq_len, args.batch_size,
                                 shuffle=False)
    model = build_model(config)
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=args.learning_rate, weight_decay=0.1,
            beta_1=0.9, beta_2=0.95, global_clipnorm=1.0),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    args.models_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, args.models_dir / "config.json")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            args.models_dir / "pretrained_best.weights.h5", save_weights_only=True,
            save_best_only=True, monitor="val_loss"),
        tf.keras.callbacks.CSVLogger(args.models_dir / "pretraining_log.csv"),
    ]
    model.fit(train_data, validation_data=val_data, epochs=args.epochs,
              steps_per_epoch=args.steps_per_epoch,
              validation_steps=args.validation_steps, callbacks=callbacks,
              verbose=2)
    model.save_weights(args.models_dir / "pretrained_final.weights.h5")


if __name__ == "__main__":
    main()
