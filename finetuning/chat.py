"""Ask the instruction-tuned model a question."""

import argparse
from pathlib import Path

from config import load_config
from data_preprocessing.tokenizer import BPETokenizer
from finetuning.dataset import format_prompt
from training.model import build_model


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--config", type=Path, default=ROOT / "models/config.json")
    parser.add_argument("--weights", type=Path,
                        default=ROOT / "models/instruction_tuned.weights.h5")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    tokenizer = BPETokenizer()
    model = build_model(load_config(args.config))
    model.load_weights(args.weights)
    prompt = format_prompt(args.question)
    prompt_ids = tokenizer.encode(prompt)
    generated = model.generate([prompt_ids], args.max_new_tokens,
                               args.temperature, eos_id=tokenizer.eos_id)[0].numpy()
    answer_ids = generated[len(prompt_ids):].tolist()
    if tokenizer.eos_id in answer_ids:
        answer_ids = answer_ids[:answer_ids.index(tokenizer.eos_id)]
    print(tokenizer.decode(answer_ids).strip())


if __name__ == "__main__":
    main()

