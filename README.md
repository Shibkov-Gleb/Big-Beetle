# Small GPT in TensorFlow/Keras

An educational decoder-only transformer. The default `gpt2-small` preset has 12
decoder blocks, a 768-dimensional hidden state, 12 attention heads, and about
124.4 million parameters. Input and output token weights are tied. A much smaller
`tiny` preset is included for quick CPU tests.

## 1. Install

From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Prepare pretraining text

Put one or more UTF-8 `.txt` files in `raw_text/`, then run:

```powershell
python -m data_preprocessing.prepare_data raw_text --output-dir data
```

This uses tiktoken's GPT-2 BPE vocabulary and writes `data/train.bin` and
`data/val.bin`. On its first use, tiktoken downloads and caches the official
GPT-2 vocabulary tables; later runs can tokenize offline.

## 3. Pretrain

First run the small smoke-test model:

```powershell
python -m training.train --train-bin data/train.bin --val-bin data/val.bin --preset tiny --seq-len 128 --batch-size 2 --steps-per-epoch 100
```

For the approximately 124M-parameter model:

```powershell
python -m training.train --train-bin data/train.bin --val-bin data/val.bin --preset gpt2-small --seq-len 256 --batch-size 2 --mixed-precision
```

Checkpoints, config, and the CSV training log are saved in `models/`. Training a
124M model from scratch still needs a large corpus and substantial GPU time;
inference and short educational experiments are much lighter.

## 4. Instruction fine-tune

Use JSONL records with `instruction`, optional `input`, and `output` fields (see
`finetuning/sample_instructions.jsonl`):

```powershell
python -m finetuning.finetune --data finetuning/sample_instructions.jsonl
```

By default only the final two transformer blocks and final normalization are
updated, which lowers optimizer memory. Add `--train-last-n 0` for full-model
fine-tuning. The three sample records only demonstrate the format; useful behavior
requires a much larger, high-quality instruction dataset.

Ask the tuned model a question:

```powershell
python -m finetuning.chat "Explain why the sky is blue."
```

## Layout

- `data_preprocessing/`: GPT-2 BPE, learned token/position embeddings, binary data
- `layers/`: causal masked multi-head attention and decoder block
- `training/`: GPT model and full pretraining cycle
- `finetuning/`: response-only instruction loss and text generation
- `models/`: model configs, checkpoints, and logs
