"""Model-size presets and config loading."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class GPTConfig:
    vocab_size: int = 50_257
    max_seq_len: int = 1_024
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ff: int = 3_072
    dropout: float = 0.1


def get_config(preset: str = "gpt2-small") -> GPTConfig:
    if preset == "gpt2-small":
        return GPTConfig()
    if preset == "tiny":
        return GPTConfig(max_seq_len=256, n_layers=4, n_heads=4,
                         d_model=256, d_ff=1_024)
    raise ValueError(f"Unknown preset: {preset}")


def save_config(config: GPTConfig, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def load_config(path: str | Path) -> GPTConfig:
    return GPTConfig(**json.loads(Path(path).read_text(encoding="utf-8")))

