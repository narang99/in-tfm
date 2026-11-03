from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from ....device import default_device

TOKENS_FILE = "{split}_tokens.npy"
LOG_FILE = "train_log.jsonl"


class ModelShape(BaseModel):
    d_model: int = 256
    n_heads: int = 8
    head_dim: int = 32
    n_layers: int = 2
    seq_len: int = 256


class TinyAttnOnlyTrainConfig(BaseSettings):
    model_config = SettingsConfigDict(
        extra="forbid", cli_kebab_case=True, cli_parse_args=True
    )

    dataset: str = "roneneldan/TinyStories"
    text_column: str = "text"
    vocab_size: int = 4096
    tokenizer_stories: int = 200_000
    """How many training stories the BPE merges are learned from. The whole split is then encoded."""
    shape: ModelShape = ModelShape()
    batch_size: int = 128
    lr: float = 3e-3
    init_std: float = 0.1
    """GPT-2's 0.02 is too small without normalization, see `model_setup.init_weights`."""
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_tokens: int = 500_000_000
    eval_every: int = 500
    eval_batches: int = 20
    snapshot_tokens: list[int] = [
        1_000_000,
        5_000_000,
        20_000_000,
        50_000_000,
        100_000_000,
        200_000_000,
    ]
    """Extra checkpoints along the way, to look at the model mid-training."""
    seed: int = 0
    device: str = default_device()
    out_dir: Path = Path("checkpoints/tiny_attn_only_2l")
