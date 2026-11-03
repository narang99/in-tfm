"""Train the attn-only model on TinyStories, into `checkpoints/tiny_attn_only_2l/` by default.

    uv run python -m in_tfm.models.attn_only_2l.train
    uv run python -m in_tfm.models.attn_only_2l.train --max-tokens 5000000 --out-dir checkpoints/tiny_timing

- The first run trains the tokenizer and encodes the corpus. Later runs with the same `--out-dir` reuse them.
- Each eval line prints the train and val loss.
"""

from .config import TinyAttnOnlyTrainConfig
from .loop import train


def main() -> None:
    config = TinyAttnOnlyTrainConfig()
    print(config)
    train(config)


if __name__ == "__main__":
    main()
