#!/usr/bin/env python
"""Train a 2-layer attention-only shortformer model on TinyStories, into `checkpoints/tiny_attn_only_2l/`.

    python scripts/train_tiny_attn_only.py
    python scripts/train_tiny_attn_only.py --max-tokens 5000000 --out-dir checkpoints/tiny_timing

- The first run trains the tokenizer and encodes the corpus. Later runs with the same `--out-dir` reuse them.
- Each eval line prints the loss on a repeated random sequence, first copy then second copy.
  - The second copy's loss falls far below the first once the induction heads form.
"""

from in_tfm.tiny_attn_only_training import TinyAttnOnlyTrainConfig, train


def main() -> None:
    config = TinyAttnOnlyTrainConfig()
    print(config)
    train(config)


if __name__ == "__main__":
    main()
