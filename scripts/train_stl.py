#!/usr/bin/env python
"""Train the STL-10 mini-Inception and write `checkpoints/stl_inception/final.pt`.

    uv run python scripts/train_stl.py --epochs 15
    uv run python scripts/train_stl.py --epochs 1 --batch-size 64 --out-dir /tmp/stl_smoke

Report on the result with `--stl.checkpoint <path>`, see configs/stl_inception_conv.yaml.
The reference run is 15 epochs of Adam with a one-cycle schedule, which reaches the accuracy in its log line.
"""

from in_tfm.stl_training import TrainConfig, load_hub_splits, train


def main() -> None:
    config = TrainConfig()
    print(config)
    train(config, *load_hub_splits(config.dataset))


if __name__ == "__main__":
    main()
