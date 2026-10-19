#!/usr/bin/env python
"""Train the two-conv MNIST network and write `checkpoints/mnist_cnn/final.pt`.

    uv run python scripts/train_mnist.py
    uv run python scripts/train_mnist.py --channels.conv1 2 --channels.conv2 4 --out-dir checkpoints/mnist_cnn_2_4

- The default widths of 8 and 16 channels reach about 98.1% test accuracy in 5 epochs.
- 2 and 4 channels, the smallest network, reach about 94.2%.
"""

from in_tfm.mnist_training import MnistTrainConfig, split_as_tensors, train


def main() -> None:
    config = MnistTrainConfig()
    print(config)
    train(config, split_as_tensors(config.dataset, "train"), split_as_tensors(config.dataset, "test"))


if __name__ == "__main__":
    main()
