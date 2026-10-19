#!/usr/bin/env python
"""Train the two-conv MNIST network and write `checkpoints/mnist_cnn/final.pt`.

    uv run python scripts/train_mnist.py
    uv run python scripts/train_mnist.py --channels.conv1 8 --channels.conv2 16 --out-dir /tmp/mnist_8_16

The default widths of 4 and 8 channels reach about 98.4% test accuracy in 5 epochs.
"""

from in_tfm.mnist_training import MnistTrainConfig, split_as_tensors, train


def main() -> None:
    config = MnistTrainConfig()
    print(config)
    train(config, split_as_tensors(config.dataset, "train"), split_as_tensors(config.dataset, "test"))


if __name__ == "__main__":
    main()
