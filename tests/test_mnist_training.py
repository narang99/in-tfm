import torch

from in_tfm.mnist_training import MnistTensors, MnistTrainConfig, load_checkpoint, train
from in_tfm.models.mnist_cnn import MnistCnn, MnistCnnChannels


def tiny_split(n: int) -> MnistTensors:
    generator = torch.Generator().manual_seed(0)
    return MnistTensors(images=torch.rand(n, 1, 28, 28, generator=generator), labels=torch.arange(n) % 10)


def test_every_conv_is_3x3_with_stride_2():
    convs = [m for m in MnistCnn().modules() if isinstance(m, torch.nn.Conv2d)]
    assert len(convs) == 2
    assert all(c.kernel_size == (3, 3) and c.stride == (2, 2) for c in convs)


def test_the_checkpoint_rebuilds_the_trained_widths(tmp_path):
    channels = MnistCnnChannels(conv1=2, conv2=3)
    config = MnistTrainConfig(
        _cli_parse_args=[], out_dir=tmp_path, channels=channels, epochs=1, batch_size=4, device="cpu"
    )
    model = load_checkpoint(train(config, tiny_split(8), tiny_split(4)))
    assert not model.training
    assert model.channels == channels
    assert model(torch.zeros(1, 1, 28, 28)).shape == (1, 10)
