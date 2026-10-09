import numpy as np
import torch
from PIL import Image

from in_tfm.models.stl_inception.adapter import load_model
from in_tfm.stl_training import TrainConfig, train


class Tiny:
    def __init__(self, n: int) -> None:
        rng = np.random.default_rng(0)
        self.images = [Image.fromarray(rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)) for _ in range(n)]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        return self.images[index], index % 10


def test_training_writes_a_checkpoint_the_adapter_loads(tmp_path):
    config = TrainConfig(_cli_parse_args=[], out_dir=tmp_path, epochs=1, batch_size=4, device="cpu")
    path = train(config, Tiny(8), Tiny(4))
    model = load_model(str(path))
    assert not model.training
    assert model(torch.zeros(1, 3, 96, 96)).shape == (1, 10)
    saved = torch.load(path)
    assert saved["epoch"] == 1
    assert 0 <= saved["test_acc"] <= 1
