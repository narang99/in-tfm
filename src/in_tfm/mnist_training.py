"""Train `MnistCnn` on MNIST, writing the checkpoint the MNIST adapter loads.

- Each split is loaded into one tensor up front, since all of MNIST is under 50MB.
  - Batches are then slices, with no per-image PIL transform in the loop.
- There is no augmentation, MNIST does not need it to pass 98%.
- The checkpoint stores the channel widths next to the state dict, so a loader can rebuild the model.
"""

from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from jaxtyping import Float, Int64
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .device import default_device
from .models.mnist_cnn import MnistCnn, MnistCnnChannels
from .stl_training import accuracy, train_one_epoch


class MnistTrainConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", cli_kebab_case=True, cli_parse_args=True)

    dataset: str = "ylecun/mnist"
    out_dir: Path = Path("checkpoints/mnist_cnn")
    channels: MnistCnnChannels = MnistCnnChannels()
    epochs: int = 5
    batch_size: int = 128
    lr: float = 3e-3
    seed: int = 0
    device: str = default_device()


class MnistSnapshot(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: dict[str, torch.Tensor]
    channels: MnistCnnChannels
    epoch: int
    test_acc: float


class MnistTensors(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    images: Float[torch.Tensor, "n 1 28 28"]
    labels: Int64[torch.Tensor, " n"]

    def as_dataset(self) -> TensorDataset:
        return TensorDataset(self.images, self.labels)


def split_as_tensors(dataset: str, split: str) -> MnistTensors:
    rows = load_dataset(dataset, split=split)
    pixels = np.stack([np.asarray(image, dtype=np.uint8) for image in rows["image"]])
    return MnistTensors(
        images=torch.from_numpy(pixels).float().div(255).unsqueeze(1),
        labels=torch.tensor(rows["label"], dtype=torch.int64),
    )


def train(config: MnistTrainConfig, train_split: MnistTensors, test_split: MnistTensors) -> Path:
    torch.manual_seed(config.seed)
    train_loader = DataLoader(train_split.as_dataset(), config.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_split.as_dataset(), 1024)
    model = MnistCnn(config.channels).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.lr, total_steps=config.epochs * len(train_loader)
    )
    for epoch in range(config.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, scheduler, config.device)
        print(f"epoch {epoch}: loss={loss:.4f} test_acc={accuracy(model, test_loader, config.device):.4f}")
    return save_snapshot(config, model, accuracy(model, test_loader, config.device))


def save_snapshot(config: MnistTrainConfig, model: nn.Module, test_acc: float) -> Path:
    snapshot = MnistSnapshot(
        model={k: v.cpu() for k, v in model.state_dict().items()},
        channels=config.channels,
        epoch=config.epochs,
        test_acc=test_acc,
    )
    config.out_dir.mkdir(parents=True, exist_ok=True)
    path = config.out_dir / "final.pt"
    torch.save(snapshot.model_dump(), path)
    print(f"saved {path} (test_acc={test_acc:.4f})")
    return path


def load_checkpoint(path: Path) -> MnistCnn:
    saved = torch.load(path, map_location="cpu")
    model = MnistCnn(MnistCnnChannels.model_validate(saved["channels"]))
    model.load_state_dict(saved["model"])
    return model.eval()
