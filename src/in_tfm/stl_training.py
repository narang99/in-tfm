"""Train `StlInception` on STL-10, writing the checkpoint `StlInceptionAdapter` loads.

- Only the final model is saved, since the reports want one frozen checkpoint.
  - It is a dict with `model` (the state dict) and the step, epoch and test accuracy it was taken at.
- Training augments with a padded random crop and a horizontal flip.
  - The crop pads with zeros, which in normalised space is a mean-coloured border.
  - So borders are in the training distribution, and a border is not a cue for any class.
- Evaluation and every report use `source.eval_transform`, so a snapshot sees what the reports feed it.
"""

from pathlib import Path

import torch
from datasets import load_dataset
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from .device import default_device
from .models.stl_inception import StlInception
from .models.stl_inception.source import STL_MEAN, STL_STD, HubImages, ImageDataset, eval_transform

train_transform = transforms.Compose(
    [
        transforms.RandomCrop(96, padding=12),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(STL_MEAN, STL_STD),
    ]
)


class TrainConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", cli_kebab_case=True, cli_parse_args=True)

    dataset: str = "tanganke/stl10"
    out_dir: Path = Path("checkpoints/stl_inception")
    epochs: int = 15
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 5e-4
    seed: int = 0
    device: str = default_device()


class Snapshot(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: dict[str, torch.Tensor]
    step: int
    epoch: int
    test_acc: float


class Transformed(Dataset):
    def __init__(self, images: ImageDataset, transform) -> None:
        self.images = images
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, label = self.images[index]
        return self.transform(image), label


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    for images, labels in loader:
        correct += (model(images.to(device)).argmax(1).cpu() == labels).sum().item()
    return correct / len(loader.dataset)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: str,
) -> float:
    model.train()
    total = 0.0
    for images, labels in tqdm(loader, desc="train"):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(images.to(device)), labels.to(device))
        loss.backward()
        optimizer.step()
        scheduler.step()
        total += loss.item()
    return total / len(loader)


def train(config: TrainConfig, train_images: ImageDataset, test_images: ImageDataset) -> Path:
    torch.manual_seed(config.seed)
    train_loader = DataLoader(
        Transformed(train_images, train_transform), config.batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(Transformed(test_images, eval_transform), config.batch_size)
    model = StlInception().to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.lr, total_steps=config.epochs * len(train_loader)
    )
    for epoch in range(config.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, scheduler, config.device)
        print(f"epoch {epoch}: loss={loss:.4f} test_acc={accuracy(model, test_loader, config.device):.4f}")
    return save_snapshot(config, model, test_loader, steps=config.epochs * len(train_loader))


def save_snapshot(config: TrainConfig, model: nn.Module, test_loader: DataLoader, steps: int) -> Path:
    snapshot = Snapshot(
        model={k: v.cpu() for k, v in model.state_dict().items()},
        step=steps,
        epoch=config.epochs,
        test_acc=accuracy(model, test_loader, config.device),
    )
    config.out_dir.mkdir(parents=True, exist_ok=True)
    path = config.out_dir / "final.pt"
    torch.save(snapshot.model_dump(), path)
    print(f"saved {path} (test_acc={snapshot.test_acc:.4f})")
    return path


def load_hub_splits(dataset: str) -> tuple[HubImages, HubImages]:
    return HubImages(load_dataset(dataset, split="train")), HubImages(load_dataset(dataset, split="test"))
