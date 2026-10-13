"""Full-batch training of the modular addition transformer, logging train and test loss every epoch.

- An epoch is one optimizer step, since the paper trains on the whole training set at once.
  - So the tracker's step numbering is the epoch's, and `StepObserver` works unchanged.
- Loss is taken at the `=` position only.
- The run is deterministic on CPU, which is how a replay reproduces the states of an earlier run.
  - The replay is how dense states around the loss drop are saved, since the drop is only known afterwards.
"""

from pathlib import Path

import torch
from jaxtyping import Float, Int
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch import nn
from tqdm import tqdm

from .device import default_device
from .models.grokking import GrokTransformer, ModularAdditionSplit, split_pairs
from .stl_training import StepObserver

METRICS_NAME = "metrics.json"


class GrokTrainConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", cli_kebab_case=True, cli_parse_args=True, cli_implicit_flags=True)

    out_dir: Path = Path("checkpoints/grokking")
    epochs: int = 40000
    lr: float = 1e-3
    weight_decay: float = 1.0
    betas: tuple[float, float] = (0.9, 0.98)
    train_fraction: float = 0.3
    data_seed: int = 598
    model_seed: int = 999
    device: str = default_device()


class EpochMetrics(BaseModel):
    epoch: int
    train_loss: float
    test_loss: float
    train_acc: float
    test_acc: float


def loss_and_accuracy(
    model: GrokTransformer, tokens: Int[torch.Tensor, "n 3"], labels: Int[torch.Tensor, "n"]
) -> tuple[Float[torch.Tensor, ""], float]:
    logits = model.logits_for_tokens(tokens)[:, -1]
    return nn.functional.cross_entropy(logits, labels), (logits.argmax(-1) == labels).float().mean().item()


def evaluate(model: GrokTransformer, split: ModularAdditionSplit, epoch: int) -> EpochMetrics:
    """Both losses are of the model after `epoch` steps, so row `e` describes `state_e`."""
    with torch.no_grad():
        train_loss, train_acc = loss_and_accuracy(model, split.train_tokens, split.train_labels)
        test_loss, test_acc = loss_and_accuracy(model, split.test_tokens, split.test_labels)
    return EpochMetrics(
        epoch=epoch, train_loss=train_loss.item(), test_loss=test_loss.item(), train_acc=train_acc, test_acc=test_acc
    )


def train(config: GrokTrainConfig, observer: StepObserver | None = None) -> list[EpochMetrics]:
    split = split_pairs(config.train_fraction, config.data_seed)
    torch.manual_seed(config.model_seed)
    model = GrokTransformer().to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=config.betas)
    split = ModularAdditionSplit(**{name: tensor.to(config.device) for name, tensor in split})
    if observer is not None:
        observer.start(model)
    metrics = [evaluate(model, split, epoch=0)]
    for epoch in tqdm(range(1, config.epochs + 1), desc="train", mininterval=60):
        optimizer.zero_grad()
        loss, _ = loss_and_accuracy(model, split.train_tokens, split.train_labels)
        loss.backward()
        optimizer.step()
        if observer is not None:
            observer.after_step(model, config.lr)
        metrics.append(evaluate(model, split, epoch))
    return metrics
