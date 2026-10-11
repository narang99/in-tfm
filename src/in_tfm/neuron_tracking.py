"""Follows one conv neuron through training, and picks the steps worth a report.

- A step is numbered from 1, and step `t` moves the network from state `t - 1` to state `t`.
  - State 0 is the initial network, before any optimizer step.
- Every state is saved in the format `StlInceptionAdapter` loads, so a report on it is a normal report.
  - The model is small, so saving all of them costs about a gigabyte and selection can wait until training is over.
- Distances are computed in float64, because late steps move the weights by less than float32 resolves in a cosine.
"""

from pathlib import Path

import torch
from jaxtyping import Float
from pydantic import BaseModel
from torch import nn

from .layers import conv_getter

NeuronWeights = Float[torch.Tensor, "fan_in"]

LOG_NAME = "steps.json"


class StepRecord(BaseModel):
    step: int
    epoch: int
    lr: float
    cosine_distance: float
    euclidean_distance: float


class TrackingLog(BaseModel):
    layer_name: str
    neuron: int
    steps: list[StepRecord]


def state_path(states_dir: Path, step: int) -> Path:
    return states_dir / f"state_{step:04d}.pt"


def neuron_weights(model: nn.Module, layer_name: str, neuron: int) -> NeuronWeights:
    weight = conv_getter(layer_name)(model).weight
    return weight[neuron].detach().cpu().double().flatten()


def cosine_distance(before: NeuronWeights, after: NeuronWeights) -> float:
    return 1.0 - torch.nn.functional.cosine_similarity(before, after, dim=0).item()


def euclidean_distance(before: NeuronWeights, after: NeuronWeights) -> float:
    return (after - before).norm().item()


class NeuronTracker:
    """Plugs into `train` as its `StepObserver`."""

    def __init__(self, layer_name: str, neuron: int, states_dir: Path, steps_per_epoch: int) -> None:
        self.layer_name = layer_name
        self.neuron = neuron
        self.states_dir = states_dir
        self.steps_per_epoch = steps_per_epoch
        self.records: list[StepRecord] = []
        self.previous: NeuronWeights | None = None

    def start(self, model: nn.Module) -> None:
        self.states_dir.mkdir(parents=True, exist_ok=True)
        self.previous = neuron_weights(model, self.layer_name, self.neuron)
        self._save_state(model, step=0)

    def after_step(self, model: nn.Module, lr: float) -> None:
        step = len(self.records) + 1
        current = neuron_weights(model, self.layer_name, self.neuron)
        self.records.append(
            StepRecord(
                step=step,
                epoch=(step - 1) // self.steps_per_epoch,
                lr=lr,
                cosine_distance=cosine_distance(self.previous, current),
                euclidean_distance=euclidean_distance(self.previous, current),
            )
        )
        self.previous = current
        self._save_state(model, step)

    def log(self) -> TrackingLog:
        return TrackingLog(layer_name=self.layer_name, neuron=self.neuron, steps=self.records)

    def _save_state(self, model: nn.Module, step: int) -> None:
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save({"model": state, "step": step}, state_path(self.states_dir, step))


def largest_steps(records: list[StepRecord], count: int, min_gap: int, skip_first_step: bool = True) -> list[StepRecord]:
    """The `count` steps that moved the neuron most by cosine distance, no two closer than `min_gap` steps apart.

    - Under Adam the moves are strongly autocorrelated, so without a gap the top steps are neighbours.
      - Neighbours show the same change twice, which is not `count` distinct events.
    - Step 1 leaves the random initialisation and is the biggest move by construction.
      - The initial network gets its own report, so step 1 is skipped by default.
    - Returned in training order.
    """
    candidates = [r for r in records if not (skip_first_step and r.step == 1)]
    chosen: list[StepRecord] = []
    for record in sorted(candidates, key=lambda r: r.cosine_distance, reverse=True):
        if len(chosen) == count:
            break
        if all(abs(record.step - c.step) >= min_gap for c in chosen):
            chosen.append(record)
    return sorted(chosen, key=lambda r: r.step)
