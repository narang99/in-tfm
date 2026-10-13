#!/usr/bin/env python
"""Train the modular addition transformer of Nanda et al. and save its states, to follow neurons through the grokking.

    uv run python scripts/train_grokking.py

Writes `<out-dir>/metrics.json` (train and test loss and accuracy after every epoch),
`<out-dir>/states/state_NNNN.pt` (every `--save-every` epochs, state 0 being the initial network, the last being the final one).
The states use the format `GrokkingAdapter` loads, so a report on one is a normal report.
"""

from pathlib import Path

import torch
from torch import nn

from in_tfm.grokking_training import METRICS_NAME, GrokTrainConfig, train
from in_tfm.neuron_tracking import state_path


class TrainAndSave(GrokTrainConfig):
    save_every: int = 10


class StateSaver:
    """Plugs into `train` as its `StepObserver`."""

    def __init__(self, states_dir: Path, every: int) -> None:
        self.states_dir = states_dir
        self.every = every
        self.step = 0

    def start(self, model: nn.Module) -> None:
        self.states_dir.mkdir(parents=True, exist_ok=True)
        self._save(model)

    def after_step(self, model: nn.Module, lr: float) -> None:
        self.step += 1
        if self.step % self.every == 0:
            self._save(model)

    def _save(self, model: nn.Module) -> None:
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save({"model": state, "step": self.step}, state_path(self.states_dir, self.step))


def main() -> None:
    config = TrainAndSave()
    print(config)
    saver = StateSaver(config.out_dir / "states", config.save_every)
    metrics = train(config, observer=saver)
    (config.out_dir / METRICS_NAME).write_text("[" + ",".join(m.model_dump_json() for m in metrics) + "]")
    last = metrics[-1]
    print(f"epoch {last.epoch}: train_loss={last.train_loss:.4g} test_loss={last.test_loss:.4g} test_acc={last.test_acc:.4f}")


if __name__ == "__main__":
    main()
