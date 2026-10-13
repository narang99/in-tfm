"""Follows value neurons through the grokking run, and picks the states worth a report.

- States are saved every `stride` epochs, so a drop in test loss is measured between two saved states.
  - The states around a drop are reported: the one before it, and the one after it.
- Distances use the same `NeuronWeights` helpers as the STL tracker, but come from the saved states.
  - The neurons are not known until training is over, so nothing could be tracked live.
"""

from pathlib import Path

import torch
from matplotlib.figure import Figure
from pydantic import BaseModel

from .grokking_training import EpochMetrics
from .layers import LayerGetter, v_proj_getter
from .models.grokking import GrokTransformer, load_model
from .neuron_tracking import StepRecord, TrackingLog, cosine_distance, euclidean_distance, neuron_weights, state_path
from .html_report import page, render_template
from .tracking_reports import ClusterMatch, StateReport

LOSS_PLOT_NAME = "loss_and_distance.png"


class LossDrop(BaseModel):
    before: int
    """The saved state the drop starts from."""
    after: int
    test_loss_before: float
    test_loss_after: float

    @property
    def drop(self) -> float:
        return self.test_loss_before - self.test_loss_after


def saved_epochs(metrics: list[EpochMetrics], stride: int) -> list[int]:
    return [m.epoch for m in metrics if m.epoch % stride == 0]


def loss_drops(metrics: list[EpochMetrics], stride: int) -> list[LossDrop]:
    """One per pair of consecutive saved states."""
    epochs = saved_epochs(metrics, stride)
    return [
        LossDrop(
            before=before, after=after, test_loss_before=metrics[before].test_loss, test_loss_after=metrics[after].test_loss
        )
        for before, after in zip(epochs, epochs[1:])
    ]


def drop_between(metrics: list[EpochMetrics], before: int, after: int) -> LossDrop:
    """Any two states, for a drop that takes longer than one stride, such as the grokking itself."""
    return LossDrop(
        before=before, after=after, test_loss_before=metrics[before].test_loss, test_loss_after=metrics[after].test_loss
    )


def largest_drops(drops: list[LossDrop], count: int, min_gap: int = 1) -> list[LossDrop]:
    """The `count` drops that took the test loss down most, no two starting closer than `min_gap` epochs. In training order."""
    chosen: list[LossDrop] = []
    for drop in sorted(drops, key=lambda d: d.drop, reverse=True):
        if len(chosen) == count:
            break
        if all(abs(drop.before - c.before) >= min_gap for c in chosen):
            chosen.append(drop)
    return sorted(chosen, key=lambda d: d.before)


def states_to_report(chosen: list[LossDrop], final_state: int) -> list[StateReport]:
    labels: dict[int, list[str]] = {0: ["initial"], final_state: ["final"]}
    for drop in chosen:
        labels.setdefault(drop.before, []).append(f"before the drop to {drop.after}")
        labels.setdefault(drop.after, []).append(f"after the drop from {drop.before}")
    return [StateReport(state=state, labels=labels[state]) for state in sorted(labels)]


def most_token_selective_neurons(model: GrokTransformer, per_head: int = 1) -> list[int]:
    """The value neurons of each head whose activation varies most across the 113 tokens at position 0.

    - A value neuron reads one token, so how much it varies over tokens is how much it has to say.
    - Flat index is `head * d_head + dim`, see `layers.v_proj_getter`.
    """
    config = model.config
    with torch.no_grad():
        tokens = torch.arange(model.embed_tokens.num_embeddings - 1)
        residual = model.embed_tokens(tokens) + model.pos_embed[0]
        spread = model.layers[0].self_attn.v_proj(residual).std(dim=0)
    chosen: list[int] = []
    for head in range(config.n_heads):
        head_spread = spread[head * config.d_head : (head + 1) * config.d_head]
        chosen += [head * config.d_head + int(d) for d in head_spread.topk(per_head).indices]
    return chosen


def tracking_log_from_states(states_dir: Path, epochs: list[int], neuron: int, lr: float) -> TrackingLog:
    """Distances between consecutive saved states, so a step here spans `stride` epochs."""
    getter: LayerGetter = v_proj_getter(0)
    previous = neuron_weights(load_model(str(state_path(states_dir, epochs[0]))), getter, neuron)
    records: list[StepRecord] = []
    for epoch in epochs[1:]:
        current = neuron_weights(load_model(str(state_path(states_dir, epoch))), getter, neuron)
        records.append(
            StepRecord(
                step=epoch,
                epoch=epoch,
                lr=lr,
                cosine_distance=cosine_distance(previous, current),
                euclidean_distance=euclidean_distance(previous, current),
            )
        )
        previous = current
    return TrackingLog(layer_name="layers.0.self_attn.v_proj", neuron=neuron, steps=records)


def save_loss_and_distance_plot(
    metrics: list[EpochMetrics], logs: list[TrackingLog], chosen: list[LossDrop], path: Path
) -> None:
    fig = Figure(figsize=(10, 6))
    loss_ax, distance_ax = fig.subplots(2, 1, sharex=True)
    epochs = [m.epoch for m in metrics]
    loss_ax.plot(epochs, [m.train_loss for m in metrics], label="train loss")
    loss_ax.plot(epochs, [m.test_loss for m in metrics], label="test loss")
    for drop in chosen:
        loss_ax.axvspan(drop.before, drop.after, color="#b25e09", alpha=0.35)
    loss_ax.set_yscale("log")
    loss_ax.legend()
    for log in logs:
        distance_ax.plot(
            [r.step for r in log.steps], [max(r.cosine_distance, 1e-12) for r in log.steps], label=f"neuron {log.neuron}"
        )
    distance_ax.set_yscale("log")
    distance_ax.set_xlabel("epoch")
    distance_ax.set_ylabel("cosine distance to the previous saved state")
    distance_ax.legend()
    fig.tight_layout()
    fig.savefig(path)


class NeuronLink(BaseModel):
    neuron: int
    href: str


class StateRow(BaseModel):
    state: int
    labels: list[str]
    neurons: list[NeuronLink]


def write_index(
    reports_dir: Path,
    neurons: list[int],
    chosen: list[LossDrop],
    states: list[StateReport],
    pairs: dict[int, dict[str, list[ClusterMatch]]],
) -> Path:
    """`pairs` is per neuron, then per drop."""
    rows = [
        StateRow(
            state=s.state,
            labels=s.labels,
            neurons=[NeuronLink(neuron=n, href=f"state_{s.state:05d}/neuron_{n}/index.html") for n in neurons],
        )
        for s in states
    ]
    body = render_template(
        "grokking_index.html", plot=LOSS_PLOT_NAME, drops=chosen, states=rows, neurons=neurons, pairs=pairs
    )
    path = reports_dir / "index.html"
    path.write_text(page("Grokking value neurons over training", body))
    return path
