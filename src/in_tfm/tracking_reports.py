"""Reports for the saved states of a tracked training run, and the page that ties them together.

- A chosen step `t` needs the network just before it (state `t - 1`) and just after it (state `t`).
- The initial network (state 0) and the final one are always reported, so the page can show where training ended up.
- HDBSCAN cluster ids mean nothing across two reports, so clusters are matched by the cosine similarity of their mean Hadamard products.
  - Matching takes `ClusterLike`, so groups of clusters can be matched the same way.
  - The Hadamard product is input times weight row, so the weights changing between states makes the match approximate.
"""

from collections.abc import Sequence
from pathlib import Path

from matplotlib.figure import Figure
from pydantic import BaseModel

from .cluster_units import ClusterLike, unit_means
from .html_report import page, render_template
from .neuron_tracking import StepRecord, TrackingLog


class StateReport(BaseModel):
    state: int
    labels: list[str]
    """Why this state is reported, such as `initial` or `before step 12`."""


class ClusterMatch(BaseModel):
    cluster_a: int
    cluster_b: int
    similarity: float


def states_to_report(chosen: list[StepRecord], n_states: int) -> list[StateReport]:
    """`n_states` counts the initial network, so the final state is `n_states - 1`."""
    labels: dict[int, list[str]] = {0: ["initial"], n_states - 1: ["final"]}
    for record in chosen:
        labels.setdefault(record.step - 1, []).append(f"before step {record.step}")
        labels.setdefault(record.step, []).append(f"after step {record.step}")
    return [StateReport(state=state, labels=labels[state]) for state in sorted(labels)]


def state_report_dir(reports_dir: Path, state: int) -> Path:
    return reports_dir / f"state_{state:04d}"


def match_clusters(units_a: Sequence[ClusterLike], units_b: Sequence[ClusterLike]) -> list[ClusterMatch]:
    """Each unit of `a` with its most similar unit of `b`, a unit of `b` may be chosen by several."""
    if not units_a or not units_b:
        return []
    similarity = unit_means(units_a) @ unit_means(units_b).T
    best = similarity.argmax(dim=1)
    return [
        ClusterMatch(cluster_a=a.id, cluster_b=units_b[int(j)].id, similarity=similarity[i, j].item())
        for i, (a, j) in enumerate(zip(units_a, best))
    ]


DISTANCE_PLOT_NAME = "distances.png"
CURVE_COLOR = "#0b7285"
CHOSEN_COLOR = "#b25e09"


def save_distance_plot(log: TrackingLog, chosen: list[StepRecord], path: Path) -> None:
    """Cosine distance per step on a log axis, since the steps span several orders of magnitude."""
    fig = Figure(figsize=(9, 3.6))
    ax = fig.subplots()
    ax.plot([r.step for r in log.steps], [max(r.cosine_distance, 1e-12) for r in log.steps], color=CURVE_COLOR)
    ax.scatter(
        [r.step for r in chosen], [r.cosine_distance for r in chosen], color=CHOSEN_COLOR, zorder=3, label="large steps"
    )
    ax.legend()
    ax.set_yscale("log")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cosine distance to the previous step")
    fig.tight_layout()
    fig.savefig(path)


class StateRow(BaseModel):
    state: int
    labels: list[str]
    href: str


def write_index(
    reports_dir: Path,
    log: TrackingLog,
    chosen: list[StepRecord],
    states: list[StateReport],
    pairs: dict[str, list[ClusterMatch]],
) -> Path:
    title = f"{log.layer_name} neuron {log.neuron} over training"
    state_rows = [
        StateRow(state=s.state, labels=s.labels, href=f"state_{s.state:04d}/neuron_{log.neuron}/index.html") for s in states
    ]
    body = render_template(
        "tracking_index.html",
        layer_name=log.layer_name,
        neuron=log.neuron,
        distance_plot=DISTANCE_PLOT_NAME,
        steps=chosen,
        states=state_rows,
        pairs=pairs,
    )
    path = reports_dir / "index.html"
    path.write_text(page(title, body))
    return path
