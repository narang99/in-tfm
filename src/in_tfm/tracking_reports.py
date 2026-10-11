"""Reports for the saved states of a tracked training run, and the page that ties them together.

- A chosen step `t` needs the network just before it (state `t - 1`) and just after it (state `t`).
- The initial network (state 0) and the final one are always reported, so the page can show where training ended up.
- HDBSCAN cluster ids mean nothing across two reports, so clusters are matched by the cosine similarity of their mean Hadamard products.
  - The Hadamard product is input times weight row, so the weights changing between states makes the match approximate.
"""

import html
from pathlib import Path

import torch
from jaxtyping import Float
from PIL import Image, ImageDraw
from pydantic import BaseModel

from .neuron_report import ReportMeta
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


def match_clusters(
    means_a: dict[int, Float[torch.Tensor, "hidden"]], means_b: dict[int, Float[torch.Tensor, "hidden"]]
) -> list[ClusterMatch]:
    """Each cluster of `a` with its most similar cluster of `b`, a cluster of `b` may be chosen by several."""
    matches = []
    for id_a, mean_a in means_a.items():
        similarities = {
            id_b: torch.nn.functional.cosine_similarity(mean_a, mean_b, dim=0).item() for id_b, mean_b in means_b.items()
        }
        if similarities:
            best = max(similarities, key=similarities.get)
            matches.append(ClusterMatch(cluster_a=id_a, cluster_b=best, similarity=similarities[best]))
    return matches


def load_cluster_means(neuron_report_dir: Path) -> dict[int, torch.Tensor]:
    meta = ReportMeta.model_validate_json((neuron_report_dir / "meta.json").read_text())
    return {cid: torch.load(neuron_report_dir / f"cluster_{cid}" / "mean.pt") for cid in meta.clusters}


PLOT_SIZE = (900, 360)
PLOT_MARGIN = 56
INK = (27, 34, 39)
GRID = (217, 222, 226)
CHOSEN = (178, 94, 9)
CURVE = (11, 114, 133)


def save_distance_plot(log: TrackingLog, chosen: list[StepRecord], path: Path) -> None:
    """Cosine distance per step on a log axis, since the steps span several orders of magnitude."""
    image = Image.new("RGB", PLOT_SIZE, "white")
    draw = ImageDraw.Draw(image)
    width, height = PLOT_SIZE
    left, right, top, bottom = PLOT_MARGIN, width - 16, 16, height - PLOT_MARGIN
    values = torch.tensor([max(r.cosine_distance, 1e-12) for r in log.steps]).log10()
    low, high = values.min().floor().item(), values.max().ceil().item()
    n_steps = len(log.steps)

    def x_of(step: int) -> float:
        return left + (step - 1) / max(n_steps - 1, 1) * (right - left)

    def y_of(log_value: float) -> float:
        return bottom - (log_value - low) / (high - low) * (bottom - top)

    for exponent in range(int(low), int(high) + 1):
        draw.line([(left, y_of(exponent)), (right, y_of(exponent))], fill=GRID)
        draw.text((4, y_of(exponent) - 6), f"1e{exponent}", fill=INK)
    for step in range(0, n_steps + 1, max(1, n_steps // 6)):
        draw.text((x_of(max(step, 1)) - 8, bottom + 6), str(step), fill=INK)
    draw.text((left, height - 20), "optimizer step", fill=INK)
    draw.text((left, 2), "cosine distance to the previous step", fill=INK)
    points = [(x_of(r.step), y_of(v.item())) for r, v in zip(log.steps, values)]
    draw.line(points, fill=CURVE, width=2)
    for record in chosen:
        x = x_of(record.step)
        draw.line([(x, top), (x, bottom)], fill=CHOSEN)
        draw.text((x + 3, top), str(record.step), fill=CHOSEN)
    image.save(path)


def _link(label: str, state: int, neuron: int) -> str:
    return f'<a href="state_{state:04d}/neuron_{neuron}/index.html">{html.escape(label)}</a>'


def _match_rows(pair_name: str, matches: list[ClusterMatch]) -> str:
    rows = "".join(
        f"<tr><td>{pair_name}</td><td>{m.cluster_a}</td><td>{m.cluster_b}</td><td>{m.similarity:.3f}</td></tr>"
        for m in matches
    )
    return rows or f"<tr><td>{pair_name}</td><td colspan=3>no clusters on one side</td></tr>"


def write_index(
    reports_dir: Path, log: TrackingLog, chosen: list[StepRecord], states: list[StateReport], pairs: dict[str, list[ClusterMatch]]
) -> Path:
    state_rows = "".join(
        f"<tr><td>{s.state}</td><td>{html.escape(', '.join(s.labels))}</td><td>{_link('report', s.state, log.neuron)}</td></tr>"
        for s in states
    )
    step_rows = "".join(
        f"<tr><td>{r.step}</td><td>{r.epoch}</td><td>{r.lr:.2e}</td><td>{r.cosine_distance:.3e}</td><td>{r.euclidean_distance:.3e}</td></tr>"
        for r in chosen
    )
    match_rows = "".join(_match_rows(name, matches) for name, matches in pairs.items())
    page = f"""<!doctype html><meta charset="utf-8"><title>Neuron {log.neuron} over training</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:900px;margin:24px auto;padding:0 16px}}
table{{border-collapse:collapse;margin:8px 0 24px}}td,th{{border-bottom:1px solid #d9dee2;padding:4px 12px;text-align:left}}
img{{max-width:100%}}</style>
<h1>{html.escape(log.layer_name)} neuron {log.neuron} over training</h1>
<img src="distances.png" alt="cosine distance per step">
<h2>Large steps</h2>
<table><tr><th>step</th><th>epoch</th><th>lr</th><th>cosine</th><th>euclidean</th></tr>{step_rows}</table>
<h2>Reported states</h2>
<table><tr><th>state</th><th>why</th><th>report</th></tr>{state_rows}</table>
<h2>Cluster matches</h2>
<p>Each cluster of the first report with its most similar cluster of the second, by cosine of the mean Hadamard product.</p>
<table><tr><th>pair</th><th>first</th><th>second</th><th>cosine</th></tr>{match_rows}</table>
"""
    path = reports_dir / "index.html"
    path.write_text(page)
    return path
