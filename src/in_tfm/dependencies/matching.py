"""Which of a writer's own clusters a firing of it looks like.

- A writer with a neuron report clustered its hits there, on points built from its own input.
- A firing is matched to the cluster of its nearest clustered hit.
  - The points and the distance are the ones that report clustered with, see `clustered_rows`.
  - Only clusters the report shows are references, so a match always links to a section.
  - There is no similarity cutoff. The distance is kept beside the match, so a far one is visible.
- hp-do instead took the cluster with the highest max cosine, and accepted anything above 0.
"""

from collections.abc import Sequence
from itertools import batched

import numpy as np
import torch
from jaxtyping import Float, Int
from scipy.spatial.distance import cdist

from ..clustered_hits import load_clustered_hits
from ..neuron_run import clustered_rows
from ..report_meta import load_report_meta
from ..run_config import ClusteringConfig
from ..weighted_jaccard.vectorized import compared_parts, weighted_jaccard_block
from .capture import CaptureSetup, capture_taps
from .firings import ClusterMatch, WriterReportRef
from .refs import Tap

ComparedRows = Float[np.ndarray, "n compared_dim"]


class WriterReport(WriterReportRef):
    """Also how the report clustered, which the page does not need."""

    clustering: ClusteringConfig


def compared(
    inputs: Float[torch.Tensor, "n hidden"],
    weight: Float[torch.Tensor, "out hidden"],
    report: WriterReport,
) -> ComparedRows:
    points = clustered_rows(inputs, weight, report.neuron, report.clustering)
    if report.clustering.distance == "weighted_jaccard":
        return compared_parts(points, report.clustering.normalise)
    return points


def distances(
    queries: ComparedRows, references: ComparedRows, report: WriterReport
) -> Float[np.ndarray, "n_queries n_references"]:
    if report.clustering.distance == "weighted_jaccard":
        return weighted_jaccard_block(queries, references)
    return cdist(queries, references)


def hadamard_rows(
    inputs: Float[torch.Tensor, "n hidden"], weight_row: Float[torch.Tensor, "hidden"]
) -> Float[np.ndarray, "n hidden"]:
    return (inputs * weight_row).double().numpy()


class WriterMatcher:
    def __init__(
        self,
        report: WriterReport,
        weight: Float[torch.Tensor, "out hidden"],
        reference_inputs: Float[torch.Tensor, "n_refs hidden"],
        reference_labels: Int[np.ndarray, "n_refs"],
    ) -> None:
        self.report = report
        self.weight = weight
        self.references = compared(reference_inputs, weight, report)
        self.reference_hadamard = hadamard_rows(reference_inputs, weight[report.neuron])
        self.labels = reference_labels

    @property
    def tap(self) -> Tap:
        return Tap(layer=self.report.layer, port="input")

    def match(self, inputs: Float[torch.Tensor, "n hidden"]) -> list[ClusterMatch]:
        nearest = distances(
            compared(inputs, self.weight, self.report), self.references, self.report
        )
        closest = nearest.argmin(axis=1)
        own = hadamard_rows(inputs, self.weight[self.report.neuron])
        return [
            ClusterMatch(
                cluster_id=int(self.labels[ref]),
                distance=float(nearest[i, ref]),
                hadamard=own[i].tolist(),
                closest_hadamard=self.reference_hadamard[ref].tolist(),
            )
            for i, ref in enumerate(closest)
        ]


def gather_inputs(
    setup: CaptureSetup, tap: Tap, sample_ids: Sequence[str], tokens: Sequence[int]
) -> Float[torch.Tensor, "n hidden"]:
    """The tapped input at each (sample, token), in the order given."""
    rows: list[torch.Tensor] = [torch.empty(0)] * len(sample_ids)
    by_sample: dict[str, list[int]] = {}
    for index, sample_id in enumerate(sample_ids):
        by_sample.setdefault(sample_id, []).append(index)
    for chunk in batched(by_sample, setup.batch_size):
        batch = setup.source.to_model_batch(list(chunk))
        captured = capture_taps(
            setup.model, batch, [tap], setup.layout_for, setup.device
        )[tap]
        for row, sample_id in enumerate(chunk):
            for index in by_sample[sample_id]:
                rows[index] = captured[row, tokens[index]]
    return torch.stack(rows)


def build_matcher(
    setup: CaptureSetup, report: WriterReport, weight: Float[torch.Tensor, "out hidden"]
) -> WriterMatcher:
    """Re-gathers the inputs of the report's shown clusters' hits, since the report keeps only their positions."""
    hits = load_clustered_hits(report.report_dir)
    shown = set(load_report_meta(report.report_dir).clusters)
    rows = [i for i, label in enumerate(hits.labels) if label in shown]
    if not rows:
        raise ValueError(f"{report.report_dir} shows no clusters to match against")
    inputs = gather_inputs(
        setup,
        Tap(layer=report.layer, port="input"),
        [hits.sample_ids[i] for i in rows],
        [hits.token_idxs[i] for i in rows],
    )
    return WriterMatcher(
        report, weight, inputs, np.array([hits.labels[i] for i in rows])
    )
