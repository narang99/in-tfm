"""The numbers a dependency page shows, from one cluster's collected firings.

- Nothing here trims or orders for display. That is `view`'s job, so it can change without re-collecting.
- A dependency is "frequent" when it fires in at least `max(min_fired_count, ceil(min_fired_ratio * n_examples))` examples.
"""

import math
from collections import defaultdict

import numpy as np
from pydantic import BaseModel

from .collect import ClusterDependencies
from .firings import DependencyFiring, WriterFiring, writer_contribution
from .refs import FixedWriter, NeuronWriter, TapColumn, Writer


class StatsSettings(BaseModel):
    min_fired_ratio: float = 0.1
    min_fired_count: int = 2


def writer_label(writer: Writer) -> str:
    match writer:
        case NeuronWriter(layer=layer, neuron=neuron):
            return f"{layer}:{neuron}"
        case FixedWriter(name=name, dim=dim):
            return f"{name}:{dim}"


class WriterSummary(BaseModel):
    column: TapColumn
    label: str
    kind: str
    n_fired: int
    fired_ratio: float
    """Of the dependency's own firings, how many kept this writer."""
    median_value: float
    median_contribution: float

    @property
    def share(self) -> float:
        return self.fired_ratio * abs(self.median_contribution)


class DependencySummary(BaseModel):
    dim: int
    n_fired: int
    fired_ratio: float
    """Of the cluster's examples."""
    median_contribution: float
    frequent: bool
    writers: list[WriterSummary]

    @property
    def share(self) -> float:
        return self.fired_ratio * abs(self.median_contribution)


class ConcentrationMarker(BaseModel):
    sign: int
    rank: int
    """1 is the dependency with the largest |median contribution| of its sign."""
    cumulative_share: float
    delta: float


class Concentration(BaseModel):
    """Lorenz-style curves of |median contribution|, one per sign, ranked by that magnitude.

    - The curves are raw cumulative sums, so the two signs stay comparable on one axis.
    - A marker's shares are fractions of its own sign's total.
    """

    positive: list[float]
    negative: list[float]
    markers: dict[int, ConcentrationMarker]
    """By dim. A dependency with a median of exactly zero has none."""


class DependencyStats(BaseModel):
    n_examples: int
    frequent_threshold: int
    settings: StatsSettings
    dependencies: list[DependencySummary]
    concentration: Concentration


def check_one_firing_per_dim(firings: list[DependencyFiring]) -> None:
    """Selection picks each dim at most once per example, so a repeat means the firings were assembled twice."""
    pairs = [(f.example, f.dim) for f in firings]
    if len(pairs) != len(set(pairs)):
        raise ValueError(
            "a dim fired more than once in one example; the firings hold duplicate rows"
        )


def frequent_threshold(n_examples: int, settings: StatsSettings) -> int:
    return max(
        settings.min_fired_count, math.ceil(settings.min_fired_ratio * n_examples)
    )


def writer_summaries(firings: list[DependencyFiring]) -> list[WriterSummary]:
    kept: dict[TapColumn, list[tuple[DependencyFiring, WriterFiring]]] = defaultdict(
        list
    )
    for firing in firings:
        for writer in firing.writers:
            kept[writer.writer.column].append((firing, writer))
    return [
        WriterSummary(
            column=column,
            label=writer_label(pairs[0][1].writer),
            kind=pairs[0][1].writer.kind,
            n_fired=len(pairs),
            fired_ratio=len(pairs) / len(firings),
            median_value=float(np.median([w.value for _, w in pairs])),
            median_contribution=float(
                np.median([writer_contribution(f, w) for f, w in pairs])
            ),
        )
        for column, pairs in kept.items()
    ]


def dependency_summaries(
    firings: list[DependencyFiring],
    n_examples: int,
    threshold: int,
) -> list[DependencySummary]:
    by_dim: dict[int, list[DependencyFiring]] = defaultdict(list)
    for firing in firings:
        by_dim[firing.dim].append(firing)
    return [
        DependencySummary(
            dim=dim,
            n_fired=len(dim_firings),
            fired_ratio=len(dim_firings) / n_examples,
            median_contribution=float(np.median([f.contribution for f in dim_firings])),
            frequent=len(dim_firings) >= threshold,
            writers=writer_summaries(dim_firings),
        )
        for dim, dim_firings in sorted(by_dim.items())
    ]


def cumulative_curve(magnitudes: list[float]) -> list[float]:
    return np.cumsum(magnitudes).tolist() if magnitudes else []


def concentration(dependencies: list[DependencySummary]) -> Concentration:
    ranked = sorted(dependencies, key=lambda d: -abs(d.median_contribution))
    curves: dict[int, list[float]] = {}
    markers: dict[int, ConcentrationMarker] = {}
    for sign in (1, -1):
        of_sign = [d for d in ranked if np.sign(d.median_contribution) == sign]
        curve = cumulative_curve([abs(d.median_contribution) for d in of_sign])
        curves[sign] = curve
        for rank, dependency in enumerate(of_sign):
            previous = curve[rank - 1] if rank else 0.0
            markers[dependency.dim] = ConcentrationMarker(
                sign=sign,
                rank=rank + 1,
                cumulative_share=curve[rank] / curve[-1],
                delta=(curve[rank] - previous) / curve[-1],
            )
    return Concentration(positive=curves[1], negative=curves[-1], markers=markers)


def dependency_stats(
    collected: ClusterDependencies, settings: StatsSettings = StatsSettings()
) -> DependencyStats:
    firings = collected.firings
    check_one_firing_per_dim(firings)
    n_examples = len(collected.examples)
    threshold = frequent_threshold(n_examples, settings)
    dependencies = dependency_summaries(firings, n_examples, threshold)
    return DependencyStats(
        n_examples=n_examples,
        frequent_threshold=threshold,
        settings=settings,
        dependencies=dependencies,
        concentration=concentration(dependencies),
    )
