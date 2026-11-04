"""The numbers a dependency page shows, from one cluster's collected firings.

- Nothing here trims or orders for display. That is `view`'s job, so it can change without re-collecting.
- Distances are in noise units of the column they are measured on.
  - A unit is `noise_radius = shorth_high - median`, where the shorth is the shortest window holding 90% of the noise.
  - So 1 unit is how far above its median a column's noise usually reaches.
- A dependency is "frequent" when it fires in at least `max(min_fired_count, ceil(min_fired_ratio * n_examples))` examples.
  - The histograms pool the writer firings of frequent dependencies only, as hp-do did.
"""

import math
from collections import defaultdict
from collections.abc import Callable

import numpy as np
from jaxtyping import Float
from pydantic import BaseModel

from .collect import ClusterDependencies
from .firings import DependencyFiring, WriterFiring, writer_contribution
from .refs import FixedWriter, NeuronWriter, Tap, TapColumn, Writer

NOISE_RADIUS_TOL = 1e-6
"""Keeps a single-valued noise sample from dividing by zero."""


class StatsSettings(BaseModel):
    min_fired_ratio: float = 0.1
    min_fired_count: int = 2
    threshold: float = 0.8
    """Noise units. A writer firing beyond it counts as clearly above its noise."""
    noise_window: float = 0.9


class NoiseStats(BaseModel):
    median: float
    low: float
    high: float
    radius: float

    def units(self, value: float) -> float:
        return (value - self.median) / self.radius


def shorth(values: Float[np.ndarray, "n"], fraction: float) -> tuple[float, float]:
    """The shortest window holding `fraction` of the values, as (low, high)."""
    ordered = np.sort(values)
    window = math.ceil(fraction * len(ordered))
    widths = ordered[window - 1 :] - ordered[: len(ordered) - window + 1]
    start = int(np.argmin(widths))
    return float(ordered[start]), float(ordered[start + window - 1])


def noise_stats(values: Float[np.ndarray, "n"], window: float) -> NoiseStats:
    low, high = shorth(values, window)
    median = float(np.median(values))
    return NoiseStats(
        median=median, low=low, high=high, radius=high - median + NOISE_RADIUS_TOL
    )


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
    noise: NoiseStats

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
    noise: NoiseStats
    """Of the target's input on this dim."""
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


class SignSplit(BaseModel):
    all: list[float]
    adding: list[float]
    inhibiting: list[float]


class DependencyStats(BaseModel):
    n_examples: int
    frequent_threshold: int
    settings: StatsSettings
    dependencies: list[DependencySummary]
    concentration: Concentration
    distances: SignSplit
    """Noise units of every writer firing of a frequent dependency, split by the writer's contribution sign."""
    fractions_above: SignSplit
    """Per example, the fraction of its writer firings beyond `threshold`. Each sign has its own denominator."""


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


ColumnNoise = dict[TapColumn, NoiseStats]


def writer_summaries(
    firings: list[DependencyFiring], noise: ColumnNoise
) -> list[WriterSummary]:
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
            noise=noise[column],
        )
        for column, pairs in kept.items()
    ]


def dependency_summaries(
    firings: list[DependencyFiring],
    n_examples: int,
    threshold: int,
    noise: ColumnNoise,
    target: Tap,
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
            noise=noise[TapColumn(tap=target, column=dim)],
            writers=writer_summaries(dim_firings, noise),
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


def column_noise(collected: ClusterDependencies, window: float) -> ColumnNoise:
    return {
        sample.column: noise_stats(np.array(sample.values), window)
        for sample in collected.noise
    }


class WriterDistance(BaseModel):
    example: int
    units: float
    contribution: float


def frequent_writer_distances(
    firings: list[DependencyFiring],
    frequent_dims: set[int],
    noise: ColumnNoise,
) -> list[WriterDistance]:
    return [
        WriterDistance(
            example=firing.example,
            units=noise[writer.writer.column].units(writer.value),
            contribution=writer_contribution(firing, writer),
        )
        for firing in firings
        if firing.dim in frequent_dims
        for writer in firing.writers
    ]


Measure = Callable[[list[WriterDistance]], list[float]]


def split_by_sign(distances: list[WriterDistance], measure: Measure) -> SignSplit:
    adding = [d for d in distances if d.contribution > 0]
    inhibiting = [d for d in distances if d.contribution <= 0]
    return SignSplit(
        all=measure(distances), adding=measure(adding), inhibiting=measure(inhibiting)
    )


def units_of(distances: list[WriterDistance]) -> list[float]:
    return [d.units for d in distances]


def fractions_above(distances: list[WriterDistance], threshold: float) -> list[float]:
    """An example with no firings of this kind has no entry, rather than a spurious 0."""
    by_example: dict[int, list[float]] = defaultdict(list)
    for distance in distances:
        by_example[distance.example].append(distance.units)
    return [
        float(np.mean(np.array(units) > threshold)) for units in by_example.values()
    ]


def dependency_stats(
    collected: ClusterDependencies, settings: StatsSettings = StatsSettings()
) -> DependencyStats:
    firings = collected.firings
    check_one_firing_per_dim(firings)
    n_examples = len(collected.examples)
    threshold = frequent_threshold(n_examples, settings)
    noise = column_noise(collected, settings.noise_window)
    target = Tap(layer=collected.origin.layer, port="input")
    dependencies = dependency_summaries(firings, n_examples, threshold, noise, target)
    distances = frequent_writer_distances(
        firings, {d.dim for d in dependencies if d.frequent}, noise
    )
    return DependencyStats(
        n_examples=n_examples,
        frequent_threshold=threshold,
        settings=settings,
        dependencies=dependencies,
        concentration=concentration(dependencies),
        distances=split_by_sign(distances, units_of),
        fractions_above=split_by_sign(
            distances, lambda d: fractions_above(d, settings.threshold)
        ),
    )
