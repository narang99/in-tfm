"""Collecting one origin cluster's dependency firings, and the noise to read them against.

- The examples are every hit the origin report put in the cluster, see `clustered_hits`.
- Two passes over the model:
  - one over the samples holding examples, for the firings
  - one over the whole corpus, for the noise of every column that fired
"""

from collections.abc import Sequence
from itertools import batched
from pathlib import Path
from typing import Any

import torch
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel
from tqdm import tqdm

from ..clustered_hits import load_clustered_hits
from ..sources import SampleId
from .capture import CaptureSetup, capture_taps, sample_captures
from .dependency_map import DependencyMap
from .firings import (
    ClusterMatch,
    DependencyFiring,
    Example,
    FiringSelection,
    Matcher,
    WriterReportRef,
    example_firings,
)
from .noise import NoiseSample, NoiseSampler
from .refs import FixedWriter, NeuronWriter, Tap, TapColumn

CONTEXT_BEFORE = 12
CONTEXT_AFTER = 3

DEPENDENCIES_DIR = "dependencies"
FIRINGS_NAME = "firings.json"


class OriginCluster(BaseModel):
    report_dir: Path
    layer: str
    neuron: int
    cluster_id: int


class TokenRef(BaseModel):
    """A token as the cluster of a `by_token` writer, like the embedding."""

    text: str
    hadamard: list[float]
    """The writer's whole row at this token times the target's weight row."""


class ClusterDependencies(BaseModel):
    origin: OriginCluster
    selection: FiringSelection
    examples: list[Example]
    firings: list[DependencyFiring]
    noise: list[NoiseSample]
    """One per column that fired: each writer's own column, and each fired dim of the target input."""
    writer_reports: list[WriterReportRef] = []
    """The writers whose firings were matched to their own clusters."""
    tokens: dict[int, TokenRef] = {}
    """By token id, every token a `by_token` writer fired at."""


Matchers = dict[tuple[str, int], Matcher]
"""By writer layer and neuron."""


class PendingMatch(BaseModel, arbitrary_types_allowed=True):
    firing: int
    writer: int
    input: Float[torch.Tensor, "hidden"]


def cluster_examples(origin: OriginCluster) -> list[Example]:
    hits = load_clustered_hits(origin.report_dir)
    return [
        Example(sample_id=hits.sample_ids[i], token=hits.token_idxs[i])
        for i in hits.cluster_rows(origin.cluster_id)
    ]


def examples_by_sample(examples: list[Example]) -> dict[SampleId, list[int]]:
    grouped: dict[SampleId, list[int]] = {}
    for index, example in enumerate(examples):
        grouped.setdefault(example.sample_id, []).append(index)
    return grouped


def matcher_of(writer: object, matchers: Matchers) -> Matcher | None:
    if isinstance(writer, NeuronWriter):
        return matchers.get((writer.layer, writer.neuron))
    return None


def pending_matches(
    firings: list[DependencyFiring],
    first: int,
    sample: dict[Tap, torch.Tensor],
    matchers: Matchers,
) -> dict[tuple[str, int], list[PendingMatch]]:
    """The inputs of every matchable writer in `firings[first:]`, read from this sample's captures."""
    pending: dict[tuple[str, int], list[PendingMatch]] = {}
    for index in range(first, len(firings)):
        for slot, writer in enumerate(firings[index].writers):
            matcher = matcher_of(writer.writer, matchers)
            if matcher is None:
                continue
            key = (matcher.report.layer, matcher.report.neuron)
            row = sample[matcher.tap][writer.writer.token]
            pending.setdefault(key, []).append(
                PendingMatch(firing=index, writer=slot, input=row)
            )
    return pending


def attach_matches(
    firings: list[DependencyFiring],
    pending: dict[tuple[str, int], list[PendingMatch]],
    matchers: Matchers,
) -> None:
    """One matcher call per writer neuron, over all its firings at once."""
    for key, rows in pending.items():
        matches = matchers[key].match(torch.stack([row.input for row in rows]))
        for row, match in zip(rows, matches):
            firings[row.firing].writers[row.writer].match = match


def collect_firings(
    setup: CaptureSetup,
    examples: list[Example],
    target: Tap,
    weight_row: Float[torch.Tensor, "hidden"],
    dependency_map: DependencyMap,
    selection: FiringSelection,
    matchers: Matchers,
    tokens: dict[int, TokenRef],
) -> list[DependencyFiring]:
    """Fills `tokens`, and each text example's token and context, as it goes."""
    tokenizer = getattr(setup.source, "tokenizer", None)
    grouped = examples_by_sample(examples)
    matcher_taps = {m.tap for m in matchers.values()} - set(dependency_map.taps)
    taps = [target, *dependency_map.taps, *matcher_taps]
    firings: list[DependencyFiring] = []
    pending: dict[tuple[str, int], list[PendingMatch]] = {}
    for chunk in tqdm(list(batched(grouped, setup.batch_size)), desc="firings"):
        batch = setup.source.to_model_batch(list(chunk))
        captures = capture_taps(
            setup.model, batch, taps, setup.layout_for, setup.device
        )
        for row, sample_id in enumerate(chunk):
            sample = sample_captures(captures, row)
            first = len(firings)
            for index in grouped[sample_id]:
                if tokenizer is not None and batch.display_ids is not None:
                    describe_text_example(
                        examples[index],
                        batch.display_ids[row],
                        batch.valid_mask[row],
                        tokenizer,
                    )
                token = examples[index].token
                firings += example_firings(
                    index,
                    token,
                    sample[target][token],
                    weight_row,
                    sample,
                    dependency_map,
                    selection,
                )
            if tokenizer is not None:
                match_by_token(
                    firings, first, examples, sample, weight_row, tokens, tokenizer
                )
            for key, rows in pending_matches(firings, first, sample, matchers).items():
                pending.setdefault(key, []).extend(rows)
    attach_matches(firings, pending, matchers)
    return firings


def describe_text_example(
    example: Example,
    ids: Int[torch.Tensor, "seq"],
    valid: Bool[torch.Tensor, "seq"],
    tokenizer: Any,
) -> None:
    start = max(example.token - CONTEXT_BEFORE, 0)
    window = [
        i
        for i in range(start, min(example.token + CONTEXT_AFTER + 1, len(ids)))
        if valid[i]
    ]
    example.token_id = int(ids[example.token])
    example.context = [tokenizer.decode([int(ids[i])]) for i in window]
    example.focus = window.index(example.token)


def match_by_token(
    firings: list[DependencyFiring],
    first: int,
    examples: list[Example],
    sample: dict[Tap, torch.Tensor],
    weight_row: Float[torch.Tensor, "hidden"],
    tokens: dict[int, TokenRef],
    tokenizer: Any,
) -> None:
    """A `by_token` writer's match is its example's token, with no distance to measure."""
    for firing in firings[first:]:
        token_id = examples[firing.example].token_id
        if token_id is None:
            continue
        for writer in firing.writers:
            fixed = writer.writer
            if not (isinstance(fixed, FixedWriter) and fixed.by_token):
                continue
            writer.match = ClusterMatch(
                cluster_id=token_id, distance=0.0, hadamard=[], closest_hadamard=[]
            )
            if token_id not in tokens:
                row = sample[fixed.tap][fixed.token].cpu() * weight_row.cpu()
                tokens[token_id] = TokenRef(
                    text=tokenizer.decode([token_id]),
                    hadamard=row.double().tolist(),
                )


def fired_columns(firings: list[DependencyFiring], target: Tap) -> list[TapColumn]:
    writers = {w.writer.column for f in firings for w in f.writers}
    dims = {TapColumn(tap=target, column=f.dim) for f in firings}
    return sorted(writers | dims, key=lambda c: (c.tap.layer, c.tap.port, c.column))


def sample_noise(
    setup: CaptureSetup, columns: list[TapColumn], seed: int
) -> list[NoiseSample]:
    ids: Sequence[SampleId] = setup.source.sample_ids()
    sampler = NoiseSampler(columns, len(ids), seed)
    for chunk in tqdm(list(batched(ids, setup.batch_size)), desc="noise"):
        batch = setup.source.to_model_batch(list(chunk))
        captures = capture_taps(
            setup.model, batch, sampler.taps, setup.layout_for, setup.device
        )
        sampler.add(captures, batch.valid_mask)
    return sampler.samples()


def collect_cluster_dependencies(
    setup: CaptureSetup,
    origin: OriginCluster,
    weight_row: Float[torch.Tensor, "hidden"],
    dependency_map: DependencyMap,
    selection: FiringSelection,
    seed: int,
    matchers: Matchers | None = None,
) -> ClusterDependencies:
    """Without `matchers`, no firing is matched to its writer's clusters."""
    matchers = matchers or {}
    target = Tap(layer=origin.layer, port="input")
    examples = cluster_examples(origin)
    tokens: dict[int, TokenRef] = {}
    firings = collect_firings(
        setup, examples, target, weight_row, dependency_map, selection, matchers, tokens
    )
    noise = sample_noise(setup, fired_columns(firings, target), seed)
    return ClusterDependencies(
        origin=origin,
        selection=selection,
        examples=examples,
        firings=firings,
        noise=noise,
        writer_reports=[m.report for m in matchers.values()],
        tokens=tokens,
    )


def dependencies_dir(report_dir: Path, cluster_id: int) -> Path:
    return report_dir / DEPENDENCIES_DIR / f"cluster_{cluster_id}"


def write_cluster_dependencies(collected: ClusterDependencies) -> Path:
    out_dir = dependencies_dir(collected.origin.report_dir, collected.origin.cluster_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / FIRINGS_NAME
    path.write_text(collected.model_dump_json())
    return path


def load_cluster_dependencies(report_dir: Path, cluster_id: int) -> ClusterDependencies:
    path = dependencies_dir(report_dir, cluster_id) / FIRINGS_NAME
    return ClusterDependencies.model_validate_json(path.read_text())
