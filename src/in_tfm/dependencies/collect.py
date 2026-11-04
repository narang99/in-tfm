"""Collecting one origin cluster's dependency firings.

- The examples are every hit the origin report put in the cluster, see `clustered_hits`.
- One pass over the model, over only the samples holding examples.
"""

from itertools import batched
from pathlib import Path

import torch
from jaxtyping import Float
from pydantic import BaseModel
from tqdm import tqdm

from ..clustered_hits import load_clustered_hits
from ..sources import SampleId, SampleSource
from .capture import LayoutFor, capture_taps, sample_captures
from .dependency_map import DependencyMap
from .firings import DependencyFiring, Example, FiringSelection, example_firings
from .refs import Tap

DEPENDENCIES_DIR = "dependencies"
FIRINGS_NAME = "firings.json"


class OriginCluster(BaseModel):
    report_dir: Path
    layer: str
    neuron: int
    cluster_id: int


class ClusterDependencies(BaseModel):
    origin: OriginCluster
    selection: FiringSelection
    examples: list[Example]
    firings: list[DependencyFiring]


class CaptureSetup(BaseModel, arbitrary_types_allowed=True):
    model: torch.nn.Module
    source: SampleSource
    layout_for: LayoutFor
    batch_size: int
    device: str


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


def collect_firings(
    setup: CaptureSetup,
    examples: list[Example],
    target: Tap,
    weight_row: Float[torch.Tensor, "hidden"],
    dependency_map: DependencyMap,
    selection: FiringSelection,
) -> list[DependencyFiring]:
    grouped = examples_by_sample(examples)
    taps = [target, *dependency_map.taps]
    firings: list[DependencyFiring] = []
    for chunk in tqdm(list(batched(grouped, setup.batch_size)), desc="firings"):
        batch = setup.source.to_model_batch(list(chunk))
        captures = capture_taps(
            setup.model, batch, taps, setup.layout_for, setup.device
        )
        for row, sample_id in enumerate(chunk):
            sample = sample_captures(captures, row)
            for index in grouped[sample_id]:
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
    return firings


def collect_cluster_dependencies(
    setup: CaptureSetup,
    origin: OriginCluster,
    weight_row: Float[torch.Tensor, "hidden"],
    dependency_map: DependencyMap,
    selection: FiringSelection,
) -> ClusterDependencies:
    target = Tap(layer=origin.layer, port="input")
    examples = cluster_examples(origin)
    firings = collect_firings(
        setup, examples, target, weight_row, dependency_map, selection
    )
    return ClusterDependencies(
        origin=origin, selection=selection, examples=examples, firings=firings
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
