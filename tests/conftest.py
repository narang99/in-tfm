"""Fixtures shared by the dependency tests."""

import numpy as np
import pytest
import torch
from PIL import Image

from in_tfm.clustered_hits import clustered_hits, write_clustered_hits
from in_tfm.dependencies import NeuronWriter
from in_tfm.dependencies.collect import CaptureSetup, ClusterDependencies, OriginCluster
from in_tfm.dependencies.firings import (
    DependencyFiring,
    Example,
    FiringSelection,
    WriterFiring,
)
from in_tfm.models.mnist_cnn.adapter import MnistCnnSmallAdapter
from in_tfm.models.mnist_cnn.model import IMAGE_SIZE, MnistCnn, MnistCnnChannels
from in_tfm.models.mnist_cnn.source import MnistSource


def writer(neuron: int, value: float, term: float) -> WriterFiring:
    return WriterFiring(
        writer=NeuronWriter(layer="up", neuron=neuron, token=0), value=value, term=term
    )


def firing(example: int, dim: int, weight: float, writers: list[WriterFiring]):
    total = sum(w.term for w in writers)
    return DependencyFiring(
        example=example,
        dim=dim,
        input_value=total,
        weight=weight,
        contribution=total * weight,
        writers=writers,
        n_writers=2,
    )


@pytest.fixture
def collected() -> ClusterDependencies:
    firings = [
        firing(0, 0, 2.0, [writer(0, 3.0, 3.0), writer(1, -0.5, -0.5)]),
        firing(1, 0, 2.0, [writer(0, 0.5, 0.5)]),
        firing(2, 1, -1.0, [writer(1, 2.0, 2.0)]),
    ]
    return ClusterDependencies(
        origin=OriginCluster(report_dir="r", layer="head", neuron=0, cluster_id=0),
        selection=FiringSelection(dim_mass=0.9, writer_mass=0.9),
        examples=[Example(sample_id=str(i), token=0) for i in range(3)],
        firings=firings,
    )


def fake_mnist_rows(n: int) -> list[dict]:
    rng = np.random.default_rng(0)
    return [
        {
            "image": Image.fromarray(
                rng.integers(0, 255, (IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
            )
        }
        for _ in range(n)
    ]


@pytest.fixture
def mnist_origin(tmp_path) -> tuple[CaptureSetup, OriginCluster, MnistCnnSmallAdapter]:
    torch.manual_seed(0)
    model = MnistCnn(MnistCnnChannels(conv1=2, conv2=4)).eval()
    adapter = MnistCnnSmallAdapter(model, "head.1", fake_mnist_rows(6), "test")
    source = MnistSource(adapter.split_rows, range(6), "test", n_tokens=1)
    write_clustered_hits(
        tmp_path,
        clustered_hits(
            list(source.sample_ids()),
            batch_idx=np.array([0, 1, 2, 3, 4, 5]),
            token_idx=np.zeros(6, dtype=int),
            labels=np.array([0, 0, -1, 0, 1, 1]),
            activations=np.ones(6),
        ),
    )
    setup = CaptureSetup(
        model=model,
        source=source,
        layout_for=adapter.layout_for,
        batch_size=2,
        device="cpu",
    )
    origin = OriginCluster(report_dir=tmp_path, layer="head.1", neuron=3, cluster_id=0)
    return setup, origin, adapter
