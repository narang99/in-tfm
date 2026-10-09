import numpy as np
import pytest
import torch

from in_tfm.hit_selection import select_hits
from in_tfm.neuron_capture import ScanResult
from in_tfm.run_config import HitSelectionConfig

BATCH, SEQ = 40, 50


@pytest.fixture
def scan() -> ScanResult:
    generator = torch.Generator().manual_seed(0)
    activations = torch.randn(BATCH, SEQ, 1, generator=generator)
    valid_mask = torch.ones(BATCH, SEQ, dtype=torch.bool)
    valid_mask[:, 40:] = False
    activations[:, 40:] = 100.0  # would dominate the tail if padding were not masked out
    return ScanResult(
        sample_ids=[str(i) for i in range(BATCH)], neuron_idxs=[7], activations=activations, valid_mask=valid_mask
    )


def values_at(scan: ScanResult, selection) -> np.ndarray:
    return scan.activations[selection.batch_idx, selection.token_idx, 0].numpy()


@pytest.mark.parametrize("method", ["elbow", "bucketed"])
def test_padding_is_never_selected(scan, method):
    selection = select_hits(scan, 7, "positive", HitSelectionConfig(method=method), seed=0)
    assert selection.token_idx.max() < 40


def test_positive_hits_are_above_the_threshold(scan):
    selection = select_hits(scan, 7, "positive", HitSelectionConfig(), seed=0)
    assert selection.threshold > 0
    assert (values_at(scan, selection) > selection.threshold).all()


def test_negative_hits_are_below_a_negative_threshold(scan):
    selection = select_hits(scan, 7, "negative", HitSelectionConfig(), seed=0)
    assert selection.threshold < 0
    assert (values_at(scan, selection) < selection.threshold).all()


def test_max_hits_caps_the_selection_and_the_seed_makes_it_repeatable(scan):
    hits = HitSelectionConfig(max_hits=5)
    first = select_hits(scan, 7, "positive", hits, seed=3)
    again = select_hits(scan, 7, "positive", hits, seed=3)
    assert len(first.batch_idx) == 5
    assert first.n_above_threshold > 5
    assert (first.batch_idx == again.batch_idx).all() and (first.token_idx == again.token_idx).all()


def test_bucketed_selection_labels_every_kept_hit(scan):
    selection = select_hits(scan, 7, "positive", HitSelectionConfig(method="bucketed", max_hits=60), seed=0)
    assert selection.bucket_ids is not None and selection.bucket_edges is not None
    assert len(selection.bucket_ids) == len(selection.batch_idx)
