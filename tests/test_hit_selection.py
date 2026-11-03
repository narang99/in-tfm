import numpy as np
import pytest
import torch

from in_tfm.hit_selection import BucketedPick, TailSelection, select_hits
from in_tfm.hit_selection.selection import tail_budgets
from in_tfm.neuron_capture import ScanResult
from in_tfm.run_config import HitSelectionConfig

BATCH, SEQ = 40, 50


@pytest.fixture
def scan() -> ScanResult:
    generator = torch.Generator().manual_seed(0)
    activations = torch.randn(BATCH, SEQ, 1, generator=generator)
    valid_mask = torch.ones(BATCH, SEQ, dtype=torch.bool)
    valid_mask[:, 40:] = False
    activations[:, 40:] = (
        100.0  # would dominate the tail if padding were not masked out
    )
    return ScanResult(
        sample_ids=[str(i) for i in range(BATCH)],
        neuron_idxs=[7],
        activations=activations,
        valid_mask=valid_mask,
    )


def values_at(scan: ScanResult, selection) -> np.ndarray:
    return scan.activations[selection.batch_idx, selection.token_idx, 0].numpy()


@pytest.mark.parametrize("method", ["elbow", "bucketed"])
def test_padding_is_never_selected(scan, method):
    selection = select_hits(
        scan, 7, "positive", HitSelectionConfig(method=method), seed=0
    )
    assert selection.token_idx.max() < 40


def only_tail(selection) -> TailSelection:
    [tail] = selection.tails
    return tail


def test_positive_hits_are_above_the_threshold(scan):
    selection = select_hits(scan, 7, "positive", HitSelectionConfig(), seed=0)
    threshold = only_tail(selection).threshold
    assert threshold > 0
    assert (values_at(scan, selection) > threshold).all()


def test_negative_hits_are_below_a_negative_threshold(scan):
    selection = select_hits(scan, 7, "negative", HitSelectionConfig(), seed=0)
    threshold = only_tail(selection).threshold
    assert threshold < 0
    assert (values_at(scan, selection) < threshold).all()


def test_max_hits_caps_the_selection_and_the_seed_makes_it_repeatable(scan):
    hits = HitSelectionConfig(max_hits=5)
    first = select_hits(scan, 7, "positive", hits, seed=3)
    again = select_hits(scan, 7, "positive", hits, seed=3)
    assert len(first.batch_idx) == 5
    assert first.n_candidates > 5
    assert (first.batch_idx == again.batch_idx).all() and (
        first.token_idx == again.token_idx
    ).all()


@pytest.mark.parametrize("method", ["elbow", "bucketed"])
def test_signed_selection_takes_each_tail_beyond_its_own_threshold(scan, method):
    hits = HitSelectionConfig(method=method, max_hits=200, bucket_floor_fraction=0.3)
    selection = select_hits(scan, 7, "signed", hits, seed=0)
    negative, positive = selection.tails
    assert (negative.tail, positive.tail) == ("negative", "positive")
    values = values_at(scan, selection)
    assert ((values < negative.threshold) | (values > positive.threshold)).all()
    assert (values < 0).sum() == len(negative.batch_idx)
    assert len(values) == 200
    assert selection.token_idx.max() < 40


def test_signed_hits_are_in_row_major_order(scan):
    hits = HitSelectionConfig(method="bucketed", max_hits=200)
    selection = select_hits(scan, 7, "signed", hits, seed=0)
    flat = selection.batch_idx * SEQ + selection.token_idx
    assert (np.diff(flat) > 0).all()


def test_the_budget_follows_each_tails_candidates():
    assert tail_budgets([300, 100], 200) == [150, 50]
    assert tail_budgets([2, 1], 2) == [2, 0]
    assert tail_budgets([30, 10], 200) == [30, 10]


@pytest.mark.parametrize("polarity", ["positive", "negative"])
def test_a_one_tailed_bucketed_selection_never_takes_the_other_tail(scan, polarity):
    hits = HitSelectionConfig(method="bucketed", max_hits=10_000)
    selection = select_hits(scan, 7, polarity, hits, seed=0)
    values = values_at(scan, selection)
    assert ((values > 0) if polarity == "positive" else (values < 0)).all()


def test_bucketed_selection_labels_every_kept_hit(scan):
    selection = select_hits(
        scan, 7, "positive", HitSelectionConfig(method="bucketed", max_hits=60), seed=0
    )
    pick = only_tail(selection).pick
    assert isinstance(pick, BucketedPick)
    assert len(pick.bucket_ids) == len(selection.batch_idx)


def test_a_neuron_that_never_fires_in_a_tail_is_refused(scan):
    relu_scan = scan.model_copy(update={"activations": scan.activations.clamp(min=0)})
    with pytest.raises(ValueError, match="never fires"):
        select_hits(relu_scan, 7, "negative", HitSelectionConfig(), seed=0)
    signed = select_hits(relu_scan, 7, "signed", HitSelectionConfig(), seed=0)
    assert [tail.tail for tail in signed.tails] == ["positive"]
