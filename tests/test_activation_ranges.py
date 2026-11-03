import numpy as np
import pytest
import torch

from in_tfm.activation_ranges import (
    ActivationRanges,
    cluster_activations,
    sample_noise,
    load_activation_ranges,
    write_activation_ranges,
)
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
    activations[:, 40:] = 100.0
    return ScanResult(
        sample_ids=[str(i) for i in range(BATCH)], neuron_idxs=[7], activations=activations, valid_mask=valid_mask
    )


@pytest.mark.parametrize("polarity", ["positive", "negative"])
def test_a_hits_activation_is_the_scans_signed_value_at_its_position(scan, polarity):
    selection = select_hits(scan, 7, polarity, HitSelectionConfig(), seed=0)
    expected = scan.activations[selection.batch_idx, selection.token_idx, 0].numpy()
    np.testing.assert_array_equal(selection.activations, expected)
    assert (selection.activations < 0).all() == (polarity == "negative")


def test_noise_is_sampled_from_valid_positions_only_and_repeatably(scan):
    noise = sample_noise(scan, 7, seed=1, size=300)
    assert len(noise) == 300
    assert noise.max() < 100.0
    np.testing.assert_array_equal(noise, sample_noise(scan, 7, seed=1, size=300))


def test_noise_cannot_be_larger_than_the_valid_positions(scan):
    assert len(sample_noise(scan, 7, seed=0, size=10**6)) == BATCH * 40


def test_clusters_get_the_activations_of_their_own_hits_and_noise_is_left_out():
    labels = np.array([0, 1, -1, 1])
    activations = np.array([1.0, 2.0, 3.0, 4.0])
    assert cluster_activations(labels, activations, [0, 1]) == {0: [1.0], 1: [2.0, 4.0]}


def test_ranges_round_trip_as_json_without_an_image(tmp_path):
    rng = np.random.default_rng(0)
    ranges = ActivationRanges(
        clusters={i: rng.normal(i, 1, 30).tolist() for i in range(20)}, noise=rng.normal(0, 1, 100).tolist()
    )
    assert load_activation_ranges(tmp_path) is None
    write_activation_ranges(tmp_path, ranges)
    assert load_activation_ranges(tmp_path) == ranges
