import numpy as np
import pytest
import torch

from in_tfm.activation_range_plot import grid_dims_for_labels
from in_tfm.activation_range_figure import panels_needed, round_robin_panels, sorted_by_mean_descending
from in_tfm.activation_ranges import (
    PLOT_NAME,
    ActivationRanges,
    cluster_activations,
    sample_noise,
    write_activation_ranges,
)
from in_tfm.hit_selection import select_hits
from in_tfm.neuron_capture import ScanResult
from in_tfm.neuron_report import _activation_ranges_plot
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


@pytest.mark.parametrize("n_labels, panels", [(0, 1), (1, 1), (8, 1), (9, 2), (40, 5)])
def test_about_eight_clusters_share_a_panel(n_labels, panels):
    assert panels_needed(n_labels) == panels


def test_the_panels_fit_in_a_near_square_grid():
    assert grid_dims_for_labels(40) == (2, 3)


def test_round_robin_gives_every_panel_a_high_and_a_low_cluster():
    by_points = {i: [float(i)] for i in range(12)}
    ranked = list(sorted_by_mean_descending(by_points))
    assert ranked[0] == 11
    panels = round_robin_panels(ranked, 3)
    assert panels == [[11, 8, 5, 2], [10, 7, 4, 1], [9, 6, 3, 0]]


def test_the_plot_is_written_and_the_report_section_is_collapsed(tmp_path):
    rng = np.random.default_rng(0)
    ranges = ActivationRanges(
        clusters={i: rng.normal(i, 1, 30).tolist() for i in range(20)}, noise=rng.normal(0, 1, 100).tolist()
    )
    assert _activation_ranges_plot(tmp_path) == ""
    write_activation_ranges(tmp_path, ranges)
    section = _activation_ranges_plot(tmp_path)
    assert (tmp_path / PLOT_NAME).stat().st_size > 0
    assert PLOT_NAME in section
    assert "<details>" in section
