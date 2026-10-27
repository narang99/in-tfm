import numpy as np
import torch

from in_tfm.activation_ranges import ActivationRanges, write_activation_ranges
from in_tfm.density_peaks import calculate_probs, density_peak_forest, forest_trees
from in_tfm.hit_selection import ElbowPick, HitSelection
from in_tfm.report_meta import ClusterMeta, ElbowPickMeta, ReportMeta
from in_tfm.report_page.data import touching_counts
from in_tfm.report_page import (
    forest_data, load_forest_data, load_selection_curve, page_parts, save_forest_data, save_selection_curve,
    selection_curve,
)


def two_blobs(n: int = 60) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.concatenate([rng.normal(0, 0.1, (n, 6)) + 2, rng.normal(0, 0.1, (n, 6)) - 2])


def a_forest():
    forest = density_peak_forest(two_blobs(), k=5, distance="euclidean")
    trees = forest_trees(forest)
    return forest_data(forest, trees, calculate_probs(forest, trees, 0.8))


def a_selection(n_positive: int = 10_000) -> HitSelection:
    magnitudes = torch.linspace(0, 1, n_positive) ** 4
    return HitSelection(
        polarity="positive", batch_idx=np.zeros(3, int), token_idx=np.arange(3), activations=np.ones(3),
        n_candidates=3, sorted_magnitudes=magnitudes, pick=ElbowPick(cutoff=0.5, elbow_idx=9_000),
    )


def a_meta() -> ReportMeta:
    return ReportMeta(
        neuron_idx=0, pick=ElbowPickMeta(cutoff=0.5), n_hits=3, min_uniq_images_per_cluster=1, hadamard_shape=(2, 3),
        clusters={0: ClusterMeta(n_hits=2, n_unique_images=1), 1: ClusterMeta(n_hits=1, n_unique_images=1)},
        sample_ids=["a"],
    )


def test_a_long_curve_is_thinned_but_keeps_its_tail(tmp_path):
    save_selection_curve(tmp_path, selection_curve(a_selection()))
    curve = load_selection_curve(tmp_path)
    assert len(curve.index) < 10_000
    assert curve.index[-1] == 9_999 and curve.index[-4000] == 6_000
    assert curve.elbow_idx == 9_000 and curve.bucket_edges is None and not curve.negative


def test_forest_data_round_trips_and_cuts_root_links(tmp_path):
    data = a_forest()
    save_forest_data(tmp_path, data)
    loaded = load_forest_data(tmp_path)
    assert loaded.n_trees == data.n_trees == 2
    assert (loaded.parent_in_tree[loaded.roots] == -1).all()
    assert np.array_equal(loaded.tree, data.tree) and loaded.cut_rho == data.cut_rho
    assert loaded.probabilities.shape == (len(loaded.rho), 2)


def test_the_page_numbers_only_the_sections_it_has_data_for(tmp_path):
    assert page_parts(tmp_path, a_meta()).n_sections == 0
    save_selection_curve(tmp_path, selection_curve(a_selection()))
    write_activation_ranges(tmp_path, ActivationRanges(clusters={0: [1.0, 2.0], 1: [3.0]}, noise=[0.1, -0.2]))
    parts = page_parts(tmp_path, a_meta())
    assert parts.n_sections == 2 and parts.has_charts
    assert "Plotly.newPlot" in parts.sections_html
    assert "positions in the tail" in parts.header_html


def test_a_forest_report_gets_four_forest_tabs_and_an_up_close_slice(tmp_path):
    points = two_blobs()
    forest = density_peak_forest(points, k=5, distance="euclidean")
    trees = forest_trees(forest)
    data = forest_data(forest, trees, calculate_probs(forest, trees, 0.8), points, "euclidean")
    assert len(data.up_close_points) > 0 and data.up_close_xy.shape == (len(data.up_close_points), 2)
    assert set(data.roots) <= set(data.up_close_points), "every sampled path ends at its root"
    save_forest_data(tmp_path, data)
    parts = page_parts(tmp_path, a_meta())
    for tab in ["Decision graph", "Peak separation", "Forest up close", "Forest landscape"]:
        assert tab in parts.sections_html
    assert 'data-plot="forest-up-close"' in parts.sections_html


def test_touching_counts_are_symmetric_and_mostly_inside_each_blob():
    counts = touching_counts(a_forest())
    assert (counts == counts.T).all()
    assert counts.trace() > 0 and counts[0, 1] == 0, "the blobs are far apart, so no mutual neighbours cross"
