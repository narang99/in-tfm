import json

import numpy as np
import pytest

from in_tfm.activation_view import (
    ACTIVATION_STEPS, ClusterGroups, TokenGrid, Vocabulary, cluster_options, grouped_probabilities, hit_clusters,
    load_token_grid, quantised, sample_records, save_token_grid, write_examples_page,
)
from in_tfm.report_links import check_links


def test_quantised_keeps_the_sign_and_clips_at_the_peak():
    assert quantised(np.array([1.0, -0.5, 2.0, 0.0]), 1.0) == [ACTIVATION_STEPS, -64, ACTIVATION_STEPS, 0]


def test_hit_clusters_keep_real_probabilities_most_probable_first():
    assert hit_clusters(np.array([0.1, 0.0, 0.89, 0.01])) == [(2, 89), (0, 10)]


def test_sample_records_drop_padding_and_reindex_hits_to_the_stored_tokens():
    valid = np.array([[False, True, True]])
    records = sample_records(
        ["s0"], [["<pad>", "▁the", "▁cat"]], np.array([[9.0, 0.5, 1.0]]), valid,
        batch_idx=np.array([0]), token_idx=np.array([2]), probabilities=np.array([[0.0, 1.0]]), peak=1.0,
        vocabulary=Vocabulary(),
    )
    assert len(records[0].tokens) == 2
    assert records[0].hits == [(1, [(1, 100)])]


def test_grouped_probabilities_sum_their_clusters_and_need_every_cluster():
    probabilities = np.array([[0.2, 0.3, 0.5]])
    grouped = grouped_probabilities(probabilities, ClusterGroups(names=["a", "b"], members=[[0, 2], [1]]))
    assert np.allclose(grouped, [[0.7, 0.3]])
    with pytest.raises(ValueError):
        grouped_probabilities(probabilities, ClusterGroups(names=["a"], members=[[0, 1]]))


def test_token_grid_round_trips(tmp_path):
    grid = TokenGrid(activations=np.ones((2, 3)), display_ids=np.arange(6).reshape(2, 3), valid=np.ones((2, 3), bool))
    save_token_grid(grid, tmp_path)
    loaded = load_token_grid(tmp_path)
    assert (loaded.display_ids == grid.display_ids).all() and loaded.valid.dtype == bool


def test_examples_page_links_land_and_its_data_holds_every_sample(tmp_path):
    (tmp_path / "clusters").mkdir()
    (tmp_path / "clusters" / "index.html").write_text('<a id="cluster-0"></a><a id="cluster-1"></a>')
    (tmp_path / "index.html").write_text("neuron")
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8]])
    write_examples_page(
        tmp_path, 7, ["a", "b"], [["▁x", "▁y"], ["▁z", "▁w"]], np.array([[0.1, -0.4], [0.2, 0.3]]),
        np.ones((2, 2), bool), np.array([0, 1]), np.array([1, 0]), probabilities,
        cluster_options(probabilities, "clusters/index.html"), "clusters/index.html",
    )
    data = json.loads((tmp_path / "examples_data.js").read_text().removeprefix("window.EXAMPLES = ").rstrip(";\n"))
    assert len(data["samples"]) == 2 and data["peak"] == pytest.approx(0.4)
    assert check_links(tmp_path).broken == []


def test_a_state_anchor_is_checked_for_its_file_only(tmp_path):
    (tmp_path / "examples.html").write_text("page")
    (tmp_path / "index.html").write_text('<a href="examples.html#cluster=3&min=0.5"></a><a href="gone.html#cluster=1"></a>')
    assert [b.reference for b in check_links(tmp_path).broken] == ["gone.html#cluster=1"]
