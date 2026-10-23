import numpy as np

from in_tfm.pair_mixed_report import (
    ENTROPY_EDGES, MIN_HITS_PER_PAIR, cluster_means, entropy_bin, entropy_bin_label, pairs_with_hits, spread_by_share,
)


def split_hits(first: int, second: int, n: int, n_clusters: int = 4) -> np.ndarray:
    """`n` hits split 0.6 to the first cluster and 0.4 to the second."""
    rows = np.zeros((n, n_clusters))
    rows[:, first], rows[:, second] = 0.6, 0.4
    return rows


def test_a_pair_is_the_two_most_probable_clusters_in_either_order():
    probabilities = np.vstack([split_hits(0, 2, MIN_HITS_PER_PAIR), split_hits(2, 0, MIN_HITS_PER_PAIR)])
    [pair] = pairs_with_hits(probabilities)
    assert pair.clusters == (0, 2)
    assert len(pair.hits) == 2 * MIN_HITS_PER_PAIR


def test_pairs_come_most_hits_first_and_small_ones_are_dropped():
    probabilities = np.vstack([
        split_hits(0, 1, MIN_HITS_PER_PAIR), split_hits(2, 3, MIN_HITS_PER_PAIR + 5), split_hits(1, 3, MIN_HITS_PER_PAIR - 1),
    ])
    assert [pair.clusters for pair in pairs_with_hits(probabilities)] == [(2, 3), (0, 1)]


def test_a_hit_with_a_negligible_second_cluster_is_not_mixed():
    sure = np.zeros((MIN_HITS_PER_PAIR, 3))
    sure[:, 0], sure[:, 1] = 0.97, 0.03
    assert pairs_with_hits(sure) == []


def test_a_bin_is_sampled_from_mostly_one_cluster_to_mostly_the_other():
    hits = np.arange(100)
    share = np.linspace(0, 1, 100)
    picked = spread_by_share(hits, share, 5)
    assert picked.tolist() == [0, 25, 50, 74, 99]


def test_a_small_bin_is_shown_whole():
    assert spread_by_share(np.array([7, 3]), np.array([0.9, 0.1]), 8).tolist() == [3, 7]


def test_entropy_bins_cover_every_value_and_have_labels():
    bins = entropy_bin(np.array([0.2, 0.5, 1.2, 2.0, 5.0]))
    assert bins.tolist() == [0, 1, 2, 4, 4]
    assert [entropy_bin_label(b) for b in range(len(ENTROPY_EDGES) + 1)][-1] == "2.0 bits and above"


def test_a_cluster_mean_uses_confident_hits_and_falls_back_to_all_its_hits():
    hadamards = np.array([[0.0], [10.0], [4.0]])
    most_probable = np.array([0, 0, 1])
    confident = np.array([True, False, False])
    assert [m.item() for m in cluster_means(hadamards, most_probable, confident, [0, 1])] == [0.0, 4.0]
