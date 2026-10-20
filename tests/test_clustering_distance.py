import numpy as np

from in_tfm.clustering import cluster_labels


def test_weighted_jaccard_separates_two_groups_with_different_dims():
    rng = np.random.default_rng(0)
    group_a = np.zeros((8, 20))
    group_a[:, :5] = rng.uniform(1.0, 1.2, (8, 5))
    group_b = np.zeros((8, 20))
    group_b[:, 10:15] = rng.uniform(1.0, 1.2, (8, 5))
    labels = cluster_labels(np.vstack([group_a, group_b]), min_cluster_size=4, min_samples=2, distance="weighted_jaccard")
    assert len(set(labels[:8])) == 1 and len(set(labels[8:])) == 1
    assert labels[0] != labels[8] and -1 not in labels
