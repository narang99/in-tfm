import numpy as np

from in_tfm.clustering import cluster_labels, mass_overlap_distances, mass_overlap_scores, mass_sets


def test_mass_set_takes_largest_dims_until_the_mass_is_reached():
    row = np.array([[5.0, 3.0, 1.0, 1.0]])
    assert mass_sets(row, 0.5).tolist() == [[True, False, False, False]]
    assert mass_sets(row, 0.8).tolist() == [[True, True, False, False]]
    assert mass_sets(row, 0.9).tolist() == [[True, True, True, False]]


def test_mass_set_of_a_row_with_nothing_positive_is_empty():
    assert not mass_sets(np.array([[0.0, -1.0, 0.0]]), 0.9).any()


def test_score_matches_positive_with_positive_and_negative_with_negative():
    a = np.array([4.0, 0.0, -4.0, 0.0])
    b = np.array([4.0, 0.0, 0.0, -4.0])
    flipped = -a
    scores = mass_overlap_scores(np.stack([a, b, flipped]), 0.9)
    assert scores[0, 0] == 2
    assert scores[0, 1] == 1
    assert scores[0, 2] == 0


def test_distance_is_the_best_pair_score_minus_the_score_with_zero_diagonal():
    points = np.array([
        [3.0, 3.0, 3.0, 0.0, -1.0],
        [3.0, 3.0, 0.0, 3.0, -1.0],
        [0.0, 0.0, 3.0, 3.0, 0.0],
    ])
    scores = mass_overlap_scores(points, 1.0)
    distances = mass_overlap_distances(points, 1.0)
    best = max(scores[0, 1], scores[0, 2], scores[1, 2])
    assert np.all(np.diag(distances) == 0)
    assert distances[0, 1] == best - scores[0, 1]
    assert distances[0, 2] == best - scores[0, 2]


def test_mass_overlap_separates_two_groups_with_different_dims():
    rng = np.random.default_rng(0)
    group_a = np.zeros((8, 20))
    group_a[:, :5] = rng.uniform(1.0, 1.2, (8, 5))
    group_b = np.zeros((8, 20))
    group_b[:, 10:15] = rng.uniform(1.0, 1.2, (8, 5))
    labels = cluster_labels(np.vstack([group_a, group_b]), min_cluster_size=4, min_samples=2, distance="mass_overlap")
    assert len(set(labels[:8])) == 1 and len(set(labels[8:])) == 1
    assert labels[0] != labels[8] and -1 not in labels
