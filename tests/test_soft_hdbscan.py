import numpy as np
from sklearn.metrics import pairwise_distances

from in_tfm.soft_hdbscan import soft_clustering_precomputed


def three_blobs() -> np.ndarray:
    rng = np.random.default_rng(0)
    centres = np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0]])
    return np.concatenate([rng.normal(c, 0.4, (60, 3)) for c in centres] + [rng.uniform(-2, 5, (15, 3))])


def label_mapping(ours: np.ndarray, theirs: np.ndarray) -> dict[int, int]:
    """The two fits may number their clusters differently, so each of ours maps to the package's majority label."""
    return {int(o): int(np.bincount(theirs[ours == o] + 1).argmax() - 1) for o in np.unique(ours) if o >= 0}


def test_precomputed_soft_clustering_matches_the_package_on_euclidean_data():
    import hdbscan

    points = three_blobs()
    package = hdbscan.HDBSCAN(
        min_cluster_size=10, cluster_selection_method="leaf", prediction_data=True, algorithm="generic"
    ).fit(points)
    expected = hdbscan.all_points_membership_vectors(package)
    ours = soft_clustering_precomputed(pairwise_distances(points), 10, "leaf", None)
    column_of = label_mapping(ours.labels, package.labels_)
    assert np.array_equal(np.array([column_of.get(l, -1) for l in ours.labels]), package.labels_)
    reordered = np.empty_like(ours.probabilities)
    reordered[:, [column_of[c] for c in range(ours.probabilities.shape[1])]] = ours.probabilities
    assert np.allclose(reordered, expected, atol=1e-9)


def test_rows_sum_to_the_chance_of_being_in_any_cluster():
    ours = soft_clustering_precomputed(pairwise_distances(three_blobs()), 10, "eom", None)
    sums = ours.probabilities.sum(axis=1)
    assert (sums <= 1 + 1e-9).all() and (sums > 0).all()
    members = ours.labels >= 0
    assert (ours.probabilities[members].argmax(axis=1) == ours.labels[members]).mean() > 0.9


def test_the_pipeline_gives_soft_hdbscan_hits_a_row_that_sums_to_one_and_a_threshold():
    from in_tfm.cluster_membership import ClusterProbabilities, Noise
    from in_tfm.neuron_run import cluster_hits, max_entropy_of
    from in_tfm.run_config import ClusteringConfig, HdbscanConfig

    clustering = ClusteringConfig(
        distance="weighted_jaccard", normalise=False,
        strategy=HdbscanConfig(min_cluster_size=10, soft=True, max_entropy=0.4),
    )
    memberships = cluster_hits(np.abs(three_blobs()) + 0.01, clustering).memberships
    rows = [m.probabilities for m in memberships if isinstance(m, ClusterProbabilities)]
    assert rows and all(np.isclose(r.sum(), 1) for r in rows)
    assert any(isinstance(m, Noise) for m in memberships)
    assert max_entropy_of(clustering) == 0.4
    assert max_entropy_of(ClusteringConfig(strategy=HdbscanConfig())) is None
