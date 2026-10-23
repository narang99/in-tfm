import numpy as np

from in_tfm.bucket_report import MIN_SHOWN_PROBABILITY, SHOWN_CLUSTERS, bucket_slug, shown_clusters


def test_a_slug_is_a_file_safe_name_and_tiers_stay_distinct():
    assert bucket_slug("clusters 0 + 4") == "clusters_0_4"
    assert bucket_slug("cluster 0 >= 0.9") != bucket_slug("cluster 0 >= 0.8")


def test_the_shown_clusters_are_the_most_probable_above_the_minimum():
    probabilities = np.array([0.5, 0.3, 0.1, 0.04, 0.03, 0.03])
    assert shown_clusters(probabilities) == [0, 1, 2]
    assert len(shown_clusters(np.full(10, 0.1))) == SHOWN_CLUSTERS


def test_the_most_probable_cluster_is_always_shown():
    probabilities = np.full(40, 1 / 40)
    assert probabilities.max() < MIN_SHOWN_PROBABILITY
    assert len(shown_clusters(probabilities)) == 1
