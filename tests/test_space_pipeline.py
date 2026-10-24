import numpy as np

from in_tfm.cluster_membership import NOISE_LABEL
from in_tfm.space_pipeline import clusters_without_confident_hits, report_labels


def test_confident_hits_keep_their_cluster_and_mixed_ones_are_noise():
    probabilities = np.array([[0.99, 0.01, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0], [0.01, 0.99, 0.0]])
    assert report_labels(probabilities, max_entropy=0.5).tolist() == [0, NOISE_LABEL, 2, 1]


def test_a_cluster_with_no_confident_hit_keeps_the_hits_it_is_most_probable_for():
    probabilities = np.array([[0.99, 0.01, 0.0], [0.0, 0.6, 0.4], [0.0, 0.45, 0.55], [0.0, 0.01, 0.99]])
    assert clusters_without_confident_hits(probabilities, max_entropy=0.5) == [1]
    labels = report_labels(probabilities, max_entropy=0.5)
    assert labels.tolist() == [0, 1, NOISE_LABEL, 2]
    assert set(labels.tolist()) - {NOISE_LABEL} == {0, 1, 2}
