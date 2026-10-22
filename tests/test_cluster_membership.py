import numpy as np
from pydantic import TypeAdapter

from in_tfm.cluster_membership import (
    NOISE_LABEL, ClusterProbabilities, HitMembership, Noise, most_probable_labels, one_hot_memberships,
)
from in_tfm.neuron_run import hit_memberships
from in_tfm.run_config import ClusteringConfig, DensityPeaksConfig, HdbscanConfig


def two_blobs() -> np.ndarray:
    rng = np.random.default_rng(0)
    blob_a = np.zeros((15, 6))
    blob_a[:, :2] = rng.uniform(1.0, 1.2, (15, 2))
    blob_b = np.zeros((15, 6))
    blob_b[:, 4:] = rng.uniform(1.0, 1.2, (15, 2))
    return np.vstack([blob_a, blob_b])


def test_one_hot_round_trips_through_the_most_probable_label():
    labels = np.array([2, NOISE_LABEL, 0, 1, 2])
    memberships = one_hot_memberships(labels)
    assert isinstance(memberships[1], Noise)
    assert memberships[0].probabilities.tolist() == [0, 0, 1]
    assert np.array_equal(most_probable_labels(memberships), labels)


def test_a_membership_is_picked_by_its_kind():
    adapter = TypeAdapter(HitMembership)
    assert isinstance(adapter.validate_python({"kind": "noise"}), Noise)
    parsed = adapter.validate_python({"kind": "probabilities", "probabilities": np.array([0.3, 0.7])})
    assert isinstance(parsed, ClusterProbabilities)


def test_the_most_probable_label_breaks_ties_toward_the_lower_cluster():
    assert most_probable_labels([ClusterProbabilities(probabilities=np.array([0.5, 0.5]))]).tolist() == [0]


def test_every_strategy_returns_one_probability_vector_per_hit():
    points = two_blobs()
    for strategy in (DensityPeaksConfig(k=3), HdbscanConfig(min_cluster_size=5)):
        memberships = hit_memberships(points, ClusteringConfig(strategy=strategy))
        assert len(memberships) == len(points)
        for m in memberships:
            if isinstance(m, ClusterProbabilities):
                assert np.isclose(m.probabilities.sum(), 1.0)
