import numpy as np
import pytest

from in_tfm.clustered_hits import (
    clustered_hits,
    load_clustered_hits,
    write_clustered_hits,
)


@pytest.fixture
def hits():
    return clustered_hits(
        sample_ids=["a", "b", "c"],
        batch_idx=np.array([2, 0, 2, 1]),
        token_idx=np.array([5, 0, 7, 3]),
        labels=np.array([1, -1, 1, 0]),
        activations=np.array([0.5, -2.0, 1.5, 3.0]),
    )


def test_each_hit_keeps_its_own_sample_and_token(hits):
    assert hits.sample_ids == ["c", "a", "c", "b"]
    assert hits.token_idxs == [5, 0, 7, 3]


def test_a_cluster_lists_its_hits_and_noise_is_not_a_cluster(hits):
    assert hits.cluster_rows(1) == [0, 2]
    assert hits.cluster_ids == [0, 1]


def test_hits_round_trip_as_json(tmp_path, hits):
    write_clustered_hits(tmp_path, hits)
    assert load_clustered_hits(tmp_path) == hits


def test_a_report_without_the_file_says_to_regenerate(tmp_path):
    with pytest.raises(FileNotFoundError, match="regenerating"):
        load_clustered_hits(tmp_path)
