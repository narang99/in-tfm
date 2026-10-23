import numpy as np

from in_tfm.hit_selection.picks import BucketedPick, ElbowPick
from in_tfm.hit_space import SelectionRecord, entropy_bits, load_pick, pick_arrays


def round_trip(pick: ElbowPick | BucketedPick) -> ElbowPick | BucketedPick:
    arrays = pick_arrays(pick)
    record = SelectionRecord(polarity="positive", n_candidates=3, pick_fields=pick.model_dump(exclude=set(arrays)))
    return load_pick(SelectionRecord.model_validate_json(record.model_dump_json()), arrays)


def test_an_elbow_pick_survives_the_saved_record():
    assert round_trip(ElbowPick(cutoff=2.5, elbow_idx=7)) == ElbowPick(cutoff=2.5, elbow_idx=7)


def test_a_bucketed_pick_keeps_its_arrays_outside_the_json():
    pick = BucketedPick(floor=1.0, bucket_ids=np.array([0, 1, 1]), bucket_edges=np.array([1.0, 2.0, 3.0]))
    loaded = round_trip(pick)
    assert isinstance(loaded, BucketedPick)
    assert loaded.floor == 1.0
    assert loaded.bucket_ids.tolist() == [0, 1, 1]
    assert loaded.bucket_edges.tolist() == [1.0, 2.0, 3.0]


def test_entropy_is_zero_for_one_cluster_and_one_bit_for_an_even_pair():
    assert entropy_bits(np.array([[1.0, 0.0], [0.5, 0.5]])).tolist() == [0.0, 1.0]
