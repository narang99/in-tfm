import json

from in_tfm.report_meta import (
    BucketedPickMeta,
    ElbowPickMeta,
    ReportMeta,
    load_report_meta,
    write_report_meta,
)


def test_a_current_meta_round_trips(tmp_path):
    meta = ReportMeta(
        neuron_idx=7,
        picks=[BucketedPickMeta(tail="negative", floor=-0.5, bucket_edges=[0.5, 1.0])],
        n_hits=30,
        min_uniq_images_per_cluster=2,
        hadamard_shape=(8, 16),
        clusters={},
        sample_ids=[],
    )
    write_report_meta(tmp_path, meta)
    assert load_report_meta(tmp_path) == meta
    assert load_report_meta(tmp_path).thresholds == [-0.5]


def test_a_version_2_meta_takes_its_tail_from_the_sign_of_its_threshold(tmp_path):
    (tmp_path / "meta.json").write_text(
        json.dumps(
            {
                "version": 2,
                "neuron_idx": 7,
                "pick": {"method": "elbow", "cutoff": -0.25},
                "n_hits": 30,
                "min_uniq_images_per_cluster": 2,
                "hadamard_shape": [8, 16],
                "clusters": {},
                "sample_ids": [],
            }
        )
    )
    meta = load_report_meta(tmp_path)
    assert meta.picks == [ElbowPickMeta(tail="negative", cutoff=-0.25)]
