from in_tfm.report_meta import (
    BucketedPickMeta,
    ReportMeta,
    load_report_meta,
    write_report_meta,
)


def test_a_current_meta_round_trips(tmp_path):
    meta = ReportMeta(
        neuron_idx=7,
        pick=BucketedPickMeta(floor=-0.5, bucket_edges=[-0.5, -1.0]),
        n_hits=30,
        min_uniq_images_per_cluster=2,
        hadamard_shape=(8, 16),
        clusters={},
        sample_ids=[],
    )
    write_report_meta(tmp_path, meta)
    assert load_report_meta(tmp_path) == meta
    assert load_report_meta(tmp_path).threshold == -0.5
