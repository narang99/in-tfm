import json

import pytest

from in_tfm.report_meta import (
    BucketedPickMeta,
    ElbowPickMeta,
    LegacyReportMetaError,
    ReportMeta,
    load_report_meta,
    write_report_meta,
)

LEGACY_COMMON = {
    "neuron_idx": 7,
    "n_hits": 30,
    "min_uniq_images_per_cluster": 2,
    "n_clusters": 1,
    "hadamard_shape": [8, 16],
    "clusters": {"0": {"n_hits": 30, "n_unique_images": 4}},
    "sample_ids": ["a", "b"],
}


def write_legacy(tmp_path, **fields):
    (tmp_path / "meta.json").write_text(json.dumps({**LEGACY_COMMON, **fields}))


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


def test_an_old_meta_raises_and_names_the_migration(tmp_path):
    write_legacy(tmp_path, threshold=0.3)
    with pytest.raises(LegacyReportMetaError, match="migrate_file"):
        load_report_meta(tmp_path)


def test_a_meta_matching_neither_structure_raises_the_current_error(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"neuron_idx": 7}))
    with pytest.raises(Exception) as raised:
        load_report_meta(tmp_path)
    assert not isinstance(raised.value, LegacyReportMetaError)


def test_an_old_elbow_meta_migrates_to_an_elbow_pick(tmp_path):
    write_legacy(tmp_path, threshold=0.3, bucket_edges=None)
    migrated = ReportMeta.migrate_file(tmp_path / "meta.json")
    assert migrated.pick == ElbowPickMeta(cutoff=0.3)
    assert load_report_meta(tmp_path) == migrated


def test_an_old_bucketed_meta_migrates_to_a_bucketed_pick(tmp_path):
    write_legacy(tmp_path, threshold=0.0, bucket_edges=[0.0, 0.5, 1.0])
    migrated = ReportMeta.migrate_file(tmp_path / "meta.json")
    assert migrated.pick == BucketedPickMeta(floor=0.0, bucket_edges=[0.0, 0.5, 1.0])


def test_migrating_a_current_meta_leaves_the_file_untouched(tmp_path):
    write_legacy(tmp_path, threshold=0.3)
    ReportMeta.migrate_file(tmp_path / "meta.json")
    before = (tmp_path / "meta.json").read_text()
    ReportMeta.migrate_file(tmp_path / "meta.json")
    assert (tmp_path / "meta.json").read_text() == before
