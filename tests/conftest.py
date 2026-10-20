import numpy as np
import torch
from pydantic import BaseModel

from in_tfm.activation_ranges import RANGES_NAME, ActivationRanges
from in_tfm.cluster_units import HitTable, write_hit_table
from in_tfm.report_meta import ClusterMeta, ElbowPickMeta, ReportMeta, write_report_meta


class FakeCluster(BaseModel):
    mean: list[float]
    activations: list[float]
    samples: list[int]
    """The sample index of each hit, one per activation."""


DEFAULT_CLUSTERS = {
    5: FakeCluster(mean=[1.0, 0.0, 0.0], activations=[1.0, 2.0, 3.0], samples=[0, 0, 1]),
    2: FakeCluster(mean=[0.0, 1.0, 0.0], activations=[10.0, 20.0], samples=[1, 2]),
    9: FakeCluster(mean=[0.0, 0.0, 1.0], activations=[7.0], samples=[3]),
}


def fake_report(report_dir, clusters=None, with_ranges=True, with_table=True):
    """A report folder as the pipeline writes it, in the order `clusters` is given."""
    clusters = clusters or DEFAULT_CLUSTERS
    labels = np.array([cid for cid, c in clusters.items() for _ in c.activations])
    batch_idx = np.array([s for c in clusters.values() for s in c.samples])
    activations = np.array([a for c in clusters.values() for a in c.activations])
    write_report_meta(report_dir, ReportMeta(
        neuron_idx=0, pick=ElbowPickMeta(cutoff=0.4), n_hits=len(labels), min_uniq_images_per_cluster=1,
        hadamard_shape=(1, len(next(iter(clusters.values())).mean)), sample_ids=[str(i) for i in range(batch_idx.max() + 1)],
        clusters={cid: ClusterMeta(n_hits=len(c.activations), n_unique_images=len(set(c.samples))) for cid, c in clusters.items()},
    ))
    for cid, cluster in clusters.items():
        (report_dir / f"cluster_{cid}").mkdir()
        torch.save(torch.tensor(cluster.mean), report_dir / f"cluster_{cid}" / "mean.pt")
    if with_ranges:
        ranges = ActivationRanges(clusters={cid: c.activations for cid, c in clusters.items()}, noise=[0.0])
        (report_dir / RANGES_NAME).write_text(ranges.model_dump_json())
    if with_table:
        table = HitTable(batch_idx=batch_idx, token_idx=np.zeros(len(labels), dtype=int), labels=labels, activations=activations)
        write_hit_table(report_dir, table)
    return report_dir
