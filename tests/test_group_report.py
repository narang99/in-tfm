import re

import numpy as np
import pytest
import torch

from conftest import FakeCluster, fake_report
from in_tfm.cluster_merging import merge_report_clusters
from in_tfm.neuron_report import MissingRenderCacheError, render_report, save_hits
from in_tfm.presenters import ClusterHit
from in_tfm.run_config import MergeClustersConfig


class StubPresenter:
    def __init__(self) -> None:
        self.calls = 0

    def page_controls(self) -> str:
        return "<p>controls</p>"

    def render(self, hits, artifacts, hadamard_shape) -> str:
        self.calls += 1
        return f"<p>body of {artifacts.name} with {len(hits)} hits</p>"


def cluster(mean, median, n=3, first_sample=0):
    return FakeCluster(mean=mean, activations=[median] * n, samples=list(range(first_sample, first_sample + n)))


CLUSTERS = {
    0: cluster([1.0, 0.0, 0.0], 3.0, n=4),
    1: cluster([1.0, 0.01, 0.0], 2.9, first_sample=4),
    2: cluster([0.0, 1.0, 0.0], 1.0, first_sample=7),
}


@pytest.fixture
def report(tmp_path):
    report = fake_report(tmp_path, CLUSTERS)
    rng = np.random.default_rng(0)
    for cid, c in CLUSTERS.items():
        hits = [
            ClusterHit(sample_id=str(s), token_idx=0, relevance=np.ones((1, 3)), model_input=torch.zeros(1, 3),
                       hadamard=np.array(c.mean) + rng.normal(scale=0.01, size=3))
            for s in c.samples
        ]
        save_hits(hits, report / f"cluster_{cid}")
    return report


def section_order(html: str) -> list[str]:
    return re.findall(r'id="(group-\d+|cluster-\d+)"', html)


def test_a_report_without_a_grouping_renders_as_before(report):
    html = render_report(report, StubPresenter()).read_text()
    assert section_order(html) == ["cluster-0", "cluster-1", "cluster-2"]
    assert "groups:" not in html and "in 1 groups" not in html


def test_a_grouped_report_nests_each_groups_clusters_in_its_section(report):
    merge_report_clusters(report, MergeClustersConfig(method="thresholds"), seed=0)
    html = render_report(report, StubPresenter()).read_text()
    assert section_order(html) == ["group-0", "cluster-0", "cluster-1", "group-1", "cluster-2"]
    assert "3 clusters in 2 groups" in html
    assert (report / "group_activation_ranges.jpg").exists() and (report / "group_pca.png").exists()


def test_a_kmeans_grouping_also_draws_its_sweep(report):
    merge_report_clusters(report, MergeClustersConfig(method="kmeans"), seed=0)
    render_report(report, StubPresenter())
    assert (report / "kmeans_sweep.png").exists()


def test_a_rendered_report_can_be_regrouped_without_a_presenter(report):
    presenter = StubPresenter()
    render_report(report, presenter)
    merge_report_clusters(report, MergeClustersConfig(method="thresholds"), seed=0)
    html = render_report(report).read_text()
    assert presenter.calls == 3
    assert "body of cluster_1 with 3 hits" in html and "<p>controls</p>" in html
    assert section_order(html)[0] == "group-0"


def test_a_report_never_rendered_needs_a_presenter(report):
    with pytest.raises(MissingRenderCacheError):
        render_report(report)
