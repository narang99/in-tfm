from pathlib import Path

import numpy as np
import pytest
import torch

from in_tfm.dependencies.collect import write_cluster_dependencies
from in_tfm.dependencies.firings import ClusterMatch, WriterReportRef
from in_tfm.dependencies.stats import StatsSettings, dependency_stats
from in_tfm.dependencies.view import CumulativeShareTrim, NoTrim, ViewPolicy
from in_tfm.dependency_page import render_dependency_page, write_dependency_page
from in_tfm.dependency_page.cards import dependency_cards, dependency_title
from in_tfm.dependency_page.graph import graph_html, origin_edges
from in_tfm.dependency_page.sparkline import sparkline
from in_tfm.neuron_report import dependency_link_html, save_hits
from in_tfm.presenters import ClusterHit
from in_tfm.report_meta import ClusterMeta, ReportMeta, write_report_meta


def test_only_frequent_dependencies_get_cards_and_the_rest_are_one_offs(collected):
    stats = dependency_stats(collected)
    cards = dependency_cards(collected, stats, ViewPolicy(trim=NoTrim()))
    assert [card.dim for card in cards.cards] == [0]
    assert [d.dim for d in cards.one_offs] == [1]
    assert cards.titles[0] == "dim 0"


def test_a_dim_with_one_writer_is_named_after_it(collected):
    single = collected.firings[2].model_copy(update={"n_writers": 1})
    assert dependency_title(1, [single]) == "up:1 at token 0"


def test_a_card_lists_its_writers_by_share_and_says_how_it_was_cut(collected):
    stats = dependency_stats(collected)
    policy = ViewPolicy(trim=CumulativeShareTrim(mass=0.8))
    (card,) = dependency_cards(collected, stats, policy).cards
    assert [row.label for row in card.writers.shown] == ["up:0"]
    assert [row.label for row in card.writers.rest] == ["up:1"]
    html = render_dependency_page(collected, StatsSettings(), policy)
    assert "shown until 80% of the total share" in html
    assert "the other 1 writers" in html


def test_the_sparkline_marks_the_dependency_on_its_own_sign(collected):
    stats = dependency_stats(collected)
    line = sparkline(stats.concentration, 0)
    assert line.marker is not None and line.marker.sign == 1
    assert line.label == "+ rank 1: 100% (+100%)"


def test_the_page_is_written_next_to_the_firings(collected, tmp_path):
    collected.origin.report_dir = tmp_path
    write_cluster_dependencies(collected)
    path = write_dependency_page(tmp_path, collected.origin.cluster_id)
    assert path == tmp_path / "dependencies" / "cluster_0" / "index.html"
    assert "head:0, cluster 0" in path.read_text()


class RecordingPresenter:
    """Draws a hit count, and writes a file the way a real presenter does."""

    def render(self, hits, artifacts, hadamard_shape) -> str:
        artifacts.path("hits.html").write_text("drawn")
        return f"<p>{len(hits)} hits of shape {hadamard_shape}</p>"

    def page_controls(self) -> str:
        return '<div id="controls"></div>'


def fake_writer_report(report_dir: Path, cluster_id: int) -> None:
    report_dir.mkdir(parents=True)
    write_report_meta(
        report_dir,
        ReportMeta(
            neuron_idx=0,
            picks=[],
            n_hits=2,
            min_uniq_images_per_cluster=1,
            hadamard_shape=(2, 1),
            clusters={cluster_id: ClusterMeta(n_hits=2, n_unique_images=2)},
            sample_ids=["0", "1"],
        ),
    )
    hit = ClusterHit(
        sample_id="0",
        token_idx=0,
        relevance=np.zeros(2),
        model_input=torch.zeros(2),
        hadamard=np.ones(2),
    )
    cluster_dir = report_dir / f"cluster_{cluster_id}"
    cluster_dir.mkdir()
    save_hits([hit, hit], cluster_dir)


def test_a_matched_writer_gets_one_bar_per_cluster_opening_to_its_own_hits(
    collected, tmp_path
):
    collected.origin.report_dir = tmp_path / "head" / "neuron_0"
    writer_dir = tmp_path / "up" / "neuron_0"
    fake_writer_report(writer_dir, cluster_id=5)
    collected.writer_reports = [
        WriterReportRef(layer="up", neuron=0, report_dir=writer_dir)
    ]
    for firing in collected.firings:
        for writer in firing.writers:
            if writer.writer.neuron == 0:
                writer.match = ClusterMatch(cluster_id=5, distance=0.25)
    stats = dependency_stats(collected)
    policy = ViewPolicy(trim=NoTrim())
    presenter = RecordingPresenter()
    (card,) = dependency_cards(collected, stats, policy, presenter).cards
    (panel,) = card.clusters
    assert panel.label == "up:0"
    (bar,) = panel.bars
    assert (bar.share.cluster_id, bar.percent) == (5, 100.0)
    assert bar.body == "<p>2 hits of shape (2, 1)</p>"
    page_dir = tmp_path / "head" / "neuron_0" / "dependencies" / "cluster_0"
    assert (page_dir / "up_0_cluster_5" / "hits.html").read_text() == "drawn"
    html = render_dependency_page(collected, policy=policy, presenter=presenter)
    assert '<template class="lazy-body"><p>2 hits' in html
    assert '<div id="controls"></div>' in html


def test_without_a_presenter_the_bars_do_not_open(collected, tmp_path):
    collected.origin.report_dir = tmp_path / "head" / "neuron_0"
    collected.writer_reports = [
        WriterReportRef(layer="up", neuron=0, report_dir=tmp_path / "up")
    ]
    for firing in collected.firings:
        for writer in firing.writers:
            if writer.writer.neuron == 0:
                writer.match = ClusterMatch(cluster_id=5, distance=0.25)
    stats = dependency_stats(collected)
    (card,) = dependency_cards(collected, stats, ViewPolicy(trim=NoTrim())).cards
    assert card.clusters[0].bars[0].body is None


def matched_collection(collected, tmp_path):
    collected.origin.report_dir = tmp_path / "head" / "neuron_0"
    collected.writer_reports = [
        WriterReportRef(
            layer="up",
            neuron=n,
            report_dir=tmp_path / "up" / f"neuron_{n}",
        )
        for n in (0, 1)
    ]
    for firing in collected.firings:
        for writer in firing.writers:
            writer.match = ClusterMatch(
                cluster_id=writer.writer.neuron + 3, distance=0.1
            )
    return collected


def test_an_origin_clusters_graph_edges_share_its_matched_contribution(
    collected, tmp_path
):
    collected = matched_collection(collected, tmp_path)
    edges = {e.target.label: e for e in origin_edges(collected, tmp_path)}
    assert set(edges) == {"up:0 c3", "up:1 c4"}
    assert sum(e.share for e in edges.values()) == pytest.approx(1.0)
    assert edges["up:0 c3"].href == "up/neuron_0/index.html#cluster-3"
    assert "head:0 c0" in graph_html([collected], tmp_path, ViewPolicy())


def test_a_cluster_links_to_its_dependency_page_only_once_it_exists(tmp_path):
    assert dependency_link_html(tmp_path, 2) == ""
    page = tmp_path / "dependencies" / "cluster_2" / "index.html"
    page.parent.mkdir(parents=True)
    page.write_text("")
    assert 'href="dependencies/cluster_2/index.html"' in dependency_link_html(
        tmp_path, 2
    )
