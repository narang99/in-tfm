import pytest

from in_tfm.dependencies.collect import write_cluster_dependencies
from in_tfm.dependencies.firings import ClusterMatch, WriterReportRef
from in_tfm.dependencies.stats import StatsSettings, dependency_stats
from in_tfm.dependencies.view import CumulativeShareTrim, NoTrim, ViewPolicy
from in_tfm.dependency_page import render_dependency_page, write_dependency_page
from in_tfm.dependency_page.cards import dependency_cards, dependency_title
from in_tfm.dependency_page.graph import graph_html, origin_edges
from in_tfm.dependency_page.sparkline import sparkline
from in_tfm.neuron_report import dependency_link_html


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


def test_matched_writers_get_a_cluster_panel_linking_to_their_own_report(
    collected, tmp_path
):
    collected.origin.report_dir = tmp_path / "head" / "neuron_0"
    collected.writer_reports = [
        WriterReportRef(
            layer="up",
            neuron=0,
            report_dir=tmp_path / "up" / "neuron_0",
            tile_shape=(2, 1, 1),
        )
    ]
    for firing in collected.firings:
        for writer in firing.writers:
            if writer.writer.neuron == 0:
                writer.match = ClusterMatch(
                    cluster_id=5,
                    distance=0.25,
                    hadamard=[1.0, -1.0],
                    closest_hadamard=[0.5, 0.0],
                )
    stats = dependency_stats(collected)
    (card,) = dependency_cards(collected, stats, ViewPolicy(trim=NoTrim())).cards
    (panel,) = card.clusters
    assert panel.label == "up:0"
    assert [row.share.cluster_id for row in panel.rows] == [5]
    assert panel.rows[0].href == "../../../../up/neuron_0/index.html#cluster-5"
    assert len(panel.pairs) == 2
    assert panel.pairs[0].image_uri.startswith("data:image/png;base64,")
    assert "firing by its own cluster" in render_dependency_page(collected)


def matched_collection(collected, tmp_path):
    collected.origin.report_dir = tmp_path / "head" / "neuron_0"
    collected.writer_reports = [
        WriterReportRef(
            layer="up",
            neuron=n,
            report_dir=tmp_path / "up" / f"neuron_{n}",
            tile_shape=(2,),
        )
        for n in (0, 1)
    ]
    for firing in collected.firings:
        for writer in firing.writers:
            writer.match = ClusterMatch(
                cluster_id=writer.writer.neuron + 3,
                distance=0.1,
                hadamard=[0.0, 0.0],
                closest_hadamard=[0.0, 0.0],
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
