from in_tfm.dependencies.collect import write_cluster_dependencies
from in_tfm.dependencies.stats import StatsSettings, dependency_stats
from in_tfm.dependencies.view import CumulativeShareTrim, NoTrim, ViewPolicy
from in_tfm.dependency_page import render_dependency_page, write_dependency_page
from in_tfm.dependency_page.cards import dependency_cards, dependency_title
from in_tfm.dependency_page.sparkline import sparkline


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
