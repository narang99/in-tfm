from in_tfm.activation_ranges import ActivationRanges, write_activation_ranges
from in_tfm.hosted_report import (
    BODY_NAME,
    current_lazy_script,
    current_stylesheet,
    hosted_index,
)
from in_tfm.html_report import cluster_section

OLD_SCRIPT = "<script>\n// Cluster bodies ship inside <template>, an older copy\nfunction stampCluster() {}\n</script>"
OLD_STYLE = "<style>\n.old-rule {}\n</style>"


def section(cluster_id: int, body: str) -> str:
    return (
        f'<section class="cluster" id="cluster-{cluster_id}" style="--accent:#fff">\n'
        f'<header class="cluster-head">Cluster {cluster_id}</header>\n'
        f'<div class="cluster-body pending"><template class="lazy-body">\n{body}\n</template></div>\n'
        "</section>"
    )


def page() -> str:
    return "\n".join(
        [
            f"<html><head>{OLD_STYLE}</head><body>",
            section(3, "<p>three</p>"),
            section(-1, "<p>noise</p>"),
            OLD_SCRIPT,
            "</body></html>",
        ]
    )


def test_each_body_moves_to_its_own_file_and_the_template_points_at_it(tmp_path):
    hosted = hosted_index(page(), tmp_path)
    assert "<p>three</p>" not in hosted and "<p>noise</p>" not in hosted
    assert (
        '<template class="lazy-body" data-src="cluster_3/body.html"></template>'
        in hosted
    )
    assert (tmp_path / "cluster_3" / BODY_NAME).read_text().strip() == "<p>three</p>"
    assert (tmp_path / "cluster_-1" / BODY_NAME).read_text().strip() == "<p>noise</p>"
    assert '<header class="cluster-head">Cluster 3</header>' in hosted


def test_the_old_script_and_stylesheet_are_replaced_by_the_current_ones(tmp_path):
    hosted = hosted_index(page(), tmp_path)
    assert "an older copy" not in hosted and ".old-rule" not in hosted
    assert current_lazy_script() in hosted and current_stylesheet() in hosted
    assert "dataset.src" in current_lazy_script()
    assert ".cluster:not(.open)" in current_stylesheet()


def test_sections_get_their_cluster_median_from_the_activation_ranges(tmp_path):
    write_activation_ranges(
        tmp_path, ActivationRanges(clusters={3: [5.0, 6.0, 9.0]}, noise=[])
    )
    hosted = hosted_index(page(), tmp_path)
    assert 'id="cluster-3" style="--accent:#fff" data-median="6.0000">' in hosted
    assert 'id="cluster--1" style="--accent:#fff">' in hosted


def test_a_new_section_carries_its_median_when_it_has_one():
    with_median = cluster_section(7, 40, 3, "", median_activation=-4.25)
    assert 'data-median="-4.2500"' in with_median
    assert "data-median" not in cluster_section(7, 40, 3, "")
