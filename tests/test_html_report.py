from in_tfm.html_report import (
    ClusterLink,
    cluster_nav,
    cluster_section,
    details,
    page,
)


def test_page_escapes_title_but_not_body():
    html = page("<b>t</b>", "<p>body</p>")
    assert "<title>&lt;b&gt;t&lt;/b&gt;</title>" in html
    assert "<p>body</p>" in html


def test_a_cluster_card_and_its_nav_link_share_the_charts_colour():
    from in_tfm.cluster_colors import cluster_color
    from in_tfm.report_page.figures import tree_color

    section = cluster_section(7, n_hits=40, n_unique_images=3, body="")
    nav = cluster_nav([ClusterLink(cluster_id=7, n_hits=40)])
    assert tree_color(7) == cluster_color(7)
    assert f"--accent:{cluster_color(7)}" in section and f"--accent:{cluster_color(7)}" in nav


def test_light_colours_get_dark_text():
    from in_tfm.cluster_colors import DARK_INK, LIGHT_INK, ink_on

    assert ink_on("#F3C300") == DARK_INK and ink_on("#604E97") == LIGHT_INK


def test_details_escapes_summary_and_opens_on_request():
    assert "<summary>a &lt; b</summary>" in details("a < b", "x")
    assert details("s", "x", start_open=True).startswith("<details open>")
    assert details("s", "x").startswith("<details>")


def test_cluster_section_links_nav_anchor():
    section = cluster_section(7, n_hits=40, n_unique_images=3, body="<i>b</i>")
    nav = cluster_nav([ClusterLink(cluster_id=7, n_hits=40)])
    assert 'id="cluster-7"' in section
    assert 'href="#cluster-7"' in nav
    assert "<i>b</i>" in section
