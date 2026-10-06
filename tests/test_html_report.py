from in_tfm.html_report import (
    ACCENTS,
    ClusterLink,
    accent_class,
    cluster_nav,
    cluster_section,
    details,
    page,
)


def test_page_escapes_title_but_not_body():
    html = page("<b>t</b>", "<p>body</p>")
    assert "<title>&lt;b&gt;t&lt;/b&gt;</title>" in html
    assert "<p>body</p>" in html


def test_page_defines_one_css_rule_per_accent():
    html = page("t", "")
    assert all(f".accent-{i} " in html for i in range(len(ACCENTS)))


def test_accent_class_wraps_around():
    assert accent_class(len(ACCENTS)) == accent_class(0)


def test_details_escapes_summary_and_opens_on_request():
    assert "<summary>a &lt; b</summary>" in details("a < b", "x")
    assert details("s", "x", start_open=True).startswith("<details open>")
    assert details("s", "x").startswith("<details>")


def test_cluster_section_links_nav_anchor():
    section = cluster_section(7, position=1, n_hits=40, n_unique_images=3, body="<i>b</i>")
    nav = cluster_nav([ClusterLink(cluster_id=7, n_hits=40, accent_class=accent_class(1))])
    assert 'id="cluster-7"' in section
    assert 'href="#cluster-7"' in nav
    assert "<i>b</i>" in section
