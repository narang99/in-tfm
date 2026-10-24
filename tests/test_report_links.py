from pathlib import Path

from in_tfm.report_links import check_links


def write(path: Path, html: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)


def test_links_to_existing_files_and_anchors_land(tmp_path: Path):
    write(tmp_path / "clusters" / "index.html", '<section id="cluster-3"></section>')
    write(tmp_path / "buckets" / "a.html", '<a href="../clusters/index.html#cluster-3">3</a><a href="#card-1">1</a><tr id="card-1">')
    result = check_links(tmp_path)
    assert (result.pages, result.checked, result.broken) == (2, 2, [])


def test_a_missing_anchor_or_file_is_broken_and_external_links_are_skipped(tmp_path: Path):
    write(tmp_path / "index.html", '<section id="cluster-1"></section>')
    write(
        tmp_path / "page.html",
        '<a href="index.html#cluster-2">2</a><img src="maps/x.png"><a href="https://example.com">x</a>',
    )
    result = check_links(tmp_path)
    assert sorted(b.reference for b in result.broken) == ["index.html#cluster-2", "maps/x.png"]
    assert result.checked == 2
