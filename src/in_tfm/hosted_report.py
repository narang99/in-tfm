"""A finished report, rewritten so a web host serves each cluster body as its own file.

- A report's `index.html` embeds every cluster body in a `<template>`, so opening it locally needs no server.
  - On a host the whole page downloads before anything shows, 127 MB for Qwen3-32B L55/N11703.
- The hosted copy keeps the cluster headers and moves each body to `cluster_{id}/body.html`.
  - Its template gets a `data-src`, and `lazy_clusters.js` fetches the body when the cluster is opened.
- Older reports are brought up to date on the way.
  - Their stylesheet and `lazy_clusters.js` are replaced with the current ones.
  - Each section gets the `data-median` the page uses to pick which clusters start open.
- Browsers refuse `fetch` on `file://` pages, so the hosted copy only works when served.
"""

import re
from pathlib import Path

from .activation_ranges import cluster_medians, load_activation_ranges
from .html_report import render_template

LAZY_BODY = re.compile(
    r'(<section class="cluster" id="cluster-(?P<id>-?\d+)".*?<template class="lazy-body")>'
    r"(?P<body>.*?)</template>",
    re.DOTALL,
)
SECTION_TAG = re.compile(
    r'<section class="cluster" id="cluster-(?P<id>-?\d+)"(?P<rest>[^>]*)>'
)
LAZY_SCRIPT = re.compile(
    r"<script>\s*// Cluster bodies ship inside <template>.*?</script>", re.DOTALL
)
STYLESHEET = re.compile(r"<style>.*?</style>", re.DOTALL)
BODY_NAME = "body.html"


def body_path(cluster_id: str) -> str:
    return f"cluster_{cluster_id}/{BODY_NAME}"


def current_lazy_script() -> str:
    return f"<script>\n{render_template('lazy_clusters.js')}</script>"


def current_stylesheet() -> str:
    return f"<style>\n{render_template('report.css')}\n</style>"


def without_bodies(index_html: str, report_dir: Path) -> str:
    """Writes each cluster body next to the report and returns the index without them."""

    def move_body(match: re.Match[str]) -> str:
        path = report_dir / body_path(match["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(match["body"])
        return f'{match[1]} data-src="{body_path(match["id"])}"></template>'

    return LAZY_BODY.sub(move_body, index_html)


def with_medians(index_html: str, medians: dict[int, float]) -> str:
    def add_median(match: re.Match[str]) -> str:
        median = medians.get(int(match["id"]))
        if median is None or "data-median" in match["rest"]:
            return match[0]
        return match[0][:-1] + f' data-median="{median:.4f}">'

    return SECTION_TAG.sub(add_median, index_html)


def replaced_once(
    pattern: re.Pattern[str], replacement: str, html: str, what: str
) -> str:
    if pattern.search(html) is None:
        raise ValueError(f"the page has no {what} to replace")
    return pattern.sub(lambda _: replacement, html, count=1)


def hosted_index(index_html: str, report_dir: Path) -> str:
    ranges = load_activation_ranges(report_dir)
    html = without_bodies(index_html, report_dir)
    html = with_medians(html, cluster_medians(ranges) if ranges is not None else {})
    html = replaced_once(STYLESHEET, current_stylesheet(), html, "stylesheet")
    return replaced_once(
        LAZY_SCRIPT, current_lazy_script(), html, "lazy_clusters.js script"
    )


def write_hosted_index(report_dir: Path, name: str = "index.html") -> Path:
    """Rewrites `report_dir/name` in place. Run it on a copy that is only for hosting."""
    index = report_dir / name
    index.write_text(hosted_index(index.read_text(), report_dir))
    return index
