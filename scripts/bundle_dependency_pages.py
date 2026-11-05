#!/usr/bin/env python
"""Self-contained copies of dependency pages, with each linked cluster's AttnLRP examples inside.

- The pages in S3 link out to the origin report and to 64 o_proj reports of ~166 MB each.
- This copies, into each page:
  - the origin cluster's own section, from the origin report
  - every linked o_proj cluster's section, trimmed to a few hits, with its Hadamard image inlined
- Chips and thumbnails then point inside the page, so the folder works anywhere.
- The o_proj reports are streamed one at a time, and only the needed sections are kept.

    python scripts/bundle_dependency_pages.py --s3-root s3://narang99-private/in-tfm/tiny-attn-only-1l \\
        --origin lm_head/263 --clusters '[519, 647]' --out-dir /tmp/bundle
"""

import base64
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

OPROJ = "model.layers.0.self_attn.o_proj"
CHIP_HREF = re.compile(
    rf'href="(?:\.\./)+{re.escape(OPROJ)}/(\d+)/index\.html#cluster-(\d+)"'
)
THUMB_SRC = re.compile(
    rf'src="(?:\.\./)+{re.escape(OPROJ)}/(\d+)/cluster_(\d+)/hadamard\.jpg"'
)
HIT = re.compile(r'<div class="hit">.*?</div>\n</div>\n', re.S)


class Args(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)

    s3_root: str
    origin: str
    clusters: list[int]
    out_dir: Path
    hits_per_cluster: int = 5


def s3_text(uri: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".html") as tmp:
        subprocess.run(["aws", "s3", "cp", "--quiet", uri, tmp.name], check=True)
        return Path(tmp.name).read_text()


def s3_data_uri(uri: str) -> str:
    data = subprocess.run(
        ["aws", "s3", "cp", "--quiet", uri, "-"], check=True, capture_output=True
    ).stdout
    return "data:image/jpeg;base64," + base64.b64encode(data).decode()


def cluster_section(report: str, cluster_id: int) -> str:
    start = report.index(f'<section class="cluster" id="cluster-{cluster_id}"')
    return report[start : report.index("</section>", start) + len("</section>")]


def trimmed(section: str, keep: int) -> str:
    hits = list(HIT.finditer(section))
    for hit in reversed(hits[keep:]):
        section = section[: hit.start()] + section[hit.end() :]
    return section.replace(
        f"{len(hits)} sampled hits in context",
        f"{min(keep, len(hits))} sampled hits in context",
    )


def linked_clusters(pages: list[str]) -> dict[int, set[int]]:
    by_neuron: dict[int, set[int]] = defaultdict(set)
    for page in pages:
        for neuron, cluster in CHIP_HREF.findall(page):
            by_neuron[int(neuron)].add(int(cluster))
    return by_neuron


def oproj_sections(
    s3_root: str, by_neuron: dict[int, set[int]], keep: int
) -> dict[tuple[int, int], str]:
    sections = {}
    for neuron, clusters in sorted(by_neuron.items()):
        report = s3_text(f"{s3_root}/{OPROJ}/{neuron}/index.html")
        for cluster in sorted(clusters):
            image = s3_data_uri(
                f"{s3_root}/{OPROJ}/{neuron}/cluster_{cluster}/hadamard.jpg"
            )
            section = trimmed(cluster_section(report, cluster), keep)
            section = section.replace(
                f'src="cluster_{cluster}/hadamard.jpg"', f'src="{image}"'
            )
            section = section.replace(
                f'id="cluster-{cluster}"', f'id="oproj-{neuron}-{cluster}"'
            ).replace(
                f'class="cluster-badge">Cluster {cluster}<',
                f'class="cluster-badge">o_proj:{neuron} cluster {cluster}<',
            )
            sections[(neuron, cluster)] = section
        print(f"[o_proj {neuron}] {len(clusters)} clusters", flush=True)
    return sections


def readable_combos(page: str) -> str:
    """The token chip had no colour, and the column names did not say which writer they were."""
    page = page.replace("<th>token</th>", "<th>embedding writer: token</th>")
    page = page.replace("<th>cluster</th>", "<th>o_proj writer: its cluster</th>")
    return page.replace(
        '<a class="cl-chip" style="" href="#token-',
        '<a class="cl-chip" style="--accent:#444;--accent-ink:#fff" href="#token-',
    )


COMBO_NOTE = (
    '<p class="view-note"><b>Reading the tables.</b> Every firing of a dim is one pair: the token '
    "(the embedding writer) and the o_proj cluster it matched (the o_proj writer). This cluster fires "
    "at a single token, so the token column is the same in every row and only the o_proj cluster varies. "
    "Click an o_proj chip to jump to that cluster's AttnLRP examples, at the bottom of the page.</p>"
)


def bundled_page(
    page: str, origin_section: str, sections: dict[tuple[int, int], str]
) -> str:
    used = sorted({(int(n), int(c)) for n, c in CHIP_HREF.findall(page)})
    thumbs = {
        key: re.search(r'<img src="(data:[^"]+)"', sections[key]).group(1)
        for key in used
    }
    page = CHIP_HREF.sub(lambda m: f'href="#oproj-{m.group(1)}-{m.group(2)}"', page)
    page = THUMB_SRC.sub(
        lambda m: f'src="{thumbs[(int(m.group(1)), int(m.group(2)))]}"', page
    )
    page = re.sub(r'href="\.\./\.\./index\.html#cluster-\d+"', 'href="#origin"', page)
    page = readable_combos(page)
    origin = (
        '<section class="page-section" id="origin"><h2>The cluster itself, with AttnLRP</h2>'
        f"{origin_section}</section>"
    )
    page = page.replace(
        '<section class="page-section" id="noise">',
        origin + '\n<section class="page-section" id="noise">',
        1,
    )
    page = page.replace(
        '<section class="page-section" id="frequent">',
        '<section class="page-section" id="frequent">' + COMBO_NOTE,
        1,
    )
    appendix = (
        '<section class="page-section" id="oproj"><h2>o_proj clusters, with AttnLRP</h2>'
        '<p class="lede">Every o_proj cluster the tables link to, from its own neuron report, '
        "with its first few sampled hits.</p>"
        + "\n".join(sections[key] for key in used)
        + "</section>"
    )
    return page.replace("</body>", appendix + "\n</body>", 1)


def index_page(clusters: list[int], origin: str) -> str:
    links = "\n".join(
        f'<li><a href="cluster_{c}.html">{origin} cluster {c}</a></li>'
        for c in clusters
    )
    return (
        '<!doctype html><meta charset="utf-8"><title>Dependency pages</title>'
        f"<h1>Dependency pages of {origin}</h1><ul>{links}</ul>"
    )


def main() -> None:
    args = Args()  # ty: ignore[missing-argument]
    root = f"{args.s3_root}/{args.origin}"
    pages = {
        c: s3_text(f"{root}/dependencies/cluster_{c}/index.html") for c in args.clusters
    }
    origin_report = s3_text(f"{root}/index.html")
    origin_sections = {
        c: cluster_section(origin_report, c).replace(
            f'src="cluster_{c}/hadamard.jpg"',
            f'src="{s3_data_uri(f"{root}/cluster_{c}/hadamard.jpg")}"',
        )
        for c in args.clusters
    }
    del origin_report
    sections = oproj_sections(
        args.s3_root, linked_clusters(list(pages.values())), args.hits_per_cluster
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for cluster, page in pages.items():
        out = args.out_dir / f"cluster_{cluster}.html"
        out.write_text(bundled_page(page, origin_sections[cluster], sections))
        print(f"[page] {out} {out.stat().st_size / 1e6:.1f} MB", flush=True)
    (args.out_dir / "index.html").write_text(index_page(args.clusters, args.origin))


if __name__ == "__main__":
    main()
