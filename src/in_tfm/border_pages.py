"""A page per border between clusters: a few of its hits, drawn by the model's presenter, with their probabilities.

- A border is a set of 2 or more clusters that together carry a hit's probability. See `report_page.borders_section`.
- Only the largest borders get a page, since each hit on a page is attributed when the report is written.
- One sample per page, ordered from mostly the set's first cluster to mostly its last.
  - The order and the cards come from the same sample, so every card's position in the order is real.
- Pages sit in `borders/`, each hit's files in a folder beside them, so presenter urls resolve from the page.
"""

import json
from collections.abc import Collection
from pathlib import Path

import numpy as np
from jaxtyping import Int
from pydantic import BaseModel

from .attribution import compute_attnlrp_relevance
from .html_report import page, render_template
from .neuron_report import AttrFn, NeuronClusterHits
from .presenters import ClusterArtifacts
from .report_page import ForestData
from .report_page.borders_section import BORDER_INDEX, BORDERS_DIR, border_key, hit_sets, largest_borders
from .report_page.figures import cluster_chip, tree_color, tree_name

PAGES = 12
MIN_HITS = 10
HITS_PER_PAGE = 6


class ProbabilityBar(BaseModel):
    name: str
    tree: int
    share: float
    color: str


class BorderHit(BaseModel):
    hit_html: str
    bars: list[ProbabilityBar]


def evenly_ranked(members: Int[np.ndarray, "n"], lead: np.ndarray, count: int) -> Int[np.ndarray, "k"]:
    """`count` members spread evenly over their ranking by `lead`, highest first."""
    ranked = members[np.argsort(-lead, kind="stable")]
    picks = np.unique(np.linspace(0, len(ranked) - 1, min(count, len(ranked))).round().astype(int))
    return ranked[picks]


def border_hits(
    hits: NeuronClusterHits, data: ForestData, members: Int[np.ndarray, "n"], clusters: tuple[int, ...],
    page_dir: Path, page_id: int, attr_fn: AttrFn,
) -> list[BorderHit]:
    probabilities = data.probabilities[data.distinct_of_hit]
    shown = []
    for row, hit in enumerate(evenly_ranked(members, probabilities[members, clusters[0]], HITS_PER_PAGE)):
        artifacts = ClusterArtifacts(cluster_dir=page_dir / f"border_{page_id}_hit_{row}")
        artifacts.cluster_dir.mkdir(exist_ok=True)
        hit_html = hits.presenter.render([hits.cluster_hit(int(hit), attr_fn)], artifacts, hits.hadamard_shape)
        bars = [
            ProbabilityBar(name=tree_name(t), tree=t, share=float(probabilities[hit, t]), color=tree_color(t))
            for t in clusters
        ]
        shown.append(BorderHit(hit_html=hit_html, bars=bars))
    return shown


def write_border_pages(
    hits: NeuronClusterHits, report_dir: Path, data: ForestData, shown: Collection[int],
    attr_fn: AttrFn = compute_attnlrp_relevance,
) -> list[Path]:
    """Writes the pages and `borders/borders.json`, which the report's borders table reads to link them."""
    page_dir = report_dir / BORDERS_DIR
    page_dir.mkdir(parents=True, exist_ok=True)
    sets = hit_sets(data)
    links: dict[str, str] = {}
    paths = []
    for page_id, (clusters, n_hits) in enumerate(largest_borders(data, PAGES, MIN_HITS)):
        members = np.flatnonzero([s == clusters for s in sets])
        on_page = border_hits(hits, data, members, clusters, page_dir, page_id, attr_fn)
        body = render_template(
            "border_page.html",
            title=f"Neuron {hits.neuron_idx}: the border of {', '.join(tree_name(t) for t in clusters)}",
            chips="".join(cluster_chip(t, shown, page="../index.html") for t in clusters),
            first_cluster=tree_name(clusters[0]), n_hits=n_hits, hits=on_page,
        )
        path = page_dir / f"border_{page_id}.html"
        path.write_text(page(f"Neuron {hits.neuron_idx}: border", body))
        links[border_key(clusters)] = f"{BORDERS_DIR}/{path.name}"
        paths.append(path)
    (page_dir / BORDER_INDEX).write_text(json.dumps(links))
    return paths
