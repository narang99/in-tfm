"""The "Borders between clusters" section: hits grouped by which clusters together carry their probability.

- A hit's set is the smallest group of its most probable clusters that reaches `SET_SHARE` of its probability.
  - Entropy alone mixes up a hit split evenly between two clusters with one spread thinly over many.
- Sets of one cluster are confident hits, so only sets of two or more are listed.
"""

from collections import Counter

import numpy as np
from markupsafe import escape

from ..html_report import render_template
from .data import ForestData
from .figures import tree_color, tree_name
from .sections import GuideEntry, PageSection, Pane

SET_SHARE = 0.85
MAX_SET_SIZE = 4
"""A hit needing more clusters than this is spread too thin to sit on a border."""
ROWS_SHOWN = 30


def cluster_set(row: np.ndarray) -> tuple[int, ...]:
    order = np.argsort(-row, kind="stable")
    reached = int(np.searchsorted(np.cumsum(row[order]), SET_SHARE)) + 1
    return tuple(sorted(int(t) for t in order[:reached]))


def border_counts(data: ForestData) -> Counter[tuple[int, ...]]:
    """Hits per set, counting each distinct point by its copies."""
    counts: Counter[tuple[int, ...]] = Counter()
    for point, row in enumerate(data.probabilities):
        clusters = cluster_set(row)
        if 1 < len(clusters) <= MAX_SET_SIZE:
            counts[clusters] += int(data.counts[point])
    return counts


def chip(tree: int) -> str:
    return f'<a class="tree-chip" href="#cluster-{tree}" style="--c:{tree_color(tree)}">{escape(tree_name(tree))}</a>'


def borders_section(data: ForestData | None, mixed_hits_page: str | None) -> PageSection | None:
    if data is None:
        return None
    counts = border_counts(data)
    if not counts:
        return None
    n_hits = len(data.distinct_of_hit)
    rows = [
        {"chips": "".join(chip(t) for t in clusters), "hits": hits, "share": hits / n_hits}
        for clusters, hits in counts.most_common(ROWS_SHOWN)
    ]
    sheet = f' The <a href="{mixed_hits_page}">sheet of split hits</a> shows them with their inputs.' if mixed_hits_page else ""
    return PageSection(
        key="borders",
        title="Borders between clusters",
        lede_html=(
            f"Hits whose probability needs 2 to {MAX_SET_SIZE} clusters to reach {SET_SHARE:.0%}, grouped by those "
            f"clusters. {sum(counts.values()):,} of {n_hits:,} hits sit on a border, over {len(counts):,} sets. "
            f"The largest {min(ROWS_SHOWN, len(counts))} are listed." + sheet
        ),
        panes=[Pane(name="Largest borders", body_html=render_template("borders_table.html", rows=rows))],
        guide=[GuideEntry(
            term="Border",
            text=(
                f"The smallest group of a hit's most probable clusters that holds {SET_SHARE:.0%} of its probability. "
                "A border of two clusters is a hit split between them."
            ),
        )],
    )
