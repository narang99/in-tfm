"""An interactive page of the forest: each tree drawn top-down from its root, opened a click at a time.

- The page carries every distinct point, so the browser can open any branch without a rebuild.
- Tiles are drawn in the browser from the raw values, so no image is written per node.
"""

from pathlib import Path

import numpy as np
from jaxtyping import Float, Int
from pydantic import BaseModel

from .density_peaks import DensityPeakForest, ForestTrees
from .forest_report import HitArrays
from .html_report import page, render_template
from .presenters import HadamardShape

TREE_PAGE = "forest_tree.html"
MAX_TREE_PAGE_VALUES = 4_000_000
"""The page holds each distinct point's Hadamard product and input. Past this many numbers it is not written."""
VALUE_DECIMALS = 4


class ForestTreeData(BaseModel):
    """Per distinct point, in the forest's order."""

    shape: HadamardShape
    roots: list[int]
    """Largest tree first, as in `ForestTrees`."""
    parent: list[int]
    """The parent within the point's tree, or -1 at a root."""
    depth: list[int]
    count: list[int]
    subtree_hits: list[int]
    """Hits at the point and everywhere below it."""
    activation: list[float]
    """The mean over the point's copies."""
    hadamard: list[list[float]]
    input: list[list[float]]
    """The mean over the point's copies. Copies share a Hadamard direction, but their inputs can differ in scale."""


def parent_in_tree(forest: DensityPeakForest, trees: ForestTrees) -> Int[np.ndarray, "n_distinct"]:
    """A root's own nearest denser point is in another tree, so its link is cut."""
    parent = forest.parent.copy()
    parent[trees.roots] = -1
    return parent


def subtree_hits(forest: DensityPeakForest, parent: Int[np.ndarray, "n_distinct"]) -> Int[np.ndarray, "n_distinct"]:
    """Sparsest first, so every child has its whole subtree summed before it passes it to its parent."""
    hits = forest.counts.copy()
    for point in np.argsort(forest.density_rank):
        if parent[point] >= 0:
            hits[parent[point]] += hits[point]
    return hits


def mean_per_distinct(
    values: Float[np.ndarray, "n_hits *rest"], forest: DensityPeakForest
) -> Float[np.ndarray, "n_distinct *rest"]:
    sums = np.zeros((len(forest.counts), *values.shape[1:]))
    np.add.at(sums, forest.distinct_of_hit, values)
    return sums / forest.counts.reshape(-1, *[1] * (values.ndim - 1))


def rounded(values: np.ndarray) -> list:
    return np.round(values.astype(np.float64), VALUE_DECIMALS).tolist()


def forest_tree_data(
    forest: DensityPeakForest, trees: ForestTrees, hits: HitArrays, shape: HadamardShape
) -> ForestTreeData:
    parent = parent_in_tree(forest, trees)
    return ForestTreeData(
        shape=shape,
        roots=trees.roots.tolist(),
        parent=parent.tolist(),
        depth=trees.depth.tolist(),
        count=forest.counts.tolist(),
        subtree_hits=subtree_hits(forest, parent).tolist(),
        activation=rounded(mean_per_distinct(hits.activations, forest)),
        hadamard=rounded(mean_per_distinct(hits.hadamards, forest)),
        input=rounded(mean_per_distinct(hits.inputs, forest)),
    )


def write_forest_tree_page(
    report_dir: Path,
    forest: DensityPeakForest,
    trees: ForestTrees,
    hits: HitArrays,
    shape: HadamardShape,
    title: str,
    max_values: int = MAX_TREE_PAGE_VALUES,
) -> Path | None:
    """Returns None, and writes nothing, when the page would hold more than `max_values` numbers."""
    n_values = 2 * len(forest.counts) * hits.hadamards.shape[1]
    if n_values > max_values:
        print(f"[forest tree] skipped: {n_values} values is over {max_values}")
        return None
    data = forest_tree_data(forest, trees, hits, shape)
    body = render_template("forest_tree.html", title=title, n_distinct=len(forest.counts), data=data.model_dump())
    path = report_dir / TREE_PAGE
    path.write_text(page(f"{title}: forest tree", body))
    return path
