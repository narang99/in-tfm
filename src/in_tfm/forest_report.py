"""A page for a density-peak forest: where the roots are, and how each tree grows.

It sits next to the neuron report as `density_peaks.html`, and the report links to it.
Plots are matplotlib, since they are real charts. Patches are PIL tiles, like the rest of the report.
"""

from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float, Int
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict

from .density_peaks import DensityPeakForest, ForestTrees, PeakDistance, log_density, root_mask
from .html_report import page, render_template
from .presenters import HadamardShape
from .viz import render_hadamard_tiles

FOREST_PAGE = "density_peaks.html"
FOREST_DIR = "density_peaks"
DEPTH_BINS: tuple[tuple[int, int | None], ...] = ((0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, None))
TREE_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#eda100", "#7b5cd6", "#3fb8c4", "#a0522d")
OTHER_TREES_COLOUR = "#9a9aa2"
"""Trees past the 8th share one grey, rather than cycling colours."""


class RootRule(BaseModel):
    k: int
    distance: PeakDistance


class GrowthStage(BaseModel):
    label: str
    n_hits: int
    mean_activation: float
    hadamard_src: str
    input_src: str


class TreeRow(BaseModel):
    tree_id: int
    colour: str
    n_hits: int
    n_distinct: int
    root_copies: int
    root_activation: float
    median_activation: float
    max_depth: int
    stages: list[GrowthStage]


class HitArrays(BaseModel):
    """Per hit, in the order the forest was built from."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hadamards: Float[np.ndarray, "n_hits dim"]
    inputs: Float[np.ndarray, "n_hits dim"]
    activations: Float[np.ndarray, "n_hits"]


def tree_colour(tree_id: int) -> str:
    return TREE_COLOURS[tree_id] if tree_id < len(TREE_COLOURS) else OTHER_TREES_COLOUR


def save_decision_graph(forest: DensityPeakForest, trees: ForestTrees, path: Path) -> None:
    """Real centres sit top right: dense, and far from anything denser."""
    x, is_root = log_density(forest.kth_distance), root_mask(forest, trees.cuts)
    fig = Figure(figsize=(7, 4.6), layout="constrained")
    ax = fig.subplots()
    ax.scatter(x[~is_root], forest.delta[~is_root], s=6, color="#b0b0b5", alpha=0.6, linewidth=0, label="distinct hit")
    for tree_id, root in enumerate(trees.roots):
        ax.scatter(x[root], forest.delta[root], s=40 + 3 * np.sqrt(forest.counts[root]), color=tree_colour(tree_id),
                   edgecolor="black", linewidth=0.6, zorder=3)
        ax.annotate(f"T{tree_id}", (x[root], forest.delta[root]), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.axhline(trees.cuts.delta, color="#eb6834", linestyle="--", linewidth=1, label=f"delta elbow = {trees.cuts.delta:.3f}")
    ax.axvline(trees.cuts.rho, color="#6b6b70", linestyle=":", linewidth=1, label=f"density elbow = {trees.cuts.rho:.2f}")
    ax.set_xlabel("density: -log10(d_k + offset), right is denser")
    ax.set_ylabel("delta: distance to the nearest denser hit")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=110)


def depth_label(low: int, high: int | None) -> str:
    if high is None:
        return f"depth {low}+"
    return f"depth {low}" if low == high else f"depth {low}–{high}"


def within_depth(depth: Int[np.ndarray, "n_hits"], low: int, high: int | None) -> Bool[np.ndarray, "n_hits"]:
    return (depth >= low) & (depth <= (high if high is not None else np.iinfo(int).max))


def save_tiles(vectors: list[np.ndarray], shape: HadamardShape, out_dir: Path, prefix: str) -> list[str]:
    """One colour scale per call, so a tree's stages compare with each other."""
    cell = max(4, 72 // max(shape))
    tiles = render_hadamard_tiles([v.reshape(shape) for v in vectors], size=(shape[1] * cell, shape[0] * cell))
    names = []
    for i, tile in enumerate(tiles):
        tile.save(out_dir / f"{prefix}_{i}.png")
        names.append(f"{FOREST_DIR}/{prefix}_{i}.png")
    return names


def tree_row(
    tree_id: int, forest: DensityPeakForest, trees: ForestTrees, hits: HitArrays, shape: HadamardShape, out_dir: Path
) -> TreeRow:
    root = trees.roots[tree_id]
    in_tree = trees.tree[forest.distinct_of_hit] == tree_id
    hit_depth = trees.depth[forest.distinct_of_hit]
    masks = [in_tree & within_depth(hit_depth, low, high) for low, high in DEPTH_BINS]
    bins = [b for b, mask in zip(DEPTH_BINS, masks) if mask.any()]
    chosen = [mask for mask in masks if mask.any()]
    hadamard_srcs = save_tiles([hits.hadamards[c].mean(0) for c in chosen], shape, out_dir, f"tree{tree_id}_hadamard")
    input_srcs = save_tiles([hits.inputs[c].mean(0) for c in chosen], shape, out_dir, f"tree{tree_id}_input")
    root_hits = forest.distinct_of_hit == root
    return TreeRow(
        tree_id=tree_id,
        colour=tree_colour(tree_id),
        n_hits=int(in_tree.sum()),
        n_distinct=int((trees.tree == tree_id).sum()),
        root_copies=int(forest.counts[root]),
        root_activation=float(hits.activations[root_hits].mean()),
        median_activation=float(np.median(hits.activations[in_tree])),
        max_depth=int(hit_depth[in_tree].max()),
        stages=[
            GrowthStage(label=depth_label(*b), n_hits=int(c.sum()), mean_activation=float(hits.activations[c].mean()),
                        hadamard_src=h, input_src=i)
            for b, c, h, i in zip(bins, chosen, hadamard_srcs, input_srcs)
        ],
    )


def write_forest_page(
    report_dir: Path,
    forest: DensityPeakForest,
    trees: ForestTrees,
    hits: HitArrays,
    shape: HadamardShape,
    rule: RootRule,
    title: str,
    tree_page: str | None = None,
) -> Path:
    out_dir = report_dir / FOREST_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    save_decision_graph(forest, trees, out_dir / "decision_graph.png")
    body = render_template(
        "density_peaks.html",
        title=title,
        n_hits=len(forest.distinct_of_hit),
        n_distinct=len(forest.counts),
        most_copies=int(forest.counts.max()),
        rule=rule,
        cuts=trees.cuts,
        n_trees=len(trees.roots),
        tree_page=tree_page,
        decision_graph_src=f"{FOREST_DIR}/decision_graph.png",
        trees=[tree_row(t, forest, trees, hits, shape, out_dir) for t in range(len(trees.roots))],
    )
    path = report_dir / FOREST_PAGE
    path.write_text(page(title, body))
    return path
