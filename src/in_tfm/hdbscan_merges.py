"""HDBSCAN's condensed tree as merges: which groups of hits join, and at what distance.

- HDBSCAN builds one tree, then picks clusters from it.
  - leaf keeps the tree's leaves.
  - eom keeps the most stable nodes, which can hold several leaves.
  - Both come from one fit here, so a leaf's place under an eom cluster is exact, not matched afterwards.
- Read from the top, the tree splits a group into two as the distance shrinks. Read from the leaves, it merges.
  - `merge_distance` is where a node's two children join.
- A hit leaves the tree at one node: below that distance it is noise to every child.
- The `hdbscan` package is used because sklearn keeps the condensed tree private.
  - Its `min_samples` counts neighbours as `clustering._cluster_precomputed` does, so the settings mean the same.
"""

import hdbscan
import numpy as np
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict


class MergeNode(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    node: int
    """HDBSCAN's own id. The root is the number of hits."""
    parent: int | None
    children: list[int]
    birth_distance: float
    """Where this node splits off its parent. Infinite for the root."""
    merge_distance: float | None
    """Where its two children join. None at a leaf."""
    members: Int[np.ndarray, "n_members"]
    """Every hit in this node's subtree."""
    dropped: Int[np.ndarray, "n_dropped"]
    """Hits that leave the tree at this node, so no child holds them."""
    persistence: Float[np.ndarray, "n_dropped"]
    """1 / the distance at which each dropped hit leaves. Higher stays longer."""
    is_leaf: bool
    is_eom: bool


def fit_tree(
    distances: Float[np.ndarray, "n n"], min_cluster_size: int, min_samples: int | None
) -> hdbscan.HDBSCAN:
    return hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="eom",
    ).fit(distances)


def merge_nodes(clusterer: hdbscan.HDBSCAN) -> dict[int, MergeNode]:
    raw = clusterer.condensed_tree_._raw_tree
    n_hits = len(clusterer.labels_)
    eom = {int(n) for n in clusterer.condensed_tree_._select_clusters()}
    is_point = raw["child"] < n_hits
    splits, drops = raw[~is_point], raw[is_point]

    children: dict[int, list[int]] = {}
    birth: dict[int, float] = {n_hits: float("inf")}
    parent: dict[int, int] = {}
    for row in splits:
        children.setdefault(int(row["parent"]), []).append(int(row["child"]))
        birth[int(row["child"])] = 1 / row["lambda_val"]
        parent[int(row["child"])] = int(row["parent"])

    nodes = {}
    for node in sorted(birth, reverse=True):
        kids = children.get(node, [])
        mine = drops[drops["parent"] == node]
        members = np.concatenate(
            [mine["child"]] + [nodes[k].members for k in kids]
        ).astype(int)
        nodes[node] = MergeNode(
            node=node,
            parent=parent.get(node),
            children=kids,
            birth_distance=birth[node],
            merge_distance=birth[kids[0]] if kids else None,
            members=np.sort(members),
            dropped=mine["child"].astype(int),
            persistence=mine["lambda_val"].astype(float),
            is_leaf=not kids,
            is_eom=node in eom,
        )
    return nodes


def root_of(nodes: dict[int, MergeNode]) -> int:
    return next(n for n, node in nodes.items() if node.parent is None)


def steadiest_hits(
    nodes: dict[int, MergeNode],
    node: int,
    k: int,
    copy_of: Int[np.ndarray, "n_hits"] | None = None,
) -> list[int]:
    """The hits that stay in this subtree the longest: its most central members.

    Identical hits never leave the tree, so they would always come first. `copy_of` keeps one hit per copy group."""
    stack, dropped, persistence = [node], [], []
    while stack:
        current = nodes[stack.pop()]
        dropped.append(current.dropped)
        persistence.append(current.persistence)
        stack.extend(current.children)
    hits = np.concatenate(dropped)
    ordered = hits[np.argsort(-np.concatenate(persistence), kind="stable")]
    if copy_of is not None:
        _, first = np.unique(copy_of[ordered], return_index=True)
        ordered = ordered[np.sort(first)]
    return ordered[:k].tolist()


def shown_hits(
    nodes: dict[int, MergeNode],
    node: int,
    per_card: int,
    copy_of: Int[np.ndarray, "n_hits"] | None = None,
) -> list[int]:
    """A leaf shows its steadiest hits. A merge shows the steadiest of each child, so both sides of the join are seen."""
    current = nodes[node]
    if current.is_leaf:
        return steadiest_hits(nodes, node, per_card, copy_of)
    per_child = max(1, per_card // len(current.children))
    return [
        h
        for child in current.children
        for h in steadiest_hits(nodes, child, per_child, copy_of)
    ]


def nearest_kept_ancestor(
    nodes: dict[int, MergeNode], node: int, kept: set[int]
) -> int | None:
    parent = nodes[node].parent
    while parent is not None and parent not in kept:
        parent = nodes[parent].parent
    return parent


def leaf_and_eom_only(nodes: dict[int, MergeNode]) -> dict[int, MergeNode]:
    """Only the clusters leaf or eom keeps. A leaf hangs straight off its eom cluster, with the merges between dropped.

    - eom picks one node on every path from a leaf to the root, so every leaf lands under exactly one eom cluster.
    - Each eom cluster becomes the top of its own tree.
    - Members and heights stay as in the full tree, so cards should still be drawn from the full tree."""
    kept = {n for n, node in nodes.items() if node.is_leaf or node.is_eom}
    parent = {n: nearest_kept_ancestor(nodes, n, kept) for n in kept}
    return {
        n: nodes[n].model_copy(
            update={
                "parent": parent[n],
                "children": [c for c in kept if parent[c] == n],
            }
        )
        for n in kept
    }
