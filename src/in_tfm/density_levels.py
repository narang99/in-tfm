"""Groups of hits at falling density cutoffs, to see where the clusters' cores form and how they merge.

- Each level keeps the densest hits up to a share of all hits: 5%, 10%, and so on up to 100%.
  - Levels are hit shares, not density values. A few near-identical hits sit far above the rest on the density axis,
    so equal-width density levels would be almost all empty.
- A group is a connected part of the forest's nearest-neighbour graph, among the points kept at that level.
  - Points join densest first, so one sweep gives every level.
- When two groups meet, the merged group keeps the elder's identity, the one whose densest point is denser.
  - A group so keeps its number and colour from the level where it appears to the level where it joins an elder.
"""

import math

import numpy as np
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict

NOT_KEPT = -1
SHOWN_CLUSTERS_PER_GROUP = 3


class DensityGraph(BaseModel):
    """What the levels need from a saved forest. Per distinct point unless named per hit."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kth_distance: Float[np.ndarray, "n_distinct"]
    """Smaller is denser."""
    counts: Int[np.ndarray, "n_distinct"]
    neighbours: Int[np.ndarray, "n_distinct n_neighbours"]
    distinct_of_hit: Int[np.ndarray, "n_hits"]


class LevelGroup(BaseModel):
    number: int
    n_hits: int
    clusters: list[tuple[int, int]]
    """The clusters most of its hits have as their most probable, with how many hits each, largest first."""


class DensityLevel(BaseModel):
    share: float
    n_hits: int
    groups: list[LevelGroup]
    """Groups of at least `min_group_hits`, oldest first."""


class DensityLevels(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    levels: list[DensityLevel]
    hit_groups: Int[np.ndarray, "n_levels n_hits"]
    """Each hit's group number at each level, 0 for a small group, `NOT_KEPT` before the hit is kept."""
    hit_birth: Int[np.ndarray, "n_hits"]
    """The position of the hit's point in the densest-first order."""


def densest_first(kth_distance: Float[np.ndarray, "n_distinct"]) -> Int[np.ndarray, "n_distinct"]:
    return np.argsort(kth_distance, kind="stable")


def level_shares(step: float) -> list[float]:
    n_levels = math.ceil(1 / step - 1e-9)
    return [min(1.0, round(step * (i + 1), 6)) for i in range(n_levels)]


def points_kept_per_level(counts_densest_first: Int[np.ndarray, "n_distinct"], shares: list[float]) -> list[int]:
    """How many of the densest points hold each share of the hits. A point with copies counts once per copy."""
    cumulative = np.cumsum(counts_densest_first)
    wanted = [math.ceil(share * cumulative[-1] - 1e-9) for share in shares]
    return [int(np.searchsorted(cumulative, hits)) + 1 for hits in wanted]


class ElderUnion:
    """Union-find where a group's root is always its densest point, so a merge keeps the elder's identity."""

    def __init__(self, birth: Int[np.ndarray, "n"]):
        self.parent = np.arange(len(birth))
        self.birth = birth

    def find(self, point: int) -> int:
        root = point
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[point] != root:
            self.parent[point], point = root, self.parent[point]
        return root

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return
        elder, younger = (root_a, root_b) if self.birth[root_a] < self.birth[root_b] else (root_b, root_a)
        self.parent[younger] = elder


def elder_of_each_point(
    order: Int[np.ndarray, "n_distinct"], neighbours: Int[np.ndarray, "n_distinct n_neighbours"], kept_per_level: list[int],
) -> Int[np.ndarray, "n_levels n_distinct"]:
    """Each point's group at each level, named by the group's densest point, or `NOT_KEPT`.

    - A point joins with every kept neighbour, and with every kept point that has it as a neighbour.
      - The second comes for free: when the later of the two joins, it finds the earlier one kept."""
    birth = np.empty(len(order), dtype=int)
    birth[order] = np.arange(len(order))
    union = ElderUnion(birth)
    reverse = [[] for _ in range(len(order))]
    for point, others in enumerate(neighbours):
        for other in others[others >= 0]:
            reverse[other].append(point)
    elders = np.full((len(kept_per_level), len(order)), NOT_KEPT)
    level = 0
    for position, point in enumerate(order):
        for other in [*neighbours[point][neighbours[point] >= 0], *reverse[point]]:
            if birth[other] < position:
                union.union(point, int(other))
        while level < len(kept_per_level) and kept_per_level[level] == position + 1:
            kept = order[: position + 1]
            elders[level, kept] = [union.find(int(p)) for p in kept]
            level += 1
    return elders


def hits_per_elder(elders_of_hits: Int[np.ndarray, "n_hits"]) -> dict[int, int]:
    kept = elders_of_hits[elders_of_hits != NOT_KEPT]
    named, sizes = np.unique(kept, return_counts=True)
    return dict(zip(named.tolist(), sizes.tolist()))


def numbered_elders(hit_elders: Int[np.ndarray, "n_levels n_hits"], birth: Int[np.ndarray, "n_distinct"], min_hits: int) -> dict[int, int]:
    """Every group that reaches `min_hits` at some level, numbered from 1 by when it appears."""
    large = {e for row in hit_elders for e, n in hits_per_elder(row).items() if n >= min_hits}
    return {elder: number for number, elder in enumerate(sorted(large, key=lambda e: birth[e]), start=1)}


def group_numbers_at_level(
    elders_of_hits: Int[np.ndarray, "n_hits"], numbers: dict[int, int], min_hits: int,
) -> Int[np.ndarray, "n_hits"]:
    """The group number of each hit, 0 in a group under `min_hits`, `NOT_KEPT` if not kept."""
    result = np.full(len(elders_of_hits), NOT_KEPT)
    kept = elders_of_hits != NOT_KEPT
    named, member_of, sizes = np.unique(elders_of_hits[kept], return_inverse=True, return_counts=True)
    number_of_named = np.array([numbers[e] if n >= min_hits else 0 for e, n in zip(named.tolist(), sizes.tolist())], dtype=int)
    result[kept] = number_of_named[member_of] if len(named) else 0
    return result


def top_clusters(clusters_of_members: Int[np.ndarray, "m"]) -> list[tuple[int, int]]:
    ids, sizes = np.unique(clusters_of_members, return_counts=True)
    ranked = np.argsort(-sizes, kind="stable")[:SHOWN_CLUSTERS_PER_GROUP]
    return [(int(ids[i]), int(sizes[i])) for i in ranked]


def describe_level(
    share: float, hit_groups: Int[np.ndarray, "n_hits"], top_cluster: Int[np.ndarray, "n_hits"],
) -> DensityLevel:
    numbers = sorted(set(hit_groups[hit_groups > 0].tolist()))
    groups = [
        LevelGroup(number=n, n_hits=int((hit_groups == n).sum()), clusters=top_clusters(top_cluster[hit_groups == n]))
        for n in numbers
    ]
    return DensityLevel(share=share, n_hits=int((hit_groups != NOT_KEPT).sum()), groups=groups)


def density_levels(
    graph: DensityGraph, probabilities: Float[np.ndarray, "n_hits n_clusters"], share_step: float, min_group_hits: int,
) -> DensityLevels:
    order = densest_first(graph.kth_distance)
    birth = np.empty(len(order), dtype=int)
    birth[order] = np.arange(len(order))
    shares = level_shares(share_step)
    elders = elder_of_each_point(order, graph.neighbours, points_kept_per_level(graph.counts[order], shares))
    hit_elders = elders[:, graph.distinct_of_hit]
    numbers = numbered_elders(hit_elders, birth, min_group_hits)
    hit_groups = np.stack([group_numbers_at_level(row, numbers, min_group_hits) for row in hit_elders])
    top_cluster = probabilities.argmax(axis=1)
    levels = [describe_level(share, row, top_cluster) for share, row in zip(shares, hit_groups)]
    return DensityLevels(levels=levels, hit_groups=hit_groups, hit_birth=birth[graph.distinct_of_hit])
