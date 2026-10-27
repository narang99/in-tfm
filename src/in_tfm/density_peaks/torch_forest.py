"""The forest with torch in float32, for a GPU. It computes what `vectorized.py` does, up to float32 rounding.

- On 20,000 hits the numpy forest took about 3 hours on Colab's CPU, almost all of it in distances.
  - It builds every pairwise distance three times, once each for the kth distance, the neighbours and the parents.
- Here the kth distance and the neighbours share one pass over the distances, and the parents take a second.
- Each pass compares every pair in float32, which a T4 does many times faster than float64.
- Float32 picks each point's kth neighbour and its parent. Float64 then recomputes only those n distances.
  - Exact ties stay exact that way. Scaled copies of one pattern tie exactly when rows are not normalised, and
    float32 alone broke those ties at random, which reordered ranks and so parents.
  - Near ties that are not exact can still swap. A float32-only forest on mixed4e:55 (20,000 hits, 8 trees) swapped
    107 ranks but changed no parent, root or tree, and no hit's most probable tree.
- Copies are merged on the CPU first, with `merge_copies`, exactly as in `vectorized.py`.
"""

import importlib.util
from collections.abc import Iterator

import numpy as np
import torch
from jaxtyping import Float, Int

from .forest import DensityPeakForest, PeakDistance, comparable_rows, merge_copies
from .vectorized import density_ranks

BLOCK_ELEMENTS = 2**27
"""The cap on a block of rows against every point. Tests pass a small value to force several blocks."""

HAS_TRITON = importlib.util.find_spec("triton") is not None


class TorchRows:
    """Distinct rows on the device, in float32 for every pair and float64 for the few pairs that set values."""

    def __init__(self, rows: Float[np.ndarray, "n compared_dim"], distance: PeakDistance, device: str):
        rows_t = torch.from_numpy(np.ascontiguousarray(rows, dtype=np.float64)).to(device)
        self.distance = distance
        self.exact_sums = rows_t.sum(dim=1) if distance == "weighted_jaccard" else None
        self.exact = signed_rows(rows_t) if distance == "weighted_jaccard" else rows_t
        self.sums = self.exact_sums.float() if self.exact_sums is not None else None
        self.rows = self.exact.float()

    def __len__(self) -> int:
        return len(self.rows)

    def block(self, start: int, stop: int) -> Float[torch.Tensor, "block n"]:
        sums_a = self.sums[start:stop] if self.sums is not None else None
        return float32_distances(self.rows[start:stop], self.rows, sums_a, self.sums, self.distance)

    def exact_pairs(self, a: Int[torch.Tensor, "m"], b: Int[torch.Tensor, "m"]) -> Float[torch.Tensor, "m"]:
        """The float64 distance between a[i] and b[i], for each i."""
        diff = self.exact[a] - self.exact[b]
        if self.distance == "euclidean":
            return diff.square().sum(dim=1).sqrt()
        both = self.exact_sums[a] + self.exact_sums[b]
        return weighted_jaccard_from_l1(diff.abs().sum(dim=1)[:, None], both[:, None])[:, 0]

    def blocks(self, block_elements: int) -> Iterator[tuple[int, Float[torch.Tensor, "block n"]]]:
        size = max(1, block_elements // len(self))
        for start in range(0, len(self), size):
            yield start, self.block(start, min(start + size, len(self)))


def signed_rows(parts: Float[torch.Tensor, "n two_dim"]) -> Float[torch.Tensor, "n dim"]:
    """Each dim's positive part minus its negative part, half as wide as the parts.

    - A dim has at most one nonzero part, so |a+ - b+| + |a- - b-| is |a - b|, with a and b the signed values.
      - Check the cases: both parts on one side, or one part on each side, give the same sum either way.
    - The L1 distance between parts so equals the L1 distance between signed rows, at half the cost."""
    half = parts.shape[1] // 2
    return parts[:, :half] - parts[:, half:]


def weighted_jaccard_from_l1(
    l1: Float[torch.Tensor, "a b"], both_sums: Float[torch.Tensor, "a b"],
) -> Float[torch.Tensor, "a b"]:
    """The same identity as `weighted_jaccard_block`: sum(min) and sum(max) both follow from the L1 distance."""
    overlap = (both_sums - l1) / 2
    union = (both_sums + l1) / 2
    return 1.0 - overlap / torch.where(union > 0, union, torch.ones_like(union))


def float32_distances(
    a: Float[torch.Tensor, "a dim"], b: Float[torch.Tensor, "b dim"],
    sums_a: Float[torch.Tensor, "a"] | None, sums_b: Float[torch.Tensor, "b"] | None, distance: PeakDistance,
) -> Float[torch.Tensor, "a b"]:
    if a.is_cuda and HAS_TRITON:
        from .triton_distance import triton_distances

        return triton_distances(a, b, sums_a, sums_b, distance)
    if distance == "euclidean":
        return torch.cdist(a, b, p=2.0, compute_mode="donot_use_mm_for_euclid_dist")
    return weighted_jaccard_from_l1(torch.cdist(a, b, p=1.0), sums_a[:, None] + sums_b[None, :])


def by_index(
    others: Int[torch.Tensor, "block c"], distance: Float[torch.Tensor, "block c"],
) -> tuple[Int[torch.Tensor, "block c"], Float[torch.Tensor, "block c"]]:
    """Columns in index order, so a stable sort after it breaks ties to the lower index."""
    order = torch.argsort(others, dim=1)
    return others.gather(1, order), distance.gather(1, order)


def kth_neighbour(
    nearest: Int[torch.Tensor, "block c"], own: Int[torch.Tensor, "block"], counts: Int[torch.Tensor, "n"], k: int,
) -> Int[torch.Tensor, "block"]:
    """`vectorized.kth_distances` for one block, as the point whose distance is the kth distance.

    `nearest` is closest first, with the point itself first."""
    is_self = nearest == own[:, None]
    hits = torch.where(is_self, counts[own, None] - 1, counts[nearest])
    reached = (torch.cumsum(hits, dim=1) < k).sum(dim=1).clamp(max=nearest.shape[1] - 1)
    return nearest[torch.arange(len(own), device=own.device), reached]


def neighbours_from_nearest(
    distance: Float[torch.Tensor, "block c"], nearest: Int[torch.Tensor, "block c"], own: Int[torch.Tensor, "block"],
    n: int,
) -> tuple[Int[torch.Tensor, "block n_neighbours"], Float[torch.Tensor, "block n_neighbours"]]:
    """Closest first, ties to the lower index, as in `vectorized.nearest_neighbours`."""
    nearest, distance = by_index(nearest, torch.where(nearest == own[:, None], torch.inf, distance))
    closest_first = torch.argsort(distance, dim=1, stable=True)[:, :n]
    return nearest.gather(1, closest_first), distance.gather(1, closest_first)


def kth_and_neighbours(
    rows: TorchRows, counts: Int[np.ndarray, "n"], k: int, block_elements: int,
) -> tuple[Float[np.ndarray, "n"], Int[np.ndarray, "n n_neighbours"], Float[np.ndarray, "n n_neighbours"]]:
    """One pass: the k + 1 nearest of each point, itself included, give both its kth distance and its neighbours."""
    device = rows.rows.device
    counts_t = torch.from_numpy(counts).to(device)
    n = min(k, len(rows) - 1)
    n_nearest = min(k + 1, len(rows))
    kth = torch.empty(len(rows), dtype=torch.float64, device=device)
    neighbours = torch.empty((len(rows), max(n, 0)), dtype=torch.int64, device=device)
    neighbour_distance = torch.empty((len(rows), max(n, 0)), dtype=torch.float64, device=device)
    for start, block in rows.blocks(block_elements):
        own = torch.arange(start, start + len(block), device=device)
        # the point itself is nearest, ahead of any other point tied with it at distance 0
        block[torch.arange(len(block), device=device), own] = -1.0
        distance, nearest = torch.topk(block, n_nearest, dim=1, largest=False, sorted=True)
        distance = distance.double()
        # the point itself as its kth neighbour means its own copies reach k, so its kth distance is 0
        kth[own] = rows.exact_pairs(own, kth_neighbour(nearest, own, counts_t, k))
        if n > 0:
            neighbours[own], neighbour_distance[own] = neighbours_from_nearest(distance, nearest, own, n)
    return kth.cpu().numpy(), neighbours.cpu().numpy(), neighbour_distance.cpu().numpy()


def nearest_denser(
    rows: TorchRows, ranks: Int[np.ndarray, "n"], block_elements: int,
) -> tuple[Int[np.ndarray, "n"], Float[np.ndarray, "n"]]:
    """`vectorized.nearest_denser`. `argmin` returns the first minimum, so ties go to the lower index.

    A point with nothing denser takes its largest distance, which only the densest point needs."""
    device = rows.rows.device
    ranks_t = torch.from_numpy(ranks).to(device)
    parent = torch.full((len(rows),), -1, dtype=torch.int64, device=device)
    delta = torch.zeros(len(rows), dtype=torch.float64, device=device)
    for start, block in rows.blocks(block_elements):
        own = torch.arange(start, start + len(block), device=device)
        denser = ranks_t[None, :] > ranks_t[own, None]
        has_denser = denser.any(dim=1)
        densest = own[~has_denser]
        delta[densest] = rows.exact_pairs(densest, block[~has_denser].argmax(dim=1))
        nearest = torch.where(denser, block, torch.inf)[has_denser].argmin(dim=1)
        parent[own[has_denser]] = nearest
        delta[own[has_denser]] = rows.exact_pairs(own[has_denser], nearest)
    return parent.cpu().numpy(), delta.cpu().numpy()


def density_peak_forest(
    points: Float[np.ndarray, "n_hits dim"],
    k: int = 5,
    distance: PeakDistance = "weighted_jaccard",
    normalise: bool = True,
    device: str = "cuda",
    block_elements: int = BLOCK_ELEMENTS,
) -> DensityPeakForest:
    """`vectorized.density_peak_forest` on `device`."""
    distinct, distinct_of_hit, counts = merge_copies(comparable_rows(points.astype(np.float64), distance, normalise))
    rows = TorchRows(distinct, distance, device)
    kth_distance, neighbours, neighbour_distance = kth_and_neighbours(rows, counts, k, block_elements)
    ranks = density_ranks(kth_distance, counts)
    parent, delta = nearest_denser(rows, ranks, block_elements)
    return DensityPeakForest(
        distinct_of_hit=distinct_of_hit, counts=counts, kth_distance=kth_distance, density_rank=ranks,
        parent=parent, delta=delta, neighbours=neighbours, neighbour_distance=neighbour_distance,
    )


def pairwise_distances(
    points: Float[np.ndarray, "n_hits dim"], distance: PeakDistance, normalise: bool, device: str,
    block_elements: int = BLOCK_ELEMENTS,
) -> Float[np.ndarray, "n_hits n_hits"]:
    """Every pair of hits, as `comparable_rows` compares them, in float32 with a zero diagonal."""
    rows = TorchRows(comparable_rows(points.astype(np.float64), distance, normalise), distance, device)
    result = np.empty((len(rows), len(rows)), dtype=np.float32)
    for start, block in rows.blocks(block_elements):
        result[start:start + len(block)] = block.cpu().numpy()
    np.fill_diagonal(result, 0.0)
    return result
