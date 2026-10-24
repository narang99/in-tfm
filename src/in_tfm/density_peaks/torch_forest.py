"""The forest with torch, for a GPU. It computes what `vectorized.py` does, and tests check that they agree.

- On 20,000 hits the numpy forest took about 3 hours on Colab's CPU, almost all of it in distances.
  - It builds every pairwise distance three times, once each for the kth distance, the neighbours and the parents.
- Here the kth distance and the neighbours share one pass over the distances, and the parents take a second.
- Everything stays in float64, so the distances match numpy's up to summation order.
  - A float32 distance could reorder near ties, which would change ranks, parents and so the trees.
- Copies are merged on the CPU first, with `merge_copies`, exactly as in `vectorized.py`.
"""

from collections.abc import Iterator

import numpy as np
import torch
from jaxtyping import Float, Int

from .forest import DensityPeakForest, PeakDistance, comparable_rows, merge_copies
from .vectorized import density_ranks

BLOCK_ELEMENTS = 2**27
"""The cap on a block of rows against every point, 1 GB of float64. Tests pass a small value to force several blocks."""


class TorchRows:
    """Distinct rows on the device, with what every block of distances needs."""

    def __init__(self, rows: Float[np.ndarray, "n compared_dim"], distance: PeakDistance, device: str):
        rows_t = torch.from_numpy(np.ascontiguousarray(rows, dtype=np.float64)).to(device)
        self.distance = distance
        if distance == "weighted_jaccard":
            self.sums = rows_t.sum(dim=1)
            self.rows = signed_rows(rows_t)
        else:
            self.rows = rows_t

    def __len__(self) -> int:
        return len(self.rows)

    def block(self, start: int, stop: int) -> Float[torch.Tensor, "block n"]:
        if self.distance == "euclidean":
            return torch.cdist(self.rows[start:stop], self.rows, p=2.0, compute_mode="donot_use_mm_for_euclid_dist")
        return weighted_jaccard_from_l1(
            torch.cdist(self.rows[start:stop], self.rows, p=1.0), self.sums[start:stop], self.sums,
        )

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
    l1: Float[torch.Tensor, "a b"], sums_a: Float[torch.Tensor, "a"], sums_b: Float[torch.Tensor, "b"],
) -> Float[torch.Tensor, "a b"]:
    """The same identity as `weighted_jaccard_block`: sum(min) and sum(max) both follow from the L1 distance."""
    total = sums_a[:, None] + sums_b[None, :]
    overlap = (total - l1) / 2
    union = (total + l1) / 2
    return 1.0 - overlap / torch.where(union > 0, union, torch.ones_like(union))


def kth_from_block(
    block: Float[torch.Tensor, "block n"], own: Int[torch.Tensor, "block"], counts: Int[torch.Tensor, "n"], k: int,
) -> Float[torch.Tensor, "block"]:
    """`vectorized.kth_distances` for one block, which it overwrites at the points themselves."""
    rows = torch.arange(len(block), device=block.device)
    block[rows, own] = -1.0
    n_nearest = min(k + 1, block.shape[1])
    nearest_distance, nearest = torch.topk(block, n_nearest, dim=1, largest=False, sorted=True)
    is_self = nearest == own[:, None]
    hits = torch.where(is_self, counts[own, None] - 1, counts[nearest])
    reached = (torch.cumsum(hits, dim=1) < k).sum(dim=1).clamp(max=n_nearest - 1)
    return nearest_distance[rows, reached].clamp(min=0.0)


def neighbours_from_block(
    block: Float[torch.Tensor, "block n"], own: Int[torch.Tensor, "block"], n: int,
) -> tuple[Int[torch.Tensor, "block n_neighbours"], Float[torch.Tensor, "block n_neighbours"]]:
    """Closest first, ties to the lower index, as in `vectorized.nearest_neighbours`. Overwrites the block at `own`."""
    block[torch.arange(len(block), device=block.device), own] = torch.inf
    distance, nearest = torch.topk(block, n, dim=1, largest=False, sorted=False)
    by_index = torch.argsort(nearest, dim=1)
    nearest, distance = nearest.gather(1, by_index), distance.gather(1, by_index)
    by_distance = torch.argsort(distance, dim=1, stable=True)
    return nearest.gather(1, by_distance), distance.gather(1, by_distance)


def kth_and_neighbours(
    rows: TorchRows, counts: Int[np.ndarray, "n"], k: int, block_elements: int,
) -> tuple[Float[np.ndarray, "n"], Int[np.ndarray, "n n_neighbours"], Float[np.ndarray, "n n_neighbours"]]:
    device = rows.rows.device
    counts_t = torch.from_numpy(counts).to(device)
    n = min(k, len(rows) - 1)
    kth = torch.empty(len(rows), dtype=torch.float64, device=device)
    neighbours = torch.empty((len(rows), max(n, 0)), dtype=torch.int64, device=device)
    neighbour_distance = torch.empty((len(rows), max(n, 0)), dtype=torch.float64, device=device)
    for start, block in rows.blocks(block_elements):
        own = torch.arange(start, start + len(block), device=device)
        if n > 0:
            neighbours[own], neighbour_distance[own] = neighbours_from_block(block.clone(), own, n)
        kth[own] = kth_from_block(block, own, counts_t, k)
    return kth.cpu().numpy(), neighbours.cpu().numpy(), neighbour_distance.cpu().numpy()


def nearest_denser(
    rows: TorchRows, ranks: Int[np.ndarray, "n"], block_elements: int,
) -> tuple[Int[np.ndarray, "n"], Float[np.ndarray, "n"]]:
    """`vectorized.nearest_denser`: `argmin` returns the first of tied minima on both, so ties go to the lower index."""
    device = rows.rows.device
    ranks_t = torch.from_numpy(ranks).to(device)
    parent = torch.full((len(rows),), -1, dtype=torch.int64, device=device)
    delta = torch.zeros(len(rows), dtype=torch.float64, device=device)
    for start, block in rows.blocks(block_elements):
        own = torch.arange(start, start + len(block), device=device)
        denser = ranks_t[None, :] > ranks_t[own, None]
        candidates = torch.where(denser, block, torch.inf)
        nearest = candidates.argmin(dim=1)
        has_denser = denser.any(dim=1)
        parent[own[has_denser]] = nearest[has_denser]
        delta[own[has_denser]] = candidates[has_denser, nearest[has_denser]]
        delta[own[~has_denser]] = block[~has_denser].max(dim=1).values
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
    """Every pair of hits, as `comparable_rows` compares them, with a zero diagonal."""
    rows = TorchRows(comparable_rows(points.astype(np.float64), distance, normalise), distance, device)
    result = np.empty((len(rows), len(rows)))
    for start, block in rows.blocks(block_elements):
        result[start:start + len(block)] = block.cpu().numpy()
    np.fill_diagonal(result, 0.0)
    return result
