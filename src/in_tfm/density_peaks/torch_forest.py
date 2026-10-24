"""The forest with torch, for a GPU. It computes what `vectorized.py` does, and tests check that they agree.

- On 20,000 hits the numpy forest took about 3 hours on Colab's CPU, almost all of it in distances.
  - It builds every pairwise distance three times, once each for the kth distance, the neighbours and the parents.
- Here the kth distance and the neighbours share one pass over the distances, and the parents take a second.
- Each pass compares every pair in float32, which a T4 does many times faster than float64.
  - Float32 alone could reorder near ties, which would change ranks, parents and so the trees.
  - So float32 only picks candidates, and float64 decides among them. See `float32_error_bound`.
  - The forest so matches `vectorized.py` up to float64 summation order.
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

FLOAT32_ROUNDING = 2.0**-24
HAS_TRITON = importlib.util.find_spec("triton") is not None


class TorchRows:
    """Distinct rows on the device, in float64 for exact distances and float32 for fast ones."""

    def __init__(self, rows: Float[np.ndarray, "n compared_dim"], distance: PeakDistance, device: str):
        rows_t = torch.from_numpy(np.ascontiguousarray(rows, dtype=np.float64)).to(device)
        self.distance = distance
        self.sums = rows_t.sum(dim=1) if distance == "weighted_jaccard" else None
        self.rows = signed_rows(rows_t) if distance == "weighted_jaccard" else rows_t
        self.rows32 = self.rows.float()
        self.sums32 = self.sums.float() if self.sums is not None else None
        self.error_bound = float32_error_bound(self.rows, distance)

    def __len__(self) -> int:
        return len(self.rows)

    def fast_block(self, start: int, stop: int) -> Float[torch.Tensor, "block n"]:
        sums_a = self.sums32[start:stop] if self.sums32 is not None else None
        return float32_distances(self.rows32[start:stop], self.rows32, sums_a, self.sums32, self.distance)

    def fast_blocks(self, block_elements: int) -> Iterator[tuple[int, Float[torch.Tensor, "block n"]]]:
        size = max(1, block_elements // len(self))
        for start in range(0, len(self), size):
            yield start, self.fast_block(start, min(start + size, len(self)))

    def exact_rows(self, own: Int[torch.Tensor, "block"]) -> Float[torch.Tensor, "block n"]:
        if self.distance == "euclidean":
            return torch.cdist(self.rows[own], self.rows, p=2.0, compute_mode="donot_use_mm_for_euclid_dist")
        both = self.sums[own, None] + self.sums[None, :]
        return weighted_jaccard_from_l1(torch.cdist(self.rows[own], self.rows, p=1.0), both)

    def exact_pairs(
        self, own: Int[torch.Tensor, "block"], others: Int[torch.Tensor, "block c"], block_elements: int,
    ) -> Float[torch.Tensor, "block c"]:
        """The float64 distance from each point in `own` to each of its own row of `others`."""
        result = torch.empty(others.shape, dtype=torch.float64, device=others.device)
        size = max(1, block_elements // (others.shape[1] * self.rows.shape[1]))
        for start in range(0, len(own), size):
            stop = start + size
            diff = self.rows[own[start:stop], None, :] - self.rows[others[start:stop]]
            if self.distance == "euclidean":
                result[start:stop] = diff.square().sum(dim=2).sqrt()
            else:
                both = self.sums[own[start:stop], None] + self.sums[others[start:stop]]
                result[start:stop] = weighted_jaccard_from_l1(diff.abs().sum(dim=2), both)
        return result


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


def float32_error_bound(rows: Float[torch.Tensor, "n dim"], distance: PeakDistance) -> float:
    """How far a float32 distance can be from the exact one, for any summation order.

    - Rounding the rows and each difference costs about 2u per dim, with u the float32 unit roundoff.
    - Summing d terms costs at most (d - 1)u of the sum of the terms.
    - Weighted Jaccard is 2L / (S + L), with L the L1 distance and S the two rows' sums.
      - L's error is at most (d + 2)u times S, and the Jaccard moves by at most 2 / S per unit of L.
      - The finishing arithmetic in float32 adds a few u more.
    - Euclidean errors scale with the rows' norms, so the bound uses the largest norm.
    - The margin is a few percent over the worst case, and only costs a few more candidates."""
    dims = rows.shape[1]
    if distance == "weighted_jaccard":
        return (2.0 * (dims + 2) * 1.01 + 16.0) * FLOAT32_ROUNDING
    largest_norm = rows.norm(dim=1).max().item() if len(rows) else 0.0
    return 2.0 * (dims + 4) * 1.01 * largest_norm * FLOAT32_ROUNDING


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


def candidates_within(
    distances: Float[torch.Tensor, "block n"], cutoff: Float[torch.Tensor, "block"],
) -> Int[torch.Tensor, "block c"]:
    """At least every column at or under each row's cutoff, closest first.

    - Rows share one width, the widest row's count, so other rows get a few extra columns.
    - Extra columns are harmless, since the exact distances decide."""
    width = max(1, int((distances <= cutoff[:, None]).sum(dim=1).max().item()))
    return torch.topk(distances, width, dim=1, largest=False, sorted=False).indices


def by_index(
    others: Int[torch.Tensor, "block c"], distance: Float[torch.Tensor, "block c"],
) -> tuple[Int[torch.Tensor, "block c"], Float[torch.Tensor, "block c"]]:
    """Columns in index order, so a stable sort or `argmin` after it breaks ties to the lower index."""
    order = torch.argsort(others, dim=1)
    return others.gather(1, order), distance.gather(1, order)


def kth_from_candidates(
    distance: Float[torch.Tensor, "block c"], others: Int[torch.Tensor, "block c"], own: Int[torch.Tensor, "block"],
    counts: Int[torch.Tensor, "n"], k: int, n_nearest: int,
) -> Float[torch.Tensor, "block"]:
    """`vectorized.kth_distances` for one block: the point's own copies come first, then the closest others."""
    is_self = others == own[:, None]
    distance = torch.where(is_self, -1.0, distance)
    order = torch.argsort(distance, dim=1)
    nearest_distance, nearest, is_self = distance.gather(1, order), others.gather(1, order), is_self.gather(1, order)
    hits = torch.where(is_self, counts[own, None] - 1, counts[nearest])
    reached = (torch.cumsum(hits, dim=1) < k).sum(dim=1).clamp(max=n_nearest - 1)
    return nearest_distance[torch.arange(len(own), device=own.device), reached].clamp(min=0.0)


def neighbours_from_candidates(
    distance: Float[torch.Tensor, "block c"], others: Int[torch.Tensor, "block c"], own: Int[torch.Tensor, "block"],
    n: int,
) -> tuple[Int[torch.Tensor, "block n_neighbours"], Float[torch.Tensor, "block n_neighbours"]]:
    """Closest first, ties to the lower index, as in `vectorized.nearest_neighbours`."""
    others, distance = by_index(others, torch.where(others == own[:, None], torch.inf, distance))
    closest_first = torch.argsort(distance, dim=1, stable=True)[:, :n]
    return others.gather(1, closest_first), distance.gather(1, closest_first)


def kth_and_neighbours(
    rows: TorchRows, counts: Int[np.ndarray, "n"], k: int, block_elements: int,
) -> tuple[Float[np.ndarray, "n"], Int[np.ndarray, "n n_neighbours"], Float[np.ndarray, "n n_neighbours"]]:
    """The true k + 1 nearest, the point included, are always within twice the error bound of the float32 (k + 1)th."""
    device = rows.rows.device
    counts_t = torch.from_numpy(counts).to(device)
    n = min(k, len(rows) - 1)
    n_nearest = min(k + 1, len(rows))
    kth = torch.empty(len(rows), dtype=torch.float64, device=device)
    neighbours = torch.empty((len(rows), max(n, 0)), dtype=torch.int64, device=device)
    neighbour_distance = torch.empty((len(rows), max(n, 0)), dtype=torch.float64, device=device)
    for start, block in rows.fast_blocks(block_elements):
        own = torch.arange(start, start + len(block), device=device)
        block[torch.arange(len(block), device=device), own] = -1.0
        furthest_kept = torch.topk(block, n_nearest, dim=1, largest=False, sorted=True).values[:, -1]
        others = candidates_within(block, furthest_kept + 2 * rows.error_bound)
        distance = rows.exact_pairs(own, others, block_elements)
        kth[own] = kth_from_candidates(distance, others, own, counts_t, k, n_nearest)
        if n > 0:
            neighbours[own], neighbour_distance[own] = neighbours_from_candidates(distance, others, own, n)
    return kth.cpu().numpy(), neighbours.cpu().numpy(), neighbour_distance.cpu().numpy()


def nearest_denser(
    rows: TorchRows, ranks: Int[np.ndarray, "n"], block_elements: int,
) -> tuple[Int[np.ndarray, "n"], Float[np.ndarray, "n"]]:
    """`vectorized.nearest_denser`, where ties go to the lower index.

    - The true nearest denser point is within twice the error bound of the float32 nearest.
    - A point with nothing denser takes its exact largest distance, which only the densest point needs."""
    device = rows.rows.device
    ranks_t = torch.from_numpy(ranks).to(device)
    parent = torch.full((len(rows),), -1, dtype=torch.int64, device=device)
    delta = torch.zeros(len(rows), dtype=torch.float64, device=device)
    for start, block in rows.fast_blocks(block_elements):
        own = torch.arange(start, start + len(block), device=device)
        denser = ranks_t[None, :] > ranks_t[own, None]
        has_denser = denser.any(dim=1)
        if (~has_denser).any():
            delta[own[~has_denser]] = rows.exact_rows(own[~has_denser]).max(dim=1).values
        if not has_denser.any():
            continue
        own, denser = own[has_denser], denser[has_denser]
        fast = torch.where(denser, block[has_denser], torch.inf)
        others = candidates_within(fast, fast.min(dim=1).values + 2 * rows.error_bound)
        distance = torch.where(denser.gather(1, others), rows.exact_pairs(own, others, block_elements), torch.inf)
        others, distance = by_index(others, distance)
        nearest = distance.argmin(dim=1, keepdim=True)
        parent[own], delta[own] = others.gather(1, nearest)[:, 0], distance.gather(1, nearest)[:, 0]
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
    """Every pair of hits, as `comparable_rows` compares them, with a zero diagonal.

    - It is float32, within `float32_error_bound` of exact, since only the layout reads it and it is saved as float32."""
    rows = TorchRows(comparable_rows(points.astype(np.float64), distance, normalise), distance, device)
    result = np.empty((len(rows), len(rows)), dtype=np.float32)
    for start, block in rows.fast_blocks(block_elements):
        result[start:start + len(block)] = block.cpu().numpy()
    np.fill_diagonal(result, 0.0)
    return result
