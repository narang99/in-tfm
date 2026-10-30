"""The forest from approximate neighbours, for more points than an all-pairs pass can take.

- Both passes of `torch_forest` compare every pair, so its time grows with n².
  - On a T4, 256k points took 50-136 s per forest, so 10M points would take days.
- Here a point is compared only with the points in a few lists of an IVF index.
  - The lists are k-means cells of the signed rows, in euclidean distance.
  - Inside the probed lists the distance is the forest's own. Only the choice of which pairs get compared is approximate.
- The parent comes from the neighbour list.
  - In a closest-first list, the first denser point is the nearest denser point among everything compared.
  - Only points denser than their whole list (local peaks) need a second search, over denser points only.
  - Peaks that the second search leaves without a denser point are compared with every point.
- Copies are not merged. Each point is its own row, with a count of 1.
  - Exact copies tie on density, and `density_ranks` puts the lower index as sparser.
- With `n_probe` equal to `n_lists` every pair is compared, so the forest is exact. The tests use that.
"""

import time
from collections.abc import Callable, Iterator

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel, ConfigDict

from .forest import DensityPeakForest, PeakDistance, comparable_rows
from .torch_forest import float32_distances, signed_rows, weighted_jaccard_from_l1
from .vectorized import density_ranks

BLOCK_ELEMENTS = 2**26
"""The cap on one block of distances. Tests pass a small value to force several blocks."""
PROBE_PAIRS = 2**25
"""The cap on (query, probed list) pairs held on the device at once."""
TRAIN_PER_LIST = 64
KMEANS_ITERATIONS = 20

AllowedPairs = Callable[[Int[torch.Tensor, "q"], Int[torch.Tensor, "m"]], Bool[torch.Tensor, "q m"]]


class IvfSettings(BaseModel):
    n_lists: int
    n_probe: int
    peak_probe: int
    """Lists probed in the second search, for local peaks."""
    n_list_neighbours: int = 32
    """The length of each point's closest-first list. Longer lists leave fewer local peaks."""
    seed: int = 0
    exact_on_device: bool = False
    """Float64 rows on the device too. At 16M points they took a quarter of a T4, but an A100 has room."""


class SearchStats(BaseModel):
    seconds: dict[str, float]
    """Per stage, in the order they ran."""
    local_peaks: int
    """Points with no denser point in their list."""
    exhaustive: int
    """Local peaks the second search left without a denser point."""


class Rows:
    """Signed rows in float32 on the device for every compared pair, and in float64 on the CPU for the few pairs that
    set values.

    - Float32 rows alone move kth distances by about 1e-6, which swaps near-tied density ranks.
    - The float64 rows stay on the CPU unless `exact_on_device`, since at 16M points they would take a quarter of a T4."""

    def __init__(self, points: Float[np.ndarray, "n dim"], distance: PeakDistance, normalise: bool, device: str,
                 chunk: int = 2**20, exact_on_device: bool = False):
        exact, exact_sums = [], []
        for start in range(0, len(points), chunk):
            parts = torch.from_numpy(comparable_rows(points[start:start + chunk].astype(np.float64), distance,
                                                     normalise))
            exact.append(signed_rows(parts) if distance == "weighted_jaccard" else parts)
            exact_sums.append(parts.sum(dim=1))
        self.distance = distance
        self.exact = torch.cat(exact)
        self.exact_sums = torch.cat(exact_sums) if distance == "weighted_jaccard" else None
        self.signed = self.exact.float().to(device)
        self.sums = self.exact_sums.float().to(device) if self.exact_sums is not None else None
        if exact_on_device:
            self.exact = self.exact.to(device)
            self.exact_sums = self.exact_sums.to(device) if self.exact_sums is not None else None

    def __len__(self) -> int:
        return len(self.signed)

    def between(self, a: Int[torch.Tensor, "a"], b: Int[torch.Tensor, "b"]) -> Float[torch.Tensor, "a b"]:
        sums_a = self.sums[a] if self.sums is not None else None
        sums_b = self.sums[b] if self.sums is not None else None
        return float32_distances(self.signed[a], self.signed[b], sums_a, sums_b, self.distance)

    def exact_pairs(self, a: Int[torch.Tensor, "m"], b: Int[torch.Tensor, "m"], chunk: int = 2**22) -> Float[torch.Tensor, "m"]:
        """The float64 distance between a[i] and b[i], for each i, on the device of `a`."""
        exact_device = self.exact.device
        parts = [self.exact_chunk(a_part.to(exact_device), b_part.to(exact_device))
                 for a_part, b_part in zip(a.split(chunk), b.split(chunk))]
        return (torch.cat(parts) if parts else torch.empty(0, dtype=torch.float64)).to(a.device)

    def exact_chunk(self, a: Int[torch.Tensor, "m"], b: Int[torch.Tensor, "m"]) -> Float[torch.Tensor, "m"]:
        diff = self.exact[a] - self.exact[b]
        if self.distance == "euclidean":
            return diff.square().sum(dim=1).sqrt()
        both = (self.exact_sums[a] + self.exact_sums[b])[:, None]
        return weighted_jaccard_from_l1(diff.abs().sum(dim=1)[:, None], both)[:, 0]


class IvfLists(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    centroids: Float[torch.Tensor, "n_lists dim"]
    members: Int[torch.Tensor, "n"]
    """Point ids, grouped by list."""
    starts: Int[torch.Tensor, "n_lists_plus_one"]
    """List l holds members[starts[l]:starts[l + 1]]."""


def nearest_centroids(
    rows: Float[torch.Tensor, "n dim"], centroids: Float[torch.Tensor, "c dim"], n: int,
) -> Int[torch.Tensor, "n n_nearest"]:
    size = max(1, BLOCK_ELEMENTS // len(centroids))
    return torch.cat([torch.topk(torch.cdist(rows[s:s + size], centroids), n, dim=1, largest=False).indices
                      for s in range(0, len(rows), size)])


def kmeans_centroids(rows: Float[torch.Tensor, "n dim"], n_lists: int, generator: torch.Generator):
    train = rows[torch.randperm(len(rows), generator=generator, device="cpu")[:n_lists * TRAIN_PER_LIST]
                 .to(rows.device)]
    centroids = train[:n_lists].clone()
    for _ in range(KMEANS_ITERATIONS):
        assigned = nearest_centroids(train, centroids, 1)[:, 0]
        sums = torch.zeros_like(centroids).index_add_(0, assigned, train)
        counts = torch.bincount(assigned, minlength=n_lists)[:, None]
        centroids = torch.where(counts > 0, sums / counts.clamp(min=1), centroids)
    return centroids


def build_lists(rows: Rows, n_lists: int, seed: int) -> IvfLists:
    n_lists = min(n_lists, len(rows))
    centroids = kmeans_centroids(rows.signed, n_lists, torch.Generator().manual_seed(seed))
    list_of_point = nearest_centroids(rows.signed, centroids, 1)[:, 0]
    members = torch.argsort(list_of_point, stable=True)
    counts = torch.bincount(list_of_point, minlength=n_lists)
    starts = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
    return IvfLists(centroids=centroids, members=members, starts=starts)


def closest_first(
    nearest: Int[torch.Tensor, "q c"], distance: Float[torch.Tensor, "q c"], n: int,
) -> tuple[Int[torch.Tensor, "q n"], Float[torch.Tensor, "q n"]]:
    """Ties go to the lower index, as in `vectorized.nearest_neighbours`. Empty slots are -1 at distance inf."""
    by_index = torch.argsort(torch.where(torch.isinf(distance), torch.iinfo(torch.int64).max, nearest), dim=1)
    nearest, distance = nearest.gather(1, by_index), distance.gather(1, by_index)
    order = torch.argsort(distance, dim=1, stable=True)[:, :n]
    return nearest.gather(1, order), distance.gather(1, order)


def probe_segments(probes: Int[torch.Tensor, "q n_probe"]) -> Iterator[tuple[int, Int[torch.Tensor, "pairs"]]]:
    """Each probed list with the queries (as positions in `probes`) that probe it."""
    lists = probes.flatten()
    order = torch.argsort(lists, stable=True)
    query = torch.arange(len(probes), device=probes.device).repeat_interleave(probes.shape[1])[order]
    present, counts = torch.unique_consecutive(lists[order], return_counts=True)
    for list_id, segment in zip(present.tolist(), torch.split(query, counts.tolist())):
        yield list_id, segment


def search_list(
    rows: Rows, queries: Int[torch.Tensor, "q"], members: Int[torch.Tensor, "m"], n: int,
    allowed: AllowedPairs | None,
) -> tuple[Int[torch.Tensor, "q n_found"], Float[torch.Tensor, "q n_found"]]:
    distance = rows.between(queries, members)
    excluded = members[None, :] == queries[:, None]
    if allowed is not None:
        excluded |= ~allowed(queries, members)
    distance = torch.where(excluded, torch.inf, distance)
    found = min(n, len(members))
    distance, column = torch.topk(distance, found, dim=1, largest=False)
    return torch.where(torch.isinf(distance), -1, members[column]), distance


def exhaustive_search(
    rows: Rows, queries: Int[torch.Tensor, "q"], n: int, allowed: AllowedPairs | None,
) -> tuple[Int[torch.Tensor, "q n"], Float[torch.Tensor, "q n"]]:
    """Every point compared, without the probe bookkeeping that probing every list would hold."""
    every = torch.arange(len(rows), device=queries.device)
    found = [search_list(rows, part, every, n, allowed) for part in torch.split(queries, max(1, BLOCK_ELEMENTS // len(rows)))]
    nearest = torch.cat([f[0] for f in found]) if found else queries.new_empty((0, n))
    distance = torch.cat([f[1] for f in found]) if found else queries.new_empty((0, n), dtype=torch.float32)
    return closest_first(nearest, distance, n)


def ivf_search(
    rows: Rows, lists: IvfLists, queries: Int[torch.Tensor, "q"], n_probe: int, n: int,
    allowed: AllowedPairs | None = None,
) -> tuple[Int[torch.Tensor, "q n"], Float[torch.Tensor, "q n"]]:
    """The n closest points to each query among the lists it probes, closest first. Never the query itself."""
    if n_probe >= len(lists.centroids):
        return exhaustive_search(rows, queries, n, allowed)
    return probed_search(rows, lists, queries, n_probe, n, allowed)


def probed_search(
    rows: Rows, lists: IvfLists, queries: Int[torch.Tensor, "q"], n_probe: int, n: int,
    allowed: AllowedPairs | None = None,
) -> tuple[Int[torch.Tensor, "q n"], Float[torch.Tensor, "q n"]]:
    device = rows.signed.device
    nearest = torch.full((len(queries), n), -1, dtype=torch.int64, device=device)
    distance = torch.full((len(queries), n), torch.inf, dtype=torch.float32, device=device)
    query_chunk = max(1, PROBE_PAIRS // n_probe)
    for chunk_start in range(0, len(queries), query_chunk):
        chunk = queries[chunk_start:chunk_start + query_chunk]
        probes = nearest_centroids(rows.signed[chunk], lists.centroids, n_probe)
        for list_id, positions in probe_segments(probes):
            members = lists.members[lists.starts[list_id]:lists.starts[list_id + 1]]
            # a centroid can end up with no points once every point is assigned
            if len(members) == 0:
                continue
            size = max(1, BLOCK_ELEMENTS // len(members))
            for part in torch.split(positions + chunk_start, size):
                found, found_distance = search_list(rows, queries[part], members, n, allowed)
                nearest[part], distance[part] = closest_first(
                    torch.cat([nearest[part], found], dim=1), torch.cat([distance[part], found_distance], dim=1), n,
                )
    return nearest, distance


def first_denser(
    nearest: Int[torch.Tensor, "n c"], ranks: Int[torch.Tensor, "n"], chunk: int = 2**20,
) -> Int[torch.Tensor, "n"]:
    """The first denser point in each closest-first list, or -1."""
    return torch.cat([first_denser_chunk(part, ranks, start) for start, part in
                      zip(range(0, len(nearest), chunk), nearest.split(chunk))])


def first_denser_chunk(
    nearest: Int[torch.Tensor, "m c"], ranks: Int[torch.Tensor, "n"], start: int,
) -> Int[torch.Tensor, "m"]:
    own_ranks = ranks[start:start + len(nearest), None]
    denser = (nearest >= 0) & (ranks[nearest.clamp(min=0)] > own_ranks)
    first = denser.int().argmax(dim=1)
    return torch.where(denser.any(dim=1), nearest.gather(1, first[:, None])[:, 0], -1)


def nearest_denser_everywhere(
    rows: Rows, points: Int[torch.Tensor, "m"], ranks: Int[torch.Tensor, "n"],
) -> Int[torch.Tensor, "m"]:
    """Every point compared, as in `torch_forest.nearest_denser`. The densest point gets its farthest point."""
    every = torch.arange(len(rows), device=ranks.device)
    size = max(1, BLOCK_ELEMENTS // len(rows))
    parents = []
    for part in torch.split(points, size):
        block = rows.between(part, every)
        denser = ranks[None, :] > ranks[part, None]
        nearest = torch.where(denser, block, torch.inf).argmin(dim=1)
        parents.append(torch.where(denser.any(dim=1), nearest, -block.argmax(dim=1) - 2))
    return torch.cat(parents)


def kth_from_list(
    rows: Rows, nearest: Int[torch.Tensor, "n c"], k: int,
) -> Float[torch.Tensor, "n"]:
    """With every count 1, the kth distance is the distance to the kth nearest other point."""
    own = torch.arange(len(rows), device=nearest.device)
    if nearest.shape[1] == 0:
        return torch.zeros(len(rows), dtype=torch.float64, device=nearest.device)
    return rows.exact_pairs(own, nearest[:, min(k, nearest.shape[1]) - 1])


class StageClock:
    def __init__(self, device: str):
        self.device = device
        self.seconds: dict[str, float] = {}
        self.last = time.perf_counter()

    def lap(self, stage: str) -> None:
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        now = time.perf_counter()
        self.seconds[stage] = now - self.last
        self.last = now


def ivf_density_peak_forest(
    points: Float[np.ndarray, "n_hits dim"],
    settings: IvfSettings,
    k: int = 5,
    distance: PeakDistance = "weighted_jaccard",
    normalise: bool = True,
    device: str = "cuda",
) -> tuple[DensityPeakForest, SearchStats]:
    clock = StageClock(device)
    rows = Rows(points, distance, normalise, device, exact_on_device=settings.exact_on_device)
    lists = build_lists(rows, settings.n_lists, settings.seed)
    clock.lap("lists")
    own = torch.arange(len(rows), device=rows.signed.device)
    list_length = min(max(settings.n_list_neighbours, k), len(rows) - 1)
    nearest, _ = ivf_search(rows, lists, own, settings.n_probe, list_length)
    kth = kth_from_list(rows, nearest, k)
    counts = np.ones(len(rows), dtype=int)
    ranks_np = density_ranks(kth.cpu().numpy(), counts)
    ranks = torch.from_numpy(ranks_np).to(rows.signed.device)
    clock.lap("neighbours")

    parent = first_denser(nearest, ranks)
    peaks = own[parent < 0]
    denser_only: AllowedPairs = lambda q, m: ranks[m][None, :] > ranks[q][:, None]  # noqa: E731
    second, _ = ivf_search(rows, lists, peaks, settings.peak_probe, 1, denser_only)
    parent[peaks] = second[:, 0]
    clock.lap("peaks")
    unresolved = own[parent < 0]
    parent[unresolved] = nearest_denser_everywhere(rows, unresolved, ranks)
    clock.lap("exhaustive")

    # the densest point is stored with -(farthest point) - 2 until its delta is read
    farthest = torch.where(parent < -1, -parent - 2, parent)
    delta = rows.exact_pairs(own, farthest)
    parent = parent.clamp(min=-1)
    n_neighbours = min(k, len(rows) - 1)
    neighbours = nearest[:, :n_neighbours]
    stats = SearchStats(seconds=clock.seconds, local_peaks=len(peaks), exhaustive=len(unresolved))
    return DensityPeakForest(
        distinct_of_hit=np.arange(len(rows)), counts=counts, kth_distance=kth.cpu().numpy(), density_rank=ranks_np,
        parent=parent.cpu().numpy(), delta=delta.cpu().numpy(), neighbours=neighbours.cpu().numpy(),
        neighbour_distance=rows.exact_pairs(own.repeat_interleave(n_neighbours), neighbours.flatten())
        .reshape(len(rows), n_neighbours).cpu().numpy(),
    ), stats
