"""Bucketed subsampling of a neuron's activations, so clustering stays tractable without
needing an elbow.

Activations are cut into equal-width buckets between a floor and the max.
Most positions sit in the low buckets and the tail buckets hold few.
Buckets with fewer than `min_bucket_size` values are merged with their neighbour, since a bucket
too small to hold a cluster of its own has no use as a separate stratum.
Each remaining bucket gets an equal share of the budget, so the rare high activations are not
drowned out.

Names used below:
- A raw bucket is one of the `n_buckets` equal-width slices.
- A group is one or more neighbouring raw buckets after merging. Group 0 is the lowest.
- A candidate is a value above the floor. `candidate_idx` arrays index into the candidates, not into all values.
"""

import numpy as np
import torch
from jaxtyping import Bool, Float, Int, UInt8
from pydantic import BaseModel, ConfigDict

from ...neuron_capture import ScanResult
from ...run_config import HitSelectionConfig
from ..picks import BucketedPick, PickedPositions

CHUNK_VALUES = 2**24
"""`np.digitize` returns int64, so a whole full-corpus column at once would take 3.9 GB."""
NOT_A_CANDIDATE = 255


class BucketLayout(BaseModel):
    raw_edges: list[float]
    """`n_buckets + 1` edges, from the floor to the max."""
    groups: list[list[int]]
    """The raw buckets in each group, lowest group first."""

    def group_edges(self) -> Float[np.ndarray, "n_groups_plus_1"]:
        """Lower edge of every group, then the max."""
        lower_edges = [self.raw_edges[raw_buckets[0]] for raw_buckets in self.groups]
        return np.array(lower_edges + [self.raw_edges[-1]])

    def group_of_raw_bucket(self) -> Int[np.ndarray, "n_buckets"]:
        group_of = np.zeros(len(self.raw_edges) - 1, dtype=int)
        for group, raw_buckets in enumerate(self.groups):
            for raw_bucket in raw_buckets:
                group_of[raw_bucket] = group
        return group_of


class BucketedSample(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    kept_value_idx: Int[np.ndarray, "n_kept"]
    """Indices into the values passed in, ascending."""
    group_of_kept: Int[np.ndarray, "n_kept"]
    n_candidates: int
    layout: BucketLayout | None
    """None when no value is above the floor."""


def equal_width_edges(
    values: Float[np.ndarray, "n"], n_buckets: int, lowest_edge: float
) -> list[float]:
    return np.linspace(lowest_edge, values.max(), n_buckets + 1).tolist()


def raw_bucket_of_each_value(
    values: Float[np.ndarray, "n"], raw_edges: list[float]
) -> Int[np.ndarray, "n"]:
    """Raw bucket `b` holds `raw_edges[b] <= value < raw_edges[b + 1]`.

    - `np.digitize` returns, per value, how many edges are at or below it, so 1 for the first bucket.
    - The max sits on the last edge and would get its own bucket past the end, so it is clipped into the last one.
    """
    last_bucket = len(raw_edges) - 2
    return np.clip(np.digitize(values, raw_edges) - 1, 0, last_bucket)


def raw_bucket_sizes(
    values: Float[np.ndarray, "n"], raw_edges: list[float]
) -> list[int]:
    """Values strictly above the lowest edge, per raw bucket, a chunk at a time."""
    n_buckets = len(raw_edges) - 1
    sizes = np.zeros(n_buckets, dtype=np.int64)
    for start in range(0, len(values), CHUNK_VALUES):
        chunk = values[start : start + CHUNK_VALUES]
        chunk = chunk[chunk > raw_edges[0]]
        sizes += np.bincount(
            raw_bucket_of_each_value(chunk, raw_edges), minlength=n_buckets
        )
    return sizes.tolist()


def merge_small_buckets(sizes: list[int], min_size: int) -> list[list[int]]:
    """The raw buckets in each group, lowest group first.

    - Walks down from the sparse top, closing a group once it holds `min_size` values.
    - A leftover lowest group that is still too small joins the group above it.
    """
    groups_from_top: list[list[int]] = []
    open_group: list[int] = []
    held = 0
    for raw_bucket in reversed(range(len(sizes))):
        open_group.append(raw_bucket)
        held += sizes[raw_bucket]
        if held >= min_size:
            groups_from_top.append(open_group)
            open_group, held = [], 0
    if open_group and groups_from_top:
        groups_from_top[-1].extend(open_group)
    elif open_group:
        groups_from_top.append(open_group)
    return [sorted(group) for group in reversed(groups_from_top)]


def plan_buckets(
    values: Float[np.ndarray, "n"],
    n_buckets: int,
    lowest_edge: float,
    min_bucket_size: int,
) -> BucketLayout:
    """Only values above `lowest_edge` count. Assumes at least one is."""
    raw_edges = equal_width_edges(values, n_buckets, lowest_edge)
    sizes = raw_bucket_sizes(values, raw_edges)
    return BucketLayout(
        raw_edges=raw_edges, groups=merge_small_buckets(sizes, min_bucket_size)
    )


def group_sizes(values: Float[np.ndarray, "n"], layout: BucketLayout) -> list[int]:
    raw_sizes = raw_bucket_sizes(values, layout.raw_edges)
    return [sum(raw_sizes[raw] for raw in raw_buckets) for raw_buckets in layout.groups]


def group_code_of_each_value(
    values: Float[np.ndarray, "n"], layout: BucketLayout
) -> UInt8[np.ndarray, "n"]:
    """Each value's group, or `NOT_A_CANDIDATE`, in one byte per value.

    - Looks up each value's raw bucket in `group_of_raw_bucket`, a chunk at a time.
    - One byte matters at full-corpus scale: int64 labels for 486M values would take 3.9 GB, on a machine with 12.
    """
    assert len(layout.groups) < NOT_A_CANDIDATE
    group_of_raw = layout.group_of_raw_bucket().astype(np.uint8)
    codes = np.full(len(values), NOT_A_CANDIDATE, dtype=np.uint8)
    for start in range(0, len(values), CHUNK_VALUES):
        chunk = values[start : start + CHUNK_VALUES]
        candidates = chunk > layout.raw_edges[0]
        codes[start : start + CHUNK_VALUES][candidates] = group_of_raw[
            raw_bucket_of_each_value(chunk[candidates], layout.raw_edges)
        ]
    return codes


def bucket_quotas(sizes: list[int], budget: int) -> list[int]:
    """Equal share per bucket, where a small bucket gives all it has.
    The unused share is spread over the larger buckets, so the whole budget is spent whenever
    there are enough values."""
    quotas = [0] * len(sizes)
    remaining = budget
    smallest_first = np.argsort(sizes)
    for position, bucket in enumerate(smallest_first):
        buckets_left = len(sizes) - position
        quotas[bucket] = min(sizes[bucket], remaining // buckets_left)
        remaining -= quotas[bucket]
    return quotas


def sample_each_group(
    group_codes: UInt8[np.ndarray, "n"],
    quotas: list[int],
    rng: np.random.Generator,
) -> Int[np.ndarray, "n_kept"]:
    """Value indices, ascending. Groups are drawn lowest first, which fixes what a seed picks."""
    picked = []
    for group, quota in enumerate(quotas):
        members = np.flatnonzero(group_codes == group)
        picked.append(rng.choice(members, quota, replace=False))
    return np.sort(np.concatenate(picked))


def bucketed_sample(
    values: Float[np.ndarray, "n"],
    n_buckets: int,
    budget: int,
    floor: float,
    min_bucket_size: int,
    rng: np.random.Generator,
) -> BucketedSample:
    if len(values) == 0 or values.max() <= floor:
        nothing = np.zeros(0, dtype=np.int64)
        return BucketedSample(
            kept_value_idx=nothing,
            group_of_kept=nothing,
            n_candidates=0,
            layout=None,
        )
    layout = plan_buckets(values, n_buckets, floor, min_bucket_size)
    sizes = group_sizes(values, layout)
    group_codes = group_code_of_each_value(values, layout)
    kept = sample_each_group(group_codes, bucket_quotas(sizes, budget), rng)
    return BucketedSample(
        kept_value_idx=kept,
        group_of_kept=group_codes[kept].astype(np.int64),
        n_candidates=sum(sizes),
        layout=layout,
    )


def valid_values(
    column: Float[torch.Tensor, "batch seq 1"],
    valid_mask: Bool[torch.Tensor, "batch seq"],
) -> Float[np.ndarray, "n_valid"]:
    """Row by row. With no padding this is a view, not a copy, which saves 1.9 GB on a full-corpus scan."""
    grid = column[..., 0].numpy()
    return grid.ravel() if valid_mask.all() else grid[valid_mask.numpy()]


def positions_of_valid(
    value_idx: Int[np.ndarray, "n"], valid_mask: Bool[torch.Tensor, "batch seq"]
) -> tuple[Int[np.ndarray, "n"], Int[np.ndarray, "n"]]:
    """The batch and token index of entries of `valid_values`.

    - With padding, `argwhere` walks the grid row by row, as `valid_values` does, so entry i is the same position.
    - With none, the positions come from arithmetic, since `argwhere` takes 7.8 GB on 486M positions.
    """
    if valid_mask.all():
        return np.divmod(value_idx, valid_mask.shape[1])
    positions = np.argwhere(valid_mask.numpy())[value_idx]
    return positions[:, 0], positions[:, 1]


def bucket_floor(
    sorted_magnitudes: Float[torch.Tensor, "n_pos"], floor_fraction: float
) -> float:
    return floor_fraction * sorted_magnitudes[-1].item()


def bucketed_hits(
    column: Float[torch.Tensor, "batch seq 1"],
    sorted_magnitudes: Float[torch.Tensor, "n_pos"],
    scan: ScanResult,
    hits: HitSelectionConfig,
    budget: int,
    seed: int,
) -> tuple[PickedPositions, BucketedPick]:
    floor = bucket_floor(sorted_magnitudes, hits.bucket_floor_fraction)
    sample = bucketed_sample(
        valid_values(column, scan.valid_mask),
        hits.n_buckets,
        budget,
        floor,
        hits.min_bucket_size,
        np.random.default_rng(seed),
    )
    if sample.layout is None:
        raise ValueError(
            f"no activation above the bucket floor {floor:.4f}, lower bucket_floor_fraction"
        )
    batch_idx, token_idx = positions_of_valid(sample.kept_value_idx, scan.valid_mask)
    picked = PickedPositions(
        batch_idx=batch_idx,
        token_idx=token_idx,
        n_candidates=sample.n_candidates,
    )
    pick = BucketedPick(
        floor=floor,
        bucket_ids=sample.group_of_kept,
        bucket_edges=sample.layout.group_edges(),
    )
    return picked, pick
