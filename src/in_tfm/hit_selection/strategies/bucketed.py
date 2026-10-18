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
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel, ConfigDict

from ...neuron_capture import ScanResult
from ...run_config import HitSelectionConfig
from ..picks import BucketedPick, PickedPositions


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


def equal_width_edges(values: Float[np.ndarray, "n"], n_buckets: int, floor: float) -> list[float]:
    return np.linspace(floor, values.max(), n_buckets + 1).tolist()


def raw_bucket_of_each_value(values: Float[np.ndarray, "n"], raw_edges: list[float]) -> Int[np.ndarray, "n"]:
    """Raw bucket `b` holds `raw_edges[b] <= value < raw_edges[b + 1]`.

    - `np.digitize` returns, per value, how many edges are at or below it, so 1 for the first bucket.
    - The max sits on the last edge and would get its own bucket past the end, so it is clipped into the last one.
    """
    last_bucket = len(raw_edges) - 2
    return np.clip(np.digitize(values, raw_edges) - 1, 0, last_bucket)


def count_per_bucket(bucket_of_each_value: Int[np.ndarray, "n"], n_buckets: int) -> list[int]:
    return [int((bucket_of_each_value == bucket).sum()) for bucket in range(n_buckets)]


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
    candidate_values: Float[np.ndarray, "n"], n_buckets: int, floor: float, min_bucket_size: int
) -> BucketLayout:
    raw_edges = equal_width_edges(candidate_values, n_buckets, floor)
    sizes = count_per_bucket(raw_bucket_of_each_value(candidate_values, raw_edges), n_buckets)
    return BucketLayout(raw_edges=raw_edges, groups=merge_small_buckets(sizes, min_bucket_size))


def group_of_each_value(values: Float[np.ndarray, "n"], layout: BucketLayout) -> Int[np.ndarray, "n"]:
    """Looks up each value's raw bucket in `group_of_raw_bucket`.

    - The indexing below is the same as `[group_of_raw_bucket[b] for b in raw_buckets]`.
    """
    raw_buckets = raw_bucket_of_each_value(values, layout.raw_edges)
    return layout.group_of_raw_bucket()[raw_buckets]


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
    group_of_candidate: Int[np.ndarray, "n_candidates"], quotas: list[int], rng: np.random.Generator
) -> Int[np.ndarray, "n_kept"]:
    """Candidate indices, ascending. Groups are drawn lowest first, which fixes what a seed picks."""
    picked = []
    for group, quota in enumerate(quotas):
        members = np.flatnonzero(group_of_candidate == group)
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
    candidate_value_idx = np.flatnonzero(values > floor)
    if len(candidate_value_idx) == 0:
        return BucketedSample(kept_value_idx=candidate_value_idx, group_of_kept=candidate_value_idx, n_candidates=0, layout=None)
    candidate_values = values[candidate_value_idx]
    layout = plan_buckets(candidate_values, n_buckets, floor, min_bucket_size)
    group_of_candidate = group_of_each_value(candidate_values, layout)
    quotas = bucket_quotas(count_per_bucket(group_of_candidate, len(layout.groups)), budget)
    kept_candidate_idx = sample_each_group(group_of_candidate, quotas, rng)
    return BucketedSample(
        kept_value_idx=candidate_value_idx[kept_candidate_idx],
        group_of_kept=group_of_candidate[kept_candidate_idx],
        n_candidates=len(candidate_value_idx),
        layout=layout,
    )


def valid_values_and_positions(
    column: Float[torch.Tensor, "batch seq 1"], valid_mask: Bool[torch.Tensor, "batch seq"]
) -> tuple[Float[np.ndarray, "n_valid"], Int[np.ndarray, "n_valid"], Int[np.ndarray, "n_valid"]]:
    """Every valid position's value, with its batch and token index.

    - Boolean-mask indexing and `argwhere` both walk the grid row by row, so entry i of each output is the same position.
    """
    valid = valid_mask.numpy()
    values = column[..., 0].numpy()[valid]
    positions = np.argwhere(valid)
    return values, positions[:, 0], positions[:, 1]


def bucketed_hits(
    column: Float[torch.Tensor, "batch seq 1"],
    sorted_magnitudes: Float[torch.Tensor, "n_pos"],
    scan: ScanResult,
    hits: HitSelectionConfig,
    seed: int,
) -> tuple[PickedPositions, BucketedPick]:
    floor = hits.bucket_floor_fraction * sorted_magnitudes[-1].item()
    values, batch_idx, token_idx = valid_values_and_positions(column, scan.valid_mask)
    sample = bucketed_sample(
        values, hits.n_buckets, hits.max_hits, floor, hits.min_bucket_size, np.random.default_rng(seed)
    )
    if sample.layout is None:
        raise ValueError(f"no activation above the bucket floor {floor:.4f}, lower bucket_floor_fraction")
    kept = sample.kept_value_idx
    picked = PickedPositions(batch_idx=batch_idx[kept], token_idx=token_idx[kept], n_candidates=sample.n_candidates)
    pick = BucketedPick(floor=floor, bucket_ids=sample.group_of_kept, bucket_edges=sample.layout.group_edges())
    return picked, pick
