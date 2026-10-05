"""Bucketed subsampling of a neuron's activations, so clustering stays tractable without
needing an elbow.

Activations are cut into equal-width buckets between a floor and the max.
Most positions sit in the low buckets and the tail buckets hold few.
Buckets with fewer than `min_bucket_size` values are merged with their neighbour, since a bucket
too small to hold a cluster of its own has no use as a separate stratum.
Each remaining bucket gets an equal share of the budget, so the rare high activations are not
drowned out.
"""

import numpy as np
from jaxtyping import Float, Int


def bucket_edges(
    values: Float[np.ndarray, "n"], n_buckets: int, floor: float
) -> Float[np.ndarray, "n_edges"]:
    return np.linspace(floor, values.max(), n_buckets + 1)


def assign_buckets(
    values: Float[np.ndarray, "n"], edges: Float[np.ndarray, "n_edges"]
) -> Int[np.ndarray, "n"]:
    """The max lands on the last edge, so it is clipped into the last bucket."""
    n_buckets = len(edges) - 1
    return np.clip(np.digitize(values, edges) - 1, 0, n_buckets - 1)


def merge_small_buckets(sizes: Int[np.ndarray, "n_buckets"], min_size: int) -> Int[np.ndarray, "n_buckets"]:
    """Group id (0 is the lowest) per bucket.
    Walks down from the sparse top, closing a group once it holds `min_size` values.
    A leftover lowest group that is still too small joins the group above it."""
    group_of = np.zeros(len(sizes), dtype=int)
    group, held = 0, 0
    for bucket in reversed(range(len(sizes))):
        group_of[bucket] = group
        held += sizes[bucket]
        if held >= min_size:
            group, held = group + 1, 0
    lowest = group_of.max()
    if lowest > 0 and sizes[group_of == lowest].sum() < min_size:
        group_of[group_of == lowest] = lowest - 1
    return group_of.max() - group_of


def merged_bucket_ids(
    values: Float[np.ndarray, "n"], n_buckets: int, floor: float, min_bucket_size: int
) -> Int[np.ndarray, "n"]:
    edges = bucket_edges(values, n_buckets, floor)
    buckets = assign_buckets(values, edges)
    return merge_small_buckets(np.bincount(buckets, minlength=n_buckets), min_bucket_size)[buckets]


def merged_bucket_edges(
    values: Float[np.ndarray, "n"], n_buckets: int, floor: float, min_bucket_size: int
) -> Float[np.ndarray, "n_groups_plus_1"]:
    """Lower edge of every merged bucket, then the max."""
    edges = bucket_edges(values, n_buckets, floor)
    sizes = np.bincount(assign_buckets(values, edges), minlength=n_buckets)
    group_of = merge_small_buckets(sizes, min_bucket_size)
    first_raw_bucket_of_each_group = np.flatnonzero(np.diff(group_of, prepend=-1))
    return np.append(edges[first_raw_bucket_of_each_group], edges[-1])


def bucket_quotas(sizes: Int[np.ndarray, "n_buckets"], budget: int) -> Int[np.ndarray, "n_buckets"]:
    """Equal share per bucket, where a small bucket gives all it has.
    The unused share is spread over the larger buckets, so the whole budget is spent whenever
    there are enough values."""
    quotas = np.zeros_like(sizes)
    remaining = budget
    smallest_first = np.argsort(sizes)
    for position, bucket in enumerate(smallest_first):
        buckets_left = len(sizes) - position
        quotas[bucket] = min(sizes[bucket], remaining // buckets_left)
        remaining -= quotas[bucket]
    return quotas


def bucketed_sample(
    values: Float[np.ndarray, "n"],
    n_buckets: int,
    budget: int,
    floor: float,
    min_bucket_size: int,
    rng: np.random.Generator,
) -> tuple[Int[np.ndarray, "n_kept"], Int[np.ndarray, "n_kept"]]:
    """Indices into `values` (ascending) of the kept values, and the merged bucket of each.
    Only values above `floor` are considered."""
    candidates = np.flatnonzero(values > floor)
    if len(candidates) == 0:
        return candidates, candidates
    ids = merged_bucket_ids(values[candidates], n_buckets, floor, min_bucket_size)
    n_groups = ids.max() + 1
    quotas = bucket_quotas(np.bincount(ids, minlength=n_groups), budget)
    kept = np.sort(
        np.concatenate([rng.choice(np.flatnonzero(ids == g), quotas[g], replace=False) for g in range(n_groups)])
    )
    return candidates[kept], ids[kept]
