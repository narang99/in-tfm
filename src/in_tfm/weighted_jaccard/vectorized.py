"""Weighted Jaccard with numpy broadcasting. `naive.py` computes the same with plain loops."""

import numpy as np
from jaxtyping import Float
from sklearn.preprocessing import normalize


def contribution_shares(points: Float[np.ndarray, "n_points dim"]) -> Float[np.ndarray, "n_points two_dim"]:
    """Each row's positive and negative parts side by side, L1-normalised.

    A row says where its contribution comes from, with signs kept apart so +a and -a never match.
    An all-zero row stays all zero."""
    parts = np.concatenate([points.clip(min=0), (-points).clip(min=0)], axis=1).astype(np.float64)
    return normalize(parts, norm="l1")


def weighted_jaccard_block(
    block: Float[np.ndarray, "block_size two_dim"], shares: Float[np.ndarray, "n_points two_dim"]
) -> Float[np.ndarray, "block_size n_points"]:
    """Distances from each row of `block` to every row of `shares`, both already contribution shares."""
    # (block_size, 1, two_dim) against (1, n_points, two_dim) broadcasts to (block_size, n_points, two_dim)
    # summing the last axis leaves one number per pair
    overlap = np.minimum(block[:, None, :], shares[None]).sum(axis=-1)
    union = np.maximum(block[:, None, :], shares[None]).sum(axis=-1)
    # two all-zero rows have no union: dividing by 1 instead gives distance 1
    return 1.0 - overlap / np.where(union > 0, union, 1.0)


def weighted_jaccard_distances(
    points: Float[np.ndarray, "n_points dim"], chunk: int = 256
) -> Float[np.ndarray, "n_points n_points"]:
    """1 - sum(min) / sum(max) between the rows' contribution shares, also called the Ruzicka distance.

    - It is plain Jaccard with each dim partly in a set, so a dim counts in proportion to its size.
    - It is a metric.
    - Pairs are built a chunk of rows at a time, since a full n x n x dim array would be large.
    - The math, line by line, with the chunking and a worked example: https://github.com/narang99/in-tfm/pull/91"""
    shares = contribution_shares(points)
    distances = np.empty((len(shares), len(shares)))
    for start in range(0, len(shares), chunk):
        distances[start:start + chunk] = weighted_jaccard_block(shares[start:start + chunk], shares)
    np.fill_diagonal(distances, 0.0)
    return distances
