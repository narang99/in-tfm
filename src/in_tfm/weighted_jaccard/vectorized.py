"""Weighted Jaccard with numpy broadcasting. `naive.py` computes the same with plain loops."""

import numpy as np
from jaxtyping import Float
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize


def sign_parts(points: Float[np.ndarray, "n_points dim"]) -> Float[np.ndarray, "n_points two_dim"]:
    """Each row's positive and negative parts side by side, so +a and -a never match."""
    return np.concatenate([points.clip(min=0), (-points).clip(min=0)], axis=1).astype(np.float64)


def contribution_shares(points: Float[np.ndarray, "n_points dim"]) -> Float[np.ndarray, "n_points two_dim"]:
    """`sign_parts`, L1-normalised, so a row says where its contribution comes from. An all-zero row stays all zero."""
    return normalize(sign_parts(points), norm="l1")


def compared_parts(points: Float[np.ndarray, "n_points dim"], normalise: bool) -> Float[np.ndarray, "n_points two_dim"]:
    """Without normalising, a hit's size counts too: x and 3x are no longer the same."""
    return contribution_shares(points) if normalise else sign_parts(points)


def weighted_jaccard_block(
    block: Float[np.ndarray, "block_size two_dim"], shares: Float[np.ndarray, "n_points two_dim"]
) -> Float[np.ndarray, "block_size n_points"]:
    """Distances from each row of `block` to every row of `shares`, both already `compared_parts`.

    - It needs only the L1 distance between rows, since for any two rows a and b:
      - sum(min) = (sum(a) + sum(b) - sum(|a - b|)) / 2
      - sum(max) = (sum(a) + sum(b) + sum(|a - b|)) / 2
    - `cdist` computes that in C without the (block_size, n_points, two_dim) arrays a min and a max would build.
      - On 14k hits of 1056 dims, a pass over all pairs took 36 minutes with those arrays, and 2 with this.
      - The results agree to float rounding."""
    l1 = cdist(block, shares, "cityblock")
    total = block.sum(axis=1)[:, None] + shares.sum(axis=1)[None]
    overlap = (total - l1) / 2
    union = (total + l1) / 2
    # two all-zero rows have no union: dividing by 1 instead gives distance 1
    return 1.0 - overlap / np.where(union > 0, union, 1.0)


def weighted_jaccard_distances(
    points: Float[np.ndarray, "n_points dim"], chunk: int = 256, normalise: bool = True
) -> Float[np.ndarray, "n_points n_points"]:
    """1 - sum(min) / sum(max) between the rows' `compared_parts`, also called the Ruzicka distance.

    - It is plain Jaccard with each dim partly in a set, so a dim counts in proportion to its size.
    - It is a metric.
    - Pairs are built a chunk of rows at a time, since a full n x n x dim array would be large.
    - The math, line by line, with the chunking and a worked example: https://github.com/narang99/in-tfm/pull/91"""
    shares = compared_parts(points, normalise)
    distances = np.empty((len(shares), len(shares)))
    for start in range(0, len(shares), chunk):
        distances[start:start + chunk] = weighted_jaccard_block(shares[start:start + chunk], shares)
    np.fill_diagonal(distances, 0.0)
    return distances
