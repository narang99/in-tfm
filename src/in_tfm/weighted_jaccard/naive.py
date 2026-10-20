"""Weighted Jaccard with plain loops, one row and one pair at a time. Mirrors `vectorized.py`."""

import numpy as np
from jaxtyping import Float


def contribution_shares(points: Float[np.ndarray, "n_points dim"]) -> Float[np.ndarray, "n_points two_dim"]:
    shares = []
    for row in points.tolist():
        parts = [max(x, 0.0) for x in row] + [max(-x, 0.0) for x in row]
        total = sum(parts)
        shares.append([part / total if total > 0 else 0.0 for part in parts])
    return np.array(shares, dtype=np.float64).reshape(len(points), 2 * points.shape[1])


def _pair_distance(a: list[float], b: list[float]) -> float:
    overlap = sum(min(x, y) for x, y in zip(a, b))
    union = sum(max(x, y) for x, y in zip(a, b))
    return 1.0 - overlap / union if union > 0 else 1.0


def weighted_jaccard_block(
    block: Float[np.ndarray, "block_size two_dim"], shares: Float[np.ndarray, "n_points two_dim"]
) -> Float[np.ndarray, "block_size n_points"]:
    rows = [[_pair_distance(a, b) for b in shares.tolist()] for a in block.tolist()]
    return np.array(rows, dtype=np.float64).reshape(len(block), len(shares))


def weighted_jaccard_distances(points: Float[np.ndarray, "n_points dim"]) -> Float[np.ndarray, "n_points n_points"]:
    shares = contribution_shares(points)
    distances = weighted_jaccard_block(shares, shares)
    for i in range(len(distances)):
        distances[i, i] = 0.0
    return distances
