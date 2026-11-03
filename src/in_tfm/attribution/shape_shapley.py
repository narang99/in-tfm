"""How much each source builds the shape of a summed row, as Shapley values on weighted Jaccard similarity.

- A row is a sum of source parts, r = sum_s c_s. For one attention head that is A[t, s] (v_s * w).
- The sum of a source's part says what it adds to the activation. It says nothing about the row's shape.
- So the score of a set S of sources is how close their partial sum is to the full row:
  - v(S) = 1 - weighted_jaccard(sum of c_s over S, r). v of no sources is 0, and v of all sources is 1.
  - The Shapley values so add up to exactly 1, a share of the shape per source.
- Shapley, not the effect of removing one source, since two similar sources each look unneeded when removed alone.
- Exact Shapley needs every subset, so only the `top` largest sources by L1 are players. The rest join as one player.
"""

from math import factorial

import numpy as np
from jaxtyping import Float, Int
from pydantic import BaseModel

from ..weighted_jaccard import sign_parts, weighted_jaccard_block


class ShapeShares(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    sources: Int[np.ndarray, "players"]
    """Source indices of the named players, largest L1 first."""
    shares: Float[np.ndarray, "players"]
    rest: float
    """The share of every other source, taken as one player."""


def shape_similarity(
    rows: Float[np.ndarray, "m dim"], target: Float[np.ndarray, "dim"]
) -> Float[np.ndarray, "m"]:
    return (
        1.0 - weighted_jaccard_block(sign_parts(rows), sign_parts(target[None]))[:, 0]
    )


def shape_shapley(parts: Float[np.ndarray, "sources dim"], top: int = 6) -> ShapeShares:
    order = np.argsort(-np.abs(parts).sum(axis=1), kind="stable")
    named = order[:top]
    players = np.vstack([parts[named], parts[order[top:]].sum(axis=0, keepdims=True)])
    shares = exact_shapley(players, parts.sum(axis=0))
    return ShapeShares(sources=named, shares=shares[:-1], rest=float(shares[-1]))


def exact_shapley(
    players: Float[np.ndarray, "n dim"], target: Float[np.ndarray, "dim"]
) -> Float[np.ndarray, "n"]:
    n = len(players)
    subsets = np.arange(2**n)
    members = (subsets[:, None] >> np.arange(n)) & 1
    value = shape_similarity(members.astype(np.float64) @ players, target)
    size = members.sum(axis=1)
    weight = np.array(
        [
            factorial(k) * factorial(n - k - 1) / factorial(n) if k < n else 0.0
            for k in size
        ]
    )
    shares = np.zeros(n)
    for i in range(n):
        without = subsets[members[:, i] == 0]
        shares[i] = (
            weight[without] * (value[without | (1 << i)] - value[without])
        ).sum()
    return shares
