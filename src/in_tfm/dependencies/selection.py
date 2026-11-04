"""Which of many terms explain most of a sum, ported from hp-do's `indices_for_percentage`."""

import numpy as np
from jaxtyping import Float, Int


def covering_indices(
    values: Float[np.ndarray, "n"], mass: float
) -> Int[np.ndarray, "k"]:
    """The fewest entries, largest |value| first, whose |values| reach `mass` of the total |value|.

    - Both signs are ranked together, not each topped up to `mass` on its own.
      - So a sign that adds little to the total magnitude keeps few or no entries.
    """
    magnitudes = np.abs(values)
    order = np.argsort(-magnitudes, kind="stable")
    covered = np.cumsum(magnitudes[order])
    if covered[-1] == 0:
        return order[:0]
    n_kept = min(int(np.searchsorted(covered, mass * covered[-1])) + 1, len(order))
    return order[:n_kept]
