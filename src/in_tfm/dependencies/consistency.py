"""A map is only trusted if its terms rebuild the target's input exactly.

- A wrong map, like a channel-major flatten read as cell-major, still gives plausible numbers.
- So every collected example is checked: each dependency's terms must sum to the input dim it claims.
"""

import torch
from jaxtyping import Float

from .refs import Dependency


def check_terms_rebuild_input(
    dependencies: list[Dependency],
    target_input: Float[torch.Tensor, "hidden"],
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> float:
    """Returns the largest absolute disagreement, so a run can log how tight it was."""
    rebuilt = torch.tensor([d.total for d in dependencies], dtype=torch.float64)
    expected = target_input[[d.dim for d in dependencies]].double().cpu()
    worst = (rebuilt - expected).abs().max().item() if dependencies else 0.0
    if not torch.allclose(rebuilt, expected, atol=atol, rtol=rtol):
        raise ValueError(
            f"a dependency's terms do not sum to the target input dim it maps "
            f"(max abs diff {worst:.3g}); the dependency map is wrong for this model"
        )
    return worst
