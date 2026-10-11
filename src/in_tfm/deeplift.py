"""DeepLift for networks of linear ops, ReLUs and max pools, complete by construction.

Why not captum's DeepLift:

- Its MaxPool2d rule spreads an output's contribution onto the winning input with `max_unpool`.
  - An input that wins several overlapping windows (the 3x3 stride-1 pool in every Inception block)
    keeps one contribution and loses the rest.
  - Replacing the rule with a scatter-add fixed a lone pool.
  - Attributions were still incomplete when a pool's output fed one branch that contained another pool
    and a second branch that did not.
  - That only reproduced under captum's backward hooks, so it was not worth chasing further.

How this works:

- Run the baseline once and keep the input and output of every ReLU and max pool.
- Run the input again, and replace each of those outputs with `y0 + m * (x - x0)`.
  - `y0` and `x0` are the baseline's values, and `m` is the multiplier, held constant.
  - The value of the output is unchanged, so the forward pass is the real one.
- Everything else is linear, so the whole network is now affine in the input with fixed multipliers.
  - It passes through the baseline output at the baseline and the real output at the input.
  - So `grad * (x - x0)` sums to exactly the change in the selected activation.
- A ReLU's multiplier is `(y - y0) / (x - x0)`, the Rescale rule.
  - Where `x == x0` it is the ReLU's own slope, which carries no attribution anyway.
- A max pool's change is split in two, as in the DeepLift paper's treatment of max.
  - The part by which the input's pooled value exceeds the baseline's goes to the input's winner.
  - The part by which it falls short goes to the baseline's winner.
  - Each is divided by that winner's own change, which is never zero when its part is not.
  - Gathering by winner index adds up the contributions when windows overlap.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import nn

Selector = Callable[[torch.Tensor], torch.Tensor]
"""Picks one scalar per sample out of the explained layer's output."""

EPS = 1e-10


def relu_multiplier(x: torch.Tensor, x0: torch.Tensor, y: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
    unchanged = (x - x0).abs() < EPS
    return torch.where(unchanged, (x > 0).to(x.dtype), (y - y0) / torch.where(unchanged, torch.ones_like(x), x - x0))


def divide_where_nonzero(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    safe = torch.where(denominator.abs() < EPS, torch.ones_like(denominator), denominator)
    return torch.where(denominator.abs() < EPS, torch.zeros_like(numerator), numerator / safe)


def pooled_with_winners(pool: nn.MaxPool2d, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return F.max_pool2d(
        x, pool.kernel_size, pool.stride, pool.padding, pool.dilation, pool.ceil_mode, return_indices=True
    )


def at_winners(x: torch.Tensor, winners: torch.Tensor) -> torch.Tensor:
    return x.flatten(2).gather(2, winners.flatten(2)).view_as(winners)


def relu_surrogate(x: torch.Tensor, x0: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
    y = F.relu(x)
    return y0 + relu_multiplier(x, x0, y, y0).detach() * (x - x0)


def maxpool_surrogate(pool: nn.MaxPool2d, x: torch.Tensor, x0: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
    y, winners = pooled_with_winners(pool, x)
    _, winners0 = pooled_with_winners(pool, x0)
    cross_max = torch.max(y, y0)
    delta_at_winner = at_winners(x, winners) - at_winners(x0, winners)
    delta_at_winner0 = at_winners(x, winners0) - at_winners(x0, winners0)
    up = divide_where_nonzero(cross_max - y0, delta_at_winner.detach())
    down = divide_where_nonzero(y - cross_max, delta_at_winner0.detach())
    return y0 + up.detach() * delta_at_winner + down.detach() * delta_at_winner0


def deeplift_attribution(
    model: nn.Module,
    x: Float[torch.Tensor, "batch ..."],
    layer: nn.Module,
    selector: Selector,
    baseline: Float[torch.Tensor, "batch ..."] | None = None,
) -> Float[torch.Tensor, "batch ..."]:
    """Attribution of `selector(layer output)` to every element of `x`, summing to its change from the baseline."""
    baseline = torch.zeros_like(x) if baseline is None else baseline
    nonlinear = [m for m in model.modules() if isinstance(m, (nn.ReLU, nn.MaxPool2d))]
    reference: dict[nn.Module, tuple[torch.Tensor, torch.Tensor]] = {}

    def record(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        reference[module] = (inputs[0].detach(), output.detach())

    def replace(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
        x0, y0 = reference[module]
        if isinstance(module, nn.ReLU):
            return relu_surrogate(inputs[0], x0, y0)
        return maxpool_surrogate(module, inputs[0], x0, y0)

    handles = [m.register_forward_hook(record) for m in nonlinear]
    try:
        with torch.no_grad():
            model(baseline)
        for handle in handles:
            handle.remove()
        handles = [m.register_forward_hook(replace) for m in nonlinear]
        explained: list[torch.Tensor] = []
        handles.append(layer.register_forward_hook(lambda m, i, o: explained.append(selector(o))))
        with torch.enable_grad():
            leaf = x.detach().clone().requires_grad_(True)
            model(leaf)
            (grad,) = torch.autograd.grad(explained[-1].sum(), leaf)
    finally:
        for handle in handles:
            handle.remove()
    return (grad * (x - baseline)).detach()
