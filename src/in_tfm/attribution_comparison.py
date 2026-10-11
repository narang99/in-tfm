"""Compares our DeepLift with captum's Integrated Gradients on the same scalar.

- Both explain one conv neuron at one output position, as a map over the input pixels.
- The position is each image's own peak for that neuron, since that is where the reports look.
- Both use a zeros baseline, so both sum to the same change in the activation.
- Integrated Gradients converges to that change as steps grow, DeepLift is exact after two passes.
  - So the question is how many IG steps it takes to agree with DeepLift, and what that costs.
"""

import time
from collections.abc import Callable
from pathlib import Path

import torch
from jaxtyping import Float
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from .deeplift import deeplift_attribution
from .device import default_device
from .integrated_gradients import Attribution, Images, Positions, explained_activation, integrated_gradients, pick_position


class ComparisonConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", cli_kebab_case=True)

    checkpoint: Path | None = Path("checkpoints/stl_inception/final.pt")
    dataset: str = "tanganke/stl10"
    split: str = "test"
    layer_name: str = "block_b.branch_3x3.1.0"
    neuron: int = 2
    n_images: int = 32
    batch_size: int = 16
    ig_steps: list[int] = [8, 32, 128, 512]
    top_k: int = 100
    internal_batch_size: int = 256
    device: str = default_device()
    out_path: Path | None = None


class StepResult(BaseModel):
    ig_steps: int
    ig_seconds: float
    deeplift_seconds: float
    pearson: float
    top_k_overlap: float
    ig_completeness_gap: float
    deeplift_completeness_gap: float


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def timed(fn: Callable[[], Attribution], device: torch.device) -> tuple[Attribution, float]:
    sync(device)
    start = time.perf_counter()
    result = fn()
    sync(device)
    return result, time.perf_counter() - start


@torch.no_grad()
def peak_positions(model: torch.nn.Module, layer: torch.nn.Module, images: Images, neuron: int) -> Positions:
    seen: list[torch.Tensor] = []
    handle = layer.register_forward_hook(lambda m, i, o: seen.append(o[:, neuron].flatten(1).argmax(dim=1)))
    try:
        model(images)
    finally:
        handle.remove()
    return seen[0]


def activation_change(forward: Callable[[Images, Positions], torch.Tensor], images: Images, positions: Positions) -> torch.Tensor:
    with torch.no_grad():
        return forward(images, positions) - forward(torch.zeros_like(images), positions)


def completeness_gap(attribution: Attribution, change: Float[torch.Tensor, "batch"]) -> float:
    """Mean over images of |sum of attribution - activation change|, relative to the change."""
    total = attribution.flatten(1).sum(dim=1)
    return ((total - change).abs() / change.abs().clamp_min(1e-8)).mean().item()


def pearson(a: Attribution, b: Attribution) -> float:
    a, b = a.flatten(1), b.flatten(1)
    a, b = a - a.mean(dim=1, keepdim=True), b - b.mean(dim=1, keepdim=True)
    return ((a * b).sum(dim=1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-12)).mean().item()


def top_k_overlap(a: Attribution, b: Attribution, k: int) -> float:
    """Fraction of the k most relevant pixels, by absolute value summed over channels, that both pick."""
    top = lambda t: t.abs().sum(dim=1).flatten(1).topk(k, dim=1).indices
    ranked_a, ranked_b = top(a), top(b)
    shared = (ranked_a.unsqueeze(2) == ranked_b.unsqueeze(1)).any(dim=2).sum(dim=1)
    return (shared.float() / k).mean().item()


def deeplift(
    model: torch.nn.Module, layer: torch.nn.Module, images: Images, positions: Positions, neuron: int
) -> Attribution:
    return deeplift_attribution(model, images, layer, lambda out: pick_position(out, neuron, positions))


def measure_batch(
    model: torch.nn.Module, layer: torch.nn.Module, images: Images, config: ComparisonConfig
) -> list[StepResult]:
    positions = peak_positions(model, layer, images, config.neuron)
    forward = explained_activation(model, layer, config.neuron)
    change = activation_change(forward, images, positions)
    reference, deeplift_seconds = timed(lambda: deeplift(model, layer, images, positions, config.neuron), images.device)
    results = []
    for n_steps in config.ig_steps:
        run = lambda: integrated_gradients(forward, images, positions, n_steps, config.internal_batch_size)
        ig, ig_seconds = timed(run, images.device)
        results.append(
            StepResult(
                ig_steps=n_steps,
                ig_seconds=ig_seconds,
                deeplift_seconds=deeplift_seconds,
                pearson=pearson(ig, reference),
                top_k_overlap=top_k_overlap(ig, reference, config.top_k),
                ig_completeness_gap=completeness_gap(ig, change),
                deeplift_completeness_gap=completeness_gap(reference, change),
            )
        )
    return results


def merge(per_batch: list[list[StepResult]], sizes: list[int]) -> list[StepResult]:
    """Times add up over batches, and the quality numbers are averaged by how many images each batch had."""
    total = sum(sizes)
    merged = []
    for steps in zip(*per_batch):
        weighted = lambda field: sum(getattr(r, field) * n for r, n in zip(steps, sizes)) / total
        merged.append(
            StepResult(
                ig_steps=steps[0].ig_steps,
                ig_seconds=sum(r.ig_seconds for r in steps),
                deeplift_seconds=sum(r.deeplift_seconds for r in steps),
                pearson=weighted("pearson"),
                top_k_overlap=weighted("top_k_overlap"),
                ig_completeness_gap=weighted("ig_completeness_gap"),
                deeplift_completeness_gap=weighted("deeplift_completeness_gap"),
            )
        )
    return merged
