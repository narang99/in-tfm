"""Neuron groups found by clustering spectra, and the checks that a group is a real part of the circuit."""

import numpy as np
import torch
from jaxtyping import Float, Int
from pydantic import BaseModel

from .clustering import cluster_labels
from .grokking_circuits import hellinger_spectra
from .grokking_fourier import attention_frequency_spectra, key_frequencies, mlp_frequency_spectra
from .models.grokking.data import all_pairs
from .models.grokking.model import GrokTransformer

GROUP_MIN_SIZE = 10
"""`eom` with this size gave frequency-pure groups of the final network, where `leaf` split them into dozens of fragments."""

Labels = Int[np.ndarray, "n_neurons"]


class GroupSummary(BaseModel):
    size: int
    dominant_freq: int
    purity: float
    """Share of the group's neurons whose own strongest frequency is the group's."""


class LayerGroups(BaseModel):
    target: str
    n_groups: int
    n_noise: int
    groups: list[GroupSummary]
    median_dominant_share: float
    """Share of a neuron's spectrum on its strongest frequency, which is how concentrated the layer is on few frequencies."""


def group_by_spectrum(spectra: Float[np.ndarray, "n freq"]) -> Labels:
    return np.asarray(cluster_labels(hellinger_spectra(spectra), GROUP_MIN_SIZE, "eom"))


def summarize_groups(target: str, spectra: Float[np.ndarray, "n freq"], labels: Labels) -> LayerGroups:
    strongest = spectra.argmax(axis=1) + 1
    groups = []
    for label in sorted(set(labels.tolist()) - {-1}):
        freqs, counts = np.unique(strongest[labels == label], return_counts=True)
        groups.append(GroupSummary(size=int(counts.sum()), dominant_freq=int(freqs[counts.argmax()]), purity=float(counts.max() / counts.sum())))
    shares = spectra.max(axis=1) / spectra.sum(axis=1)
    return LayerGroups(
        target=target, n_groups=len(groups), n_noise=int((labels == -1).sum()), groups=groups,
        median_dominant_share=float(np.median(shares)),
    )


def layer_spectra(model: GrokTransformer) -> dict[str, Float[np.ndarray, "n freq"]]:
    return {
        "up_proj": mlp_frequency_spectra(model),
        "v_proj": attention_frequency_spectra(model, "v_proj"),
        "k_proj": attention_frequency_spectra(model, "k_proj"),
    }


class AblationResult(BaseModel):
    loss: float
    accuracy: float


def loss_with_mlp_neurons_set_to_mean(model: GrokTransformer, neurons: list[int]) -> AblationResult:
    """Mean ablation over all pairs, so the neurons stop carrying information but the layer keeps its usual scale."""
    tokens, labels = all_pairs()
    mlp_act = model.layers[0].mlp.act
    state: dict[str, torch.Tensor] = {}
    def record(_m, _i, out: torch.Tensor) -> None:
        state["mean"] = out[:, -1].mean(dim=0)
    def ablate(_m, _i, out: torch.Tensor) -> torch.Tensor:
        out = out.clone()
        out[:, -1, neurons] = state["mean"][neurons]
        return out
    handle = mlp_act.register_forward_hook(record)
    with torch.no_grad():
        model.logits_for_tokens(tokens)
    handle.remove()
    handle = mlp_act.register_forward_hook(ablate)
    with torch.no_grad():
        logits = model.logits_for_tokens(tokens)[:, -1]
    handle.remove()
    return AblationResult(
        loss=float(torch.nn.functional.cross_entropy(logits, labels)), accuracy=float((logits.argmax(-1) == labels).float().mean())
    )


def key_frequency_set(model: GrokTransformer, min_share: float = 0.05) -> list[int]:
    """Frequencies holding at least `min_share` of the embedding energy, in order of strength."""
    from .grokking_fourier import frequency_energy
    embed = model.embed_tokens.weight[:113].detach().numpy()
    energy = frequency_energy(embed).sum(axis=1)
    order = np.argsort(-energy)
    return [int(k) + 1 for k in order if energy[k] / energy.sum() >= min_share]
