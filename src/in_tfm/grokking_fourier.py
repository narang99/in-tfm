"""The paper's Fourier view of the modular addition network, used as ground truth for the discovered circuits."""

import numpy as np
import torch
from jaxtyping import Float
from pydantic import BaseModel

from .models.grokking.model import EQUALS_TOKEN, MODULUS, GrokTransformer

N_FREQS = MODULUS // 2
"""Frequencies 1..56, since k and P - k are the same real Fourier component."""

PairGrid = Float[np.ndarray, "a b"]


def fourier_basis() -> Float[np.ndarray, "freq residue"]:
    """Row 0 is the constant, then cos and sin of each frequency, orthonormal over the residues."""
    x = np.arange(MODULUS)
    rows = [np.ones(MODULUS) / np.sqrt(MODULUS)]
    for k in range(1, N_FREQS + 1):
        rows += [np.cos(2 * np.pi * k * x / MODULUS), np.sin(2 * np.pi * k * x / MODULUS)]
    basis = np.stack(rows)
    basis[1:] *= np.sqrt(2 / MODULUS)
    return basis


def frequency_energy(signal: Float[np.ndarray, "residue ..."]) -> Float[np.ndarray, "freq ..."]:
    """Energy per frequency k = 1..56 along the first axis, cos and sin summed."""
    coeffs = np.tensordot(fourier_basis()[1:], signal, axes=(1, 0))
    return (coeffs**2).reshape(N_FREQS, 2, *signal.shape[1:]).sum(axis=1)


def key_frequencies(model: GrokTransformer, top: int = 6) -> list[int]:
    """The frequencies carrying most of the token embedding, as in the paper's Figure 2."""
    embed = model.embed_tokens.weight[:MODULUS].detach().numpy()
    energy = frequency_energy(embed).sum(axis=1)
    return [int(k) + 1 for k in np.argsort(-energy)[:top]]


class NeuronFourier(BaseModel):
    dominant_freq: int
    dominant_share: float
    """Share of the neuron's (a, b) variance in terms where `a` or `b` has the dominant frequency."""


def mlp_pre_activation_grid(model: GrokTransformer) -> Float[np.ndarray, "a b neuron"]:
    """The `=` position of every pair, read at the MLP's pre-activation, which is what `up_proj` neurons are."""
    a = torch.arange(MODULUS).repeat_interleave(MODULUS)
    b = torch.arange(MODULUS).repeat(MODULUS)
    tokens = torch.stack([a, b, torch.full_like(a, EQUALS_TOKEN)], dim=1)
    captured: list[torch.Tensor] = []
    handle = model.layers[0].mlp.up_proj.register_forward_hook(lambda _m, _i, out: captured.append(out[:, -1].detach()))
    with torch.no_grad():
        model.logits_for_tokens(tokens)
    handle.remove()
    return captured[0].reshape(MODULUS, MODULUS, -1).numpy()


def pair_frequency_energy(grid: PairGrid) -> Float[np.ndarray, "freq_a freq_b"]:
    """Energy at each (frequency of `a`, frequency of `b`), where index 0 is "constant along that axis" and the mean is removed."""
    basis = fourier_basis()
    coeffs = basis @ grid @ basis.T
    coeffs[0, 0] = 0.0
    energy = coeffs**2
    pooled = energy[1:, 1:].reshape(N_FREQS, 2, N_FREQS, 2).sum(axis=(1, 3))
    a_only = energy[1:, 0].reshape(N_FREQS, 2).sum(axis=1)
    b_only = energy[0, 1:].reshape(N_FREQS, 2).sum(axis=1)
    out = np.zeros((N_FREQS + 1, N_FREQS + 1))
    out[1:, 1:], out[1:, 0], out[0, 1:] = pooled, a_only, b_only
    return out


def neuron_fourier(grid: PairGrid) -> NeuronFourier:
    """The paper's neurons are sums and products of cos/sin(k a) and cos/sin(k b), so a frequency counts when either input has it."""
    energy = pair_frequency_energy(grid)
    involving = energy[1:].sum(axis=1) + energy[:, 1:].sum(axis=0) - np.diag(energy)[1:]
    k = int(involving.argmax())
    return NeuronFourier(dominant_freq=k + 1, dominant_share=float(involving[k] / energy.sum()))


def mlp_frequency_spectra(model: GrokTransformer) -> Float[np.ndarray, "neuron freq"]:
    """Per MLP neuron, the energy of its (a, b) map in terms where `a` or `b` has frequency k = 1..56."""
    grid = mlp_pre_activation_grid(model)
    spectra = []
    for n in range(grid.shape[-1]):
        energy = pair_frequency_energy(grid[..., n])
        spectra.append(energy[1:].sum(axis=1) + energy[:, 1:].sum(axis=0) - np.diag(energy)[1:])
    return np.stack(spectra)


def attention_frequency_spectra(model: GrokTransformer, target: str) -> Float[np.ndarray, "coord freq"]:
    """Per `v_proj` or `k_proj` coordinate, the energy of its value over the residues, summed over the two operand positions."""
    layer = model.layers[0].self_attn
    weight = getattr(layer, target).weight.detach().numpy()
    embed = model.embed_tokens.weight[:MODULUS].detach().numpy()
    pos = model.pos_embed.detach().numpy()
    spectra = [
        sum(frequency_energy(((embed + pos[p]) @ weight[c])[:, None])[:, 0] for p in (0, 1))
        for c in range(weight.shape[0])
    ]
    return np.stack(spectra)
