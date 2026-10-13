"""Edges of the circuit: how much of an MLP neuron's input comes from each attention head, and where each head attends."""

import numpy as np
import torch
from jaxtyping import Float

from .models.grokking.data import all_pairs
from .models.grokking.model import GrokTransformer


def head_contributions_to_mlp_neurons(model: GrokTransformer) -> Float[np.ndarray, "pair head neuron"]:
    """Each head's share of every MLP pre-activation at `=`, using that `o_proj` is linear and the residual is a sum."""
    tokens, _ = all_pairs()
    layer = model.layers[0]
    mixed: list[torch.Tensor] = []
    handle = layer.self_attn.mixed.register_forward_hook(lambda _m, _i, out: mixed.append(out[:, -1].detach()))
    with torch.no_grad():
        model.logits_for_tokens(tokens)
    handle.remove()
    n_heads, d_head = model.config.n_heads, model.config.d_head
    per_head = mixed[0].reshape(-1, n_heads, d_head)
    o_weight = layer.self_attn.o_proj.weight.detach().reshape(-1, n_heads, d_head)
    up = layer.mlp.up_proj.weight.detach()
    head_out = torch.einsum("phd,mhd->phm", per_head, o_weight)
    return torch.einsum("phm,nm->phn", head_out, up).numpy()


def variance_shares(contributions: Float[np.ndarray, "pair head neuron"], pre_activation: Float[np.ndarray, "pair neuron"]) -> Float[np.ndarray, "head neuron"]:
    """Covariance of a head's contribution with the neuron over the neuron's variance, which sums to one over heads and the constant paths."""
    centered_pre = pre_activation - pre_activation.mean(axis=0)
    centered = contributions - contributions.mean(axis=0)
    return np.einsum("phn,pn->hn", centered, centered_pre) / (centered_pre**2).sum(axis=0)


def attention_to_operands(model: GrokTransformer) -> Float[np.ndarray, "head position"]:
    """Mean attention weight from `=` to each of the three positions, for every head."""
    tokens, _ = all_pairs()
    layer = model.layers[0].self_attn
    captured: dict[str, torch.Tensor] = {}
    handles = [layer.q_proj.register_forward_hook(lambda _m, _i, o: captured.__setitem__("q", o.detach())),
               layer.k_proj.register_forward_hook(lambda _m, _i, o: captured.__setitem__("k", o.detach()))]
    with torch.no_grad():
        model.logits_for_tokens(tokens)
    for handle in handles:
        handle.remove()
    n_heads, d_head = model.config.n_heads, model.config.d_head
    q = captured["q"][:, -1].reshape(-1, n_heads, d_head)
    k = captured["k"].reshape(len(tokens), -1, n_heads, d_head)
    scores = torch.einsum("phd,pshd->phs", q, k) * layer.scaling
    return torch.softmax(scores, dim=-1).mean(dim=0).numpy()
