"""A neuron's activation split over SAE latents, set against our clusters of its hits.

- A q_proj neuron computes w · norm(x), with x the residual stream.
  - Gemma's RMSNorm is x / rms(x) * (1 + γ), so this is (w ⊙ (1 + γ)) · x / rms(x).
- The SAE writes x = Σ_j f_j d_j + b_dec + error, which splits the activation exactly into
  - one term per active latent, f_j (w_eff · d_j) / rms(x),
  - a bias term and an error term.
- The Hadamard product splits the same dot product per residual coordinate instead.
  - So both views explain the same number, in two different bases.
"""

from collections import Counter

import numpy as np
import torch
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict
from sae_lens import SAE
from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure
from sklearn.preprocessing import normalize

NOISE = -1


class LatentSplit(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    activation: Float[np.ndarray, "n_hits"]
    """Recomputed from the residual, so it can be checked against the captured activation."""
    top_latents: Int[np.ndarray, "n_hits k"]
    """Largest contribution first."""
    top_contributions: Float[np.ndarray, "n_hits k"]
    latent_total: Float[np.ndarray, "n_hits"]
    """Over every active latent, not only the top k."""
    positive_total: Float[np.ndarray, "n_hits"]
    bias_term: Float[np.ndarray, "n_hits"]
    error_term: Float[np.ndarray, "n_hits"]
    read_strength: Float[np.ndarray, "width"]
    """w_eff · d_j: how much the neuron responds to one unit of latent j, before 1 / rms(x)."""


class SaeCheck(BaseModel):
    """On random tokens, to show the SAE is loaded and fed the activation it was trained on."""

    mean_l0: float
    fraction_of_variance_unexplained: float


class ClusterRow(BaseModel):
    cluster: int
    n_hits: int
    top_tokens: list[str]
    dominant_latent: int
    """The latent that most often contributes most to this cluster's hits."""
    dominant_share: float
    """Fraction of the cluster's hits whose top latent it is."""
    precision: float
    """Of all the neuron's hits topped by that latent, the fraction in this cluster."""
    median_error_share: float


class LatentRow(BaseModel):
    latent: int
    n_hits_topped: int
    clusters_spanned: int
    """Clusters holding at least `min_hits_per_cluster` of the hits it tops."""
    noise_share: float
    """Fraction of the hits it tops that HDBSCAN called noise."""
    top_tokens: list[str]
    read_rank: int
    """Rank of its read strength among all latents, 0 being the strongest."""


class NeuronComparison(BaseModel):
    neuron: int
    n_hits: int
    n_clusters: int
    noise_fraction: float
    max_activation_mismatch: float
    median_latent_share: float
    median_bias_share: float
    median_error_share: float
    median_top_latent_share: float
    """Top latent's contribution over the sum of positive latent contributions."""
    median_latents_for_80_percent: float
    distinct_top_latents: int
    n_sae_clusters: int
    """HDBSCAN on the hits' latent contributions, the SAE-basis twin of our clustering."""
    sae_noise_fraction: float
    adjusted_rand: dict[str, float]
    """Between labelings of the same hits, each pair over the hits neither calls noise."""
    homogeneity: float
    """1 when each latent's hits sit in one cluster: latents are no coarser than clusters."""
    completeness: float
    """1 when each cluster's hits share one top latent: clusters are no coarser than latents."""
    clusters: list[ClusterRow]
    latents: list[LatentRow]


def check_sae(sae: SAE, resid: Float[torch.Tensor, "n d_model"]) -> SaeCheck:
    with torch.no_grad():
        latents = sae.encode(resid)
        error = resid - sae.decode(latents)
    total_variance = (resid - resid.mean(0)).pow(2).sum()
    return SaeCheck(
        mean_l0=(latents > 0).sum(1).float().mean().item(),
        fraction_of_variance_unexplained=(error.pow(2).sum() / total_variance).item(),
    )


def split_over_latents(
    resid: Float[torch.Tensor, "n_hits d_model"],
    weight_row: Float[torch.Tensor, "d_model"],
    norm: torch.nn.Module,
    sae: SAE,
    k: int = 32,
    chunk: int = 2048,
) -> LatentSplit:
    w_eff = weight_row.float() * (1 + norm.weight.float())
    read_strength = sae.W_dec @ w_eff
    parts = [_split_chunk(resid[i : i + chunk], w_eff, read_strength, norm.eps, sae, k) for i in range(0, len(resid), chunk)]
    joined = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    return LatentSplit(**joined, read_strength=read_strength.cpu().numpy())


def _split_chunk(
    x: Float[torch.Tensor, "n d_model"],
    w_eff: Float[torch.Tensor, "d_model"],
    read_strength: Float[torch.Tensor, "width"],
    eps: float,
    sae: SAE,
    k: int,
) -> dict[str, np.ndarray]:
    with torch.no_grad():
        inv_rms = torch.rsqrt(x.pow(2).mean(-1) + eps)
        latents = sae.encode(x)
        contributions = latents * read_strength * inv_rms[:, None]
        top = contributions.topk(k, dim=1)
        parts = {
            "activation": inv_rms * (x @ w_eff),
            "top_latents": top.indices,
            "top_contributions": top.values,
            "latent_total": contributions.sum(1),
            "positive_total": contributions.clamp(min=0).sum(1),
            "bias_term": inv_rms * (sae.b_dec @ w_eff),
            "error_term": inv_rms * ((x - sae.decode(latents)) @ w_eff),
        }
    return {key: value.cpu().numpy() for key, value in parts.items()}


def latents_for_share(split: LatentSplit, share: float) -> Int[np.ndarray, "n_hits"]:
    """How many top latents it takes to reach `share` of the positive latent total."""
    cumulative = np.cumsum(np.clip(split.top_contributions, 0, None), axis=1)
    reached = cumulative >= share * split.positive_total[:, None]
    return reached.argmax(1) + 1


def latent_contribution_vectors(split: LatentSplit) -> Float[np.ndarray, "n_hits n_latents"]:
    """Each hit's contributions over every latent that is among some hit's top k, L2-normalised
    as the Hadamard products are, so HDBSCAN sees the same kind of points in the other basis."""
    used, columns = np.unique(split.top_latents, return_inverse=True)
    vectors = np.zeros((len(split.top_latents), len(used)), dtype=np.float32)
    np.put_along_axis(vectors, columns.reshape(split.top_latents.shape), split.top_contributions, axis=1)
    return normalize(vectors, "l2")


def rand_over_clustered(a: Int[np.ndarray, "n"], b: Int[np.ndarray, "n"]) -> float:
    """Latent ids and token ids are never noise, so only our labels and the SAE clusters drop hits."""
    kept = (a != NOISE) & (b != NOISE)
    return float(adjusted_rand_score(a[kept], b[kept])) if kept.any() else float("nan")


def common_tokens(tokens: list[str], n: int = 4) -> list[str]:
    return [token for token, _ in Counter(tokens).most_common(n)]


def compare_neuron(
    neuron: int,
    labels: Int[np.ndarray, "n_hits"],
    split: LatentSplit,
    captured_activation: Float[np.ndarray, "n_hits"],
    sae_labels: Int[np.ndarray, "n_hits"],
    tokens: list[str],
    n_rows: int = 20,
    min_hits_per_cluster: int = 5,
) -> NeuronComparison:
    top_latent = split.top_latents[:, 0]
    clustered = labels != NOISE
    homogeneity, completeness, _ = (
        homogeneity_completeness_v_measure(labels[clustered], top_latent[clustered]) if clustered.any() else (np.nan,) * 3
    )
    token_labels = np.unique(tokens, return_inverse=True)[1]
    return NeuronComparison(
        neuron=neuron,
        n_hits=len(labels),
        n_clusters=len(set(labels) - {NOISE}),
        noise_fraction=float((~clustered).mean()),
        max_activation_mismatch=float(np.abs(split.activation - captured_activation).max()),
        median_latent_share=float(np.median(split.latent_total / split.activation)),
        median_bias_share=float(np.median(split.bias_term / split.activation)),
        median_error_share=float(np.median(split.error_term / split.activation)),
        median_top_latent_share=float(np.nanmedian(split.top_contributions[:, 0] / split.positive_total)),
        median_latents_for_80_percent=float(np.median(latents_for_share(split, 0.8))),
        distinct_top_latents=len(set(top_latent)),
        n_sae_clusters=len(set(sae_labels) - {NOISE}),
        sae_noise_fraction=float((sae_labels == NOISE).mean()),
        adjusted_rand={
            "ours_vs_sae_clusters": rand_over_clustered(labels, sae_labels),
            "ours_vs_top_latent": rand_over_clustered(labels, top_latent),
            "ours_vs_token": rand_over_clustered(labels, token_labels),
            "sae_clusters_vs_token": rand_over_clustered(sae_labels, token_labels),
            "top_latent_vs_token": rand_over_clustered(top_latent, token_labels),
        },
        homogeneity=float(homogeneity),
        completeness=float(completeness),
        clusters=cluster_rows(labels, split, top_latent, tokens, n_rows),
        latents=latent_rows(labels, split, top_latent, tokens, n_rows, min_hits_per_cluster),
    )


def cluster_rows(
    labels: Int[np.ndarray, "n_hits"],
    split: LatentSplit,
    top_latent: Int[np.ndarray, "n_hits"],
    tokens: list[str],
    n_rows: int,
) -> list[ClusterRow]:
    largest = [c for c, _ in Counter(labels[labels != NOISE].tolist()).most_common(n_rows)]
    rows = []
    for cluster in largest:
        members = labels == cluster
        dominant, count = Counter(top_latent[members].tolist()).most_common(1)[0]
        rows.append(
            ClusterRow(
                cluster=cluster,
                n_hits=int(members.sum()),
                top_tokens=common_tokens([t for t, m in zip(tokens, members) if m]),
                dominant_latent=dominant,
                dominant_share=count / members.sum(),
                precision=count / (top_latent == dominant).sum(),
                median_error_share=float(np.median(split.error_term[members] / split.activation[members])),
            )
        )
    return rows


def latent_rows(
    labels: Int[np.ndarray, "n_hits"],
    split: LatentSplit,
    top_latent: Int[np.ndarray, "n_hits"],
    tokens: list[str],
    n_rows: int,
    min_hits_per_cluster: int,
) -> list[LatentRow]:
    read_rank = np.empty(len(split.read_strength), dtype=int)
    read_rank[np.argsort(-split.read_strength)] = np.arange(len(split.read_strength))
    rows = []
    for latent, n_topped in Counter(top_latent.tolist()).most_common(n_rows):
        topped = top_latent == latent
        per_cluster = Counter(labels[topped & (labels != NOISE)].tolist())
        rows.append(
            LatentRow(
                latent=latent,
                n_hits_topped=n_topped,
                clusters_spanned=sum(n >= min_hits_per_cluster for n in per_cluster.values()),
                noise_share=float((labels[topped] == NOISE).mean()),
                top_tokens=common_tokens([t for t, m in zip(tokens, topped) if m]),
                read_rank=int(read_rank[latent]),
            )
        )
    return rows
