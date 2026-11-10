"""Attention scores of one layer's heads, split across head dimensions, one row per query-key pair.

- The score of query p and key t is `scaling * q[p] · k[t]`, with q and k after q_norm and rope.
  - So `scaling * q[p] ⊙ k[t]` sums exactly to it, as a neuron's Hadamard sums to its activation.
- Two passes, like `neuron_capture`.
  - The scan keeps every causal pair's score, so the elbow is exact.
  - The gather re-runs only the samples holding hits, and keeps the products at those pairs.
- The forward stops inside the target layer's attention, so no later layer runs.
"""

from collections.abc import Sequence
from itertools import batched

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from .hit_selection.strategies.elbow import elbow_pick
from .sources import ModelBatch, SampleId, SampleSource

QueryKey = tuple[
    Float[torch.Tensor, "batch heads seq head_dim"],
    Float[torch.Tensor, "batch kv_heads seq head_dim"],
]


class StopForward(Exception):
    pass


def post_rope_query_key(
    attention: torch.nn.Module,
    hidden: Float[torch.Tensor, "batch seq hidden"],
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> QueryKey:
    """The same steps as `Qwen3Attention.forward` up to the score."""
    head_shape = (*hidden.shape[:-1], -1, attention.head_dim)
    query = attention.q_norm(attention.q_proj(hidden).view(head_shape)).transpose(1, 2)
    key = attention.k_norm(attention.k_proj(hidden).view(head_shape)).transpose(1, 2)
    cos, sin = position_embeddings
    return apply_rotary_pos_emb(query, key, cos, sin)


def attention_of(model: torch.nn.Module, layer: int) -> torch.nn.Module:
    return model.get_submodule(f"layers.{layer}.self_attn")


def capture_query_key(
    model: torch.nn.Module, layer: int, batch: ModelBatch
) -> QueryKey:
    captured: list[QueryKey] = []

    def keep_and_stop(module, args, kwargs):
        captured.append(
            post_rope_query_key(
                module, kwargs["hidden_states"], kwargs["position_embeddings"]
            )
        )
        raise StopForward

    handle = attention_of(model, layer).register_forward_pre_hook(
        keep_and_stop, with_kwargs=True
    )
    try:
        with torch.no_grad():
            model(**batch.kwargs, use_cache=False)
    except StopForward:
        pass
    finally:
        handle.remove()
    return captured[0]


def kv_head_of(attention: torch.nn.Module, head: int) -> int:
    return head // attention.num_key_value_groups


def head_scores(
    query_key: QueryKey, attention: torch.nn.Module, head: int
) -> Float[torch.Tensor, "batch seq seq"]:
    query, key = query_key
    kv_head = kv_head_of(attention, head)
    return attention.scaling * (
        query[:, head].float() @ key[:, kv_head].float().transpose(-1, -2)
    )


def causal_pairs(
    valid_mask: Bool[torch.Tensor, "batch seq"],
) -> Bool[torch.Tensor, "batch seq seq"]:
    """Query on axis 1, key on axis 2. Both must be real tokens, and the key not after the query."""
    seq = valid_mask.shape[1]
    causal = torch.ones(seq, seq, dtype=torch.bool, device=valid_mask.device).tril()
    return valid_mask[:, :, None] & valid_mask[:, None, :] & causal


class PairScan(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sample_ids: list[SampleId]
    heads: list[int]
    scores: Float[torch.Tensor, "batch heads seq seq"]
    """float16, which halves what a large corpus holds in memory. `-inf` where the pair is not causal or touches padding."""


def scan_pairs(
    model: torch.nn.Module,
    source: SampleSource,
    layer: int,
    heads: Sequence[int],
    batch_size: int,
    device: str,
) -> PairScan:
    attention = attention_of(model, layer)
    sample_ids = list(source.sample_ids())
    chunks = []
    for ids in tqdm(list(batched(sample_ids, batch_size)), desc="scan"):
        batch = source.to_model_batch(ids).to(device)
        query_key = capture_query_key(model, layer, batch)
        pairs = causal_pairs(batch.valid_mask)
        scores = torch.stack(
            [head_scores(query_key, attention, head) for head in heads], dim=1
        )
        scores = scores.masked_fill(~pairs[:, None], -torch.inf)
        chunks.append(scores.half().cpu())
    return PairScan(sample_ids=sample_ids, heads=list(heads), scores=torch.cat(chunks))


class PairHits(BaseModel):
    """Hits of one head, as (sample, query, key) index triples into the scan."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    head: int
    cutoff: float
    n_candidates: int
    sample_idx: Int[np.ndarray, "n_hits"]
    query_idx: Int[np.ndarray, "n_hits"]
    key_idx: Int[np.ndarray, "n_hits"]


def positive_sorted_scores(
    scores: Float[torch.Tensor, "batch seq seq"], device: str
) -> Float[torch.Tensor, "n_pairs"]:
    """Ascending, as `selection.positive_sorted_activations` hands the elbow a neuron's tail."""
    values = scores[torch.isfinite(scores) & (scores > 0)].float()
    return torch.sort(values.to(device)).values


def elbow_pairs(
    scan: PairScan, head: int, max_hits: int, seed: int, device: str
) -> PairHits:
    scores = scan.scores[:, scan.heads.index(head)]
    pick = elbow_pick(positive_sorted_scores(scores, device))
    above = torch.argwhere(scores > pick.cutoff).numpy()
    n_candidates = len(above)
    if n_candidates > max_hits:
        keep = np.random.default_rng(seed).choice(n_candidates, max_hits, replace=False)
        above = above[np.sort(keep)]
    return PairHits(
        head=head,
        cutoff=pick.cutoff,
        n_candidates=n_candidates,
        sample_idx=above[:, 0],
        query_idx=above[:, 1],
        key_idx=above[:, 2],
    )


def indices_on(values, like: torch.Tensor) -> Int[torch.Tensor, "n"]:
    return torch.as_tensor(np.asarray(values), dtype=torch.long, device=like.device)


class PairRows(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    hits: PairHits
    rows: Float[torch.Tensor, "n_hits head_dim"]
    """`scaling * q ⊙ k` at each hit. Each row sums to the hit's score."""
    scores: Float[torch.Tensor, "n_hits"]
    """Read in the scan, to check the gather against. Rounded to float16 there."""


def gather_rows(
    model: torch.nn.Module,
    source: SampleSource,
    layer: int,
    scan: PairScan,
    hits: Sequence[PairHits],
    batch_size: int,
    device: str,
) -> list[PairRows]:
    """Re-runs only the samples holding a hit of any head."""
    attention = attention_of(model, layer)
    rows = [torch.empty(len(h.sample_idx), attention.head_dim) for h in hits]
    needed = np.unique(np.concatenate([h.sample_idx for h in hits]))
    for chunk in tqdm(list(batched(needed.tolist(), batch_size)), desc="gather"):
        ids = [scan.sample_ids[i] for i in chunk]
        query, key = capture_query_key(
            model, layer, source.to_model_batch(ids).to(device)
        )
        position_in_batch = {sample: i for i, sample in enumerate(chunk)}
        for head_hits, head_rows in zip(hits, rows):
            in_chunk = np.flatnonzero(np.isin(head_hits.sample_idx, chunk))
            if len(in_chunk) == 0:
                continue
            batch_idx = indices_on(
                [position_in_batch[s] for s in head_hits.sample_idx[in_chunk]], query
            )
            query_idx = indices_on(head_hits.query_idx[in_chunk], query)
            key_idx = indices_on(head_hits.key_idx[in_chunk], query)
            q = query[batch_idx, head_hits.head, query_idx]
            k = key[batch_idx, kv_head_of(attention, head_hits.head), key_idx]
            head_rows[in_chunk] = (attention.scaling * q.float() * k.float()).cpu()
    return [
        PairRows(
            hits=h,
            rows=r,
            scores=scan.scores[:, scan.heads.index(h.head)][
                torch.from_numpy(h.sample_idx),
                torch.from_numpy(h.query_idx),
                torch.from_numpy(h.key_idx),
            ].float(),
        )
        for h, r in zip(hits, rows)
    ]


def worst_gather_mismatch(pair_rows: PairRows) -> float:
    """Each row's sum against the score the scan read, relative to the score.

    - The scan rounded the scores to float16, so about 1e-3 is expected.
    """
    sums = pair_rows.rows.sum(-1)
    return ((sums - pair_rows.scores).abs() / pair_rows.scores.abs()).max().item()
