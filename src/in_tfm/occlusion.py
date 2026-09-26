"""Occlusion test for a q_proj coordinate: does zeroing it at a set of positions change the loss?

Asks which way of picking "high activation" hits finds positions the model depends on. The
coordinate is zeroed at the *output of q_norm* - the value attention actually reads - not at
raw q_proj: zeroing before the norm also changes the head's RMS and rescales every other
coordinate of that head, so the effect would not be that of one coordinate.

Every set is zeroed at all of its positions in a sample at once, one forward pass per touched
sample, so the effect of a set is the summed change in that sample's next-token loss.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel, ConfigDict

from .sources import ModelBatch, SampleSource


class ScoredHits(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    batch_idx: Int[np.ndarray, "n_hits"]
    token_idx: Int[np.ndarray, "n_hits"]
    activation: Float[np.ndarray, "n_hits"]

    def __len__(self) -> int:
        return len(self.batch_idx)

    def top(self, n: int) -> "ScoredHits":
        order = np.argsort(-self.activation, kind="stable")[:n]
        return self._take(order)

    def excluding(self, other: "ScoredHits", seq_len: int) -> "ScoredHits":
        """Hits whose position is not a hit in `other`."""
        keep = ~np.isin(self._keys(seq_len), other._keys(seq_len))
        return self._take(np.flatnonzero(keep))

    def overlap_with(self, other: "ScoredHits", seq_len: int) -> int:
        return int(np.isin(self._keys(seq_len), other._keys(seq_len)).sum())

    def _keys(self, seq_len: int) -> Int[np.ndarray, "n_hits"]:
        return self.batch_idx.astype(np.int64) * seq_len + self.token_idx

    def _take(self, rows: Int[np.ndarray, "n_taken"]) -> "ScoredHits":
        return ScoredHits(
            batch_idx=self.batch_idx[rows], token_idx=self.token_idx[rows], activation=self.activation[rows]
        )


class HitSet(BaseModel):
    """Positions to zero. `expected` is the q_norm value the scan recorded at each of them, so
    the hook can check it is zeroing the coordinate it thinks it is."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    batch_idx: Int[np.ndarray, "n_hits"]
    token_idx: Int[np.ndarray, "n_hits"]
    expected: Float[np.ndarray, "n_hits"]


class SetEffect(BaseModel):
    name: str
    n_hits: int
    n_samples_touched: int
    net_loss_change: float
    net_loss_change_per_hit: float
    abs_sample_loss_change: float
    """Sum over samples of |change|, so hits that help and hits that hurt do not cancel."""


def hit_set_from(name: str, hits: ScoredHits, norm_activations: Float[np.ndarray, "batch seq"]) -> HitSet:
    return HitSet(
        name=name,
        batch_idx=hits.batch_idx,
        token_idx=hits.token_idx,
        expected=norm_activations[hits.batch_idx, hits.token_idx],
    )


def random_hit_set(
    name: str,
    valid_mask: Bool[np.ndarray, "batch seq"],
    norm_activations: Float[np.ndarray, "batch seq"],
    n_hits: int,
    seed: int,
) -> HitSet:
    """Uniform over real (unpadded) positions: the noise floor for zeroing this coordinate
    anywhere at all."""
    positions = np.argwhere(valid_mask)
    chosen = positions[np.random.default_rng(seed).choice(len(positions), n_hits, replace=False)]
    scored = ScoredHits(
        batch_idx=chosen[:, 0], token_idx=chosen[:, 1], activation=np.zeros(n_hits, dtype=np.float32)
    )
    return hit_set_from(name, scored, norm_activations)


def build_hit_sets(
    raw_hits: ScoredHits,
    norm_hits: ScoredHits,
    valid_mask: Bool[np.ndarray, "batch seq"],
    norm_activations: Float[np.ndarray, "batch seq"],
    n_hits: int,
    n_random_controls: int,
    seed: int,
) -> list[HitSet]:
    """All sets have the same size, since zeroing more positions changes the loss more.

    - `raw_top` / `norm_top`: the strongest hits under each selection, by its own activation.
    - `raw_only` / `norm_only`: strongest hits that the other selection does not find. The
      two selections agree on most of their hits, so this is where they actually differ.
    - `random_k`: uniformly random real positions.
    """
    seq_len = valid_mask.shape[1]
    raw_only = raw_hits.excluding(norm_hits, seq_len)
    norm_only = norm_hits.excluding(raw_hits, seq_len)
    n_hits = min(n_hits, len(raw_hits), len(norm_hits))
    n_disagreeing = min(n_hits, len(raw_only), len(norm_only))
    named_hits = {
        "raw_top": raw_hits.top(n_hits),
        "norm_top": norm_hits.top(n_hits),
        "raw_only": raw_only.top(n_disagreeing),
        "norm_only": norm_only.top(n_disagreeing),
    }
    sets = [hit_set_from(name, hits, norm_activations) for name, hits in named_hits.items()]
    sets += [
        random_hit_set(f"random_{k}", valid_mask, norm_activations, n_hits, seed + k) for k in range(n_random_controls)
    ]
    return sets


def shared_positions(first: HitSet, second: HitSet, seq_len: int) -> int:
    def keys(hit_set: HitSet) -> Int[np.ndarray, "n_hits"]:
        return hit_set.batch_idx.astype(np.int64) * seq_len + hit_set.token_idx

    return int(np.isin(keys(first), keys(second)).sum())


def split_neuron(neuron_idx: int, head_dim: int) -> tuple[int, int]:
    """Neuron index is `head * head_dim + dim`, the layout of q_proj's output."""
    return neuron_idx // head_dim, neuron_idx % head_dim


@contextmanager
def zeroed_coordinate(
    q_norm: torch.nn.Module,
    head: int,
    dim: int,
    token_idx: Int[torch.Tensor, "n_positions"],
    expected: Float[torch.Tensor, "n_positions"],
    atol: float = 1e-3,
) -> Iterator[None]:
    """q_norm's output is (batch, heads, seq, head_dim). Raises if the value about to be zeroed
    is not the one the scan saw at that position, which would mean the hook is on the wrong
    coordinate rather than the experiment showing something."""

    def hook(_module: torch.nn.Module, _inputs: tuple, output: Float[torch.Tensor, "1 heads seq head_dim"]):
        assert output.shape[0] == 1, "one sample per forward pass"
        before = output[0, head, token_idx, dim]
        if not torch.allclose(before, expected.to(before.dtype), atol=atol, rtol=atol):
            worst = (before - expected).abs().max().item()
            raise ValueError(f"q_norm value differs from the scan (max abs diff {worst:.3g})")
        zeroed = output.clone()
        zeroed[0, head, token_idx, dim] = 0.0
        return zeroed

    handle = q_norm.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def summed_next_token_loss(model: torch.nn.Module, batch: ModelBatch) -> float:
    """Over real tokens only: a position counts if it and the token it predicts are unpadded."""
    with torch.no_grad():
        logits = model(**batch.kwargs).logits[0, :-1].float()
    targets = batch.display_ids[0, 1:]
    counted = batch.valid_mask[0, :-1] & batch.valid_mask[0, 1:]
    return F.cross_entropy(logits[counted], targets[counted], reduction="sum").item()


class OcclusionRunner:
    def __init__(
        self,
        model: torch.nn.Module,
        q_norm: torch.nn.Module,
        source: SampleSource,
        sample_ids: list[str],
        neuron_idx: int,
        head_dim: int,
        device: str,
    ) -> None:
        self.model = model
        self.q_norm = q_norm
        self.source = source
        self.sample_ids = sample_ids
        self.head, self.dim = split_neuron(neuron_idx, head_dim)
        self.device = device
        self._batches: dict[int, ModelBatch] = {}
        self._baseline: dict[int, float] = {}

    def effect_of(self, hit_set: HitSet) -> SetEffect:
        changes = [
            self._sample_loss_change(sample_idx, hit_set) for sample_idx in np.unique(hit_set.batch_idx)
        ]
        return SetEffect(
            name=hit_set.name,
            n_hits=len(hit_set.batch_idx),
            n_samples_touched=len(changes),
            net_loss_change=sum(changes),
            net_loss_change_per_hit=sum(changes) / len(hit_set.batch_idx),
            abs_sample_loss_change=sum(abs(change) for change in changes),
        )

    def check_baseline_is_repeatable(self, sample_idx: int = 0, atol: float = 1e-2) -> float:
        """Without this, a loss change could be nondeterminism in the forward pass."""
        first = summed_next_token_loss(self.model, self._batch(sample_idx))
        second = summed_next_token_loss(self.model, self._batch(sample_idx))
        if abs(first - second) > atol:
            raise ValueError(f"the same sample gave two losses: {first} vs {second}")
        return abs(first - second)

    def _sample_loss_change(self, sample_idx: int, hit_set: HitSet) -> float:
        rows = hit_set.batch_idx == sample_idx
        with zeroed_coordinate(
            self.q_norm,
            self.head,
            self.dim,
            torch.from_numpy(hit_set.token_idx[rows]).to(self.device),
            torch.from_numpy(hit_set.expected[rows]).to(self.device),
        ):
            zeroed = summed_next_token_loss(self.model, self._batch(sample_idx))
        return zeroed - self._baseline_loss(sample_idx)

    def _baseline_loss(self, sample_idx: int) -> float:
        if sample_idx not in self._baseline:
            self._baseline[sample_idx] = summed_next_token_loss(self.model, self._batch(sample_idx))
        return self._baseline[sample_idx]

    def _batch(self, sample_idx: int) -> ModelBatch:
        if sample_idx not in self._batches:
            self._batches[sample_idx] = self.source.to_model_batch([self.sample_ids[sample_idx]]).to(self.device)
        return self._batches[sample_idx]
