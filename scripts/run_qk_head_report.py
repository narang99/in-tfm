#!/usr/bin/env python
"""Initial test: cluster the q * k Hadamard product of one attention head of the attn-only model.

A hit is a (query i, key j) pair with j <= i, and its vector is `q_i * k_j` over the head's
`head_dim` coordinates, so its sum is the pre-softmax score times sqrt(head_dim). Everything
else is the usual pipeline: bucketed hit selection, HDBSCAN, AttnLRP, the text report.

    python scripts/run_qk_head_report.py --config configs/attn_only_2l_qk_head.yaml --head 7

Hacky on purpose, see the hooks below:
- a pair is flattened to the position `i * seq + j` so the bucketed strategy runs unchanged
- the report reuses `NeuronClusterHits`, with a `sample_cluster` that explains a pair
- the report titles still say "Neuron", the number is the head
"""

import gc
import sys
from types import SimpleNamespace

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from sklearn.preprocessing import normalize
from tqdm import tqdm

from in_tfm.clustering import cluster_labels
from in_tfm.device import empty_cache
from in_tfm.hit_selection import HitSelection
from in_tfm.hit_selection.strategies.bucketed import bucketed_hits
from in_tfm.layers import q_proj_getter
from in_tfm.models.attn_only_2l import AttnOnly2LAdapter
from in_tfm.neuron_report import NeuronClusterHits
from in_tfm.presenters import ClusterHit
from in_tfm.run_config import RunConfig, load_run_config
from in_tfm.sources import ModelBatch
from in_tfm.sources.wikitext import load_texts

HEAD_FLAG = "--head"


def pop_head(argv: list[str]) -> tuple[int, list[str]]:
    """`RunConfig` forbids unknown flags, so the head is taken off before it sees them."""
    at = argv.index(HEAD_FLAG)
    return int(argv[at + 1]), argv[:at] + argv[at + 2 :]


def head_columns(head: int, head_dim: int) -> slice:
    return slice(head * head_dim, (head + 1) * head_dim)


class QKCapture:
    """The head's raw q_proj and k_proj outputs of one layer, for one forward pass."""

    def __init__(self, module: torch.nn.Module, layer_idx: int) -> None:
        attn = module.layers[layer_idx].self_attn
        self.captured: dict[str, torch.Tensor] = {}
        self.handles = [
            proj.register_forward_hook(lambda _m, _i, out, name=name: self.captured.__setitem__(name, out))
            for name, proj in (("q", attn.q_proj), ("k", attn.k_proj))
        ]

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


def scan_head(
    module: torch.nn.Module, source, layer_idx: int, head: int, batch_size: int, device: str
) -> tuple[Float[torch.Tensor, "n seq head_dim"], Float[torch.Tensor, "n seq head_dim"], Bool[torch.Tensor, "n seq"]]:
    columns = head_columns(head, module.config.head_dim)
    capture = QKCapture(module, layer_idx)
    queries, keys, masks = [], [], []
    ids = list(source.sample_ids())
    try:
        for start in tqdm(range(0, len(ids), batch_size), desc="scan"):
            batch = source.to_model_batch(ids[start : start + batch_size]).to(device)
            with torch.no_grad():
                module(**batch.kwargs)
            queries.append(capture.captured["q"][..., columns].cpu())
            keys.append(capture.captured["k"][..., columns].cpu())
            masks.append(batch.valid_mask.cpu())
    finally:
        capture.remove()
    return torch.cat(queries), torch.cat(keys), torch.cat(masks)


def pair_scores(
    queries: Float[torch.Tensor, "n seq dim"], keys: Float[torch.Tensor, "n seq dim"], valid: Bool[torch.Tensor, "n seq"]
) -> tuple[Float[torch.Tensor, "n seq_pairs 1"], Bool[torch.Tensor, "n seq_pairs"]]:
    """Every (i, j) flattened to `i * seq + j`. A pair counts when j <= i and both are real tokens."""
    n, seq, _ = queries.shape
    scores = torch.einsum("bid,bjd->bij", queries, keys)
    causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
    pair_valid = causal[None] & valid[:, :, None] & valid[:, None, :]
    return scores.reshape(n, seq * seq, 1), pair_valid.reshape(n, seq * seq)


def select_pairs(
    scores: Float[torch.Tensor, "n seq_pairs 1"], pair_valid: Bool[torch.Tensor, "n seq_pairs"], config: RunConfig
) -> tuple[HitSelection, Int[np.ndarray, "n_hits"]]:
    column = scores.masked_fill(~pair_valid[..., None], 0.0)
    positive = scores[..., 0][pair_valid]
    sorted_magnitudes = torch.sort(positive[positive > 0]).values
    picked, pick = bucketed_hits(
        column, sorted_magnitudes, SimpleNamespace(valid_mask=pair_valid), config.hits, config.seed
    )
    seq_pairs = scores.shape[1]
    seq = int(round(seq_pairs**0.5))
    selection = HitSelection(
        polarity="positive",
        batch_idx=picked.batch_idx,
        token_idx=picked.token_idx // seq,
        activations=scores[picked.batch_idx, picked.token_idx, 0].numpy(),
        n_candidates=picked.n_candidates,
        sorted_magnitudes=sorted_magnitudes,
        pick=pick,
    )
    print(f"[pairs] bucketed floor={pick.floor:.4f}, {picked.n_candidates} pairs above it, {len(picked.batch_idx)} kept")
    return selection, picked.token_idx % seq


def pair_relevance(
    module: torch.nn.Module, batch: ModelBatch, layer_idx: int, head: int, query_idx: int, key_idx: int
) -> Float[np.ndarray, "1 seq hidden"]:
    """AttnLRP for the pair's score: grad * input of the embeddings, back from `sum(q_i * k_j)`.

    The score is linear in the embeddings up to the position term and the biases, which have no
    token to be attributed to, so the relevance sums to the token part of the score only.
    """
    module.eval()
    kwargs, leaf = batch.differentiable()
    capture = QKCapture(module, layer_idx)
    columns = head_columns(head, module.config.head_dim)
    try:
        with torch.enable_grad():
            module(**kwargs)
            q, k = capture.captured["q"][0, query_idx, columns], capture.captured["k"][0, key_idx, columns]
            (q * k).sum().backward()
    finally:
        capture.remove()
    return (leaf.grad * leaf).detach().cpu().numpy()


class QKPairHits(NeuronClusterHits):
    layer_idx: int
    head: int
    key_idx: Int[np.ndarray, "n_hits"]

    def sample_cluster(self, cluster_id: int, max_n: int = 5, attr_fn=None) -> list[ClusterHit]:
        from in_tfm.neuron_report import sampled_hit_indices

        module = self.model.to(self.device)
        results: list[ClusterHit] = []
        for i in tqdm(sampled_hit_indices(self.labels, cluster_id, max_n)):
            sample_id = self.sample_ids[self.selection.batch_idx[i]]
            batch = self.source.to_model_batch([sample_id]).to(self.device)
            relevance = pair_relevance(
                module, batch, self.layer_idx, self.head, int(self.selection.token_idx[i]), int(self.key_idx[i])
            )
            results.append(
                ClusterHit(
                    sample_id=sample_id,
                    token_idx=int(self.selection.token_idx[i]),
                    relevance=relevance,
                    model_input=batch.grad_leaf.detach().cpu(),
                    hadamard=self.hdmds[i],
                    display_ids=batch.display_ids[0].cpu(),
                )
            )
            del batch
            empty_cache()
            gc.collect()
        return results


def main() -> None:
    head, argv = pop_head(sys.argv[1:])
    config = load_run_config(argv)
    layer_idx = config.layer_idx

    adapter = AttnOnly2LAdapter.from_pretrained(config.text.max_length)
    adapter.patch_for_attn_lrp()
    module = adapter.get_model().to(config.device)
    source = adapter.make_source(load_texts(config.text, config.seed))

    queries, keys, valid = scan_head(module, source, layer_idx, head, config.batch_size, config.device)
    scores, pair_valid = pair_scores(queries, keys, valid)
    selection, key_idx = select_pairs(scores, pair_valid, config)

    hadamards = queries[selection.batch_idx, selection.token_idx] * keys[selection.batch_idx, key_idx]
    hdmd = normalize(hadamards.numpy(), "l2")
    c = config.clustering
    labels = cluster_labels(hdmd, c.min_cluster_size, c.selection_method, c.min_samples, c.distance)
    print(f"[pairs] {len(set(labels) - {-1})} clusters, {int((labels == -1).sum())} noise")

    hits = QKPairHits(
        model=module,
        source=source,
        presenter=adapter.make_presenter(source, "q * k hadamard products"),
        sample_ids=list(source.sample_ids()),
        layer_getter=q_proj_getter(layer_idx),
        neuron_idx=head,
        selection=selection,
        labels=labels,
        hdmds=hdmd,
        hadamard_shape=(8, 8),
        device=config.device,
        layer_idx=layer_idx,
        head=head,
        key_idx=key_idx,
    )
    path = hits.write_report(
        config.out_dir,
        name=f"head_{layer_idx}_{head}",
        max_n=config.report.max_hits_per_cluster,
        min_uniq_images=config.report.min_uniq_samples_per_cluster,
    )
    print(f"[pairs] report -> {path}")


if __name__ == "__main__":
    main()
