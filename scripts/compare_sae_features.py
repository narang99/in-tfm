#!/usr/bin/env python
"""Our clusters of a neuron's hits, against the Gemma Scope 2 SAE latents active at those hits.

    python scripts/compare_sae_features.py --config configs/gemma3_q_proj_sae.yaml

- Hits and clusters come from the same steps as a neuron report, without rendering one.
- The SAE reads the residual stream entering the layer, which is what the neuron reads after the norm.
- See in_tfm.sae_comparison for how an activation splits over latents.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from pydantic import BaseModel, Field
from sae_lens import SAE

from in_tfm.clustering import cluster_labels
from in_tfm.hadamard import hadamard_from_rows
from in_tfm.hit_selection import select_hits
from in_tfm.layers import input_layernorm_getter
from in_tfm.models.gemma3 import Gemma3Adapter
from in_tfm.neuron_capture import NeuronCapture, ScanResult, merge_positions
from in_tfm.neuron_run import layer_getter_for, prepare_run, timed
from in_tfm.run_config import RunConfig, config_path_from
from in_tfm.sae_comparison import (
    HitTable,
    HitTables,
    LatentSplit,
    NeuronComparison,
    SaeCheck,
    check_sae,
    compare_neuron,
    latent_contribution_vectors,
    split_over_latents,
)
from in_tfm.sources import TextSource
from in_tfm.text_corpus import load_texts


class SaeComparisonConfig(RunConfig):
    sae_release: str = Field("gemma-scope-2-270m-pt-res-all", description="sae-lens release name")
    sae_id: str = Field("layer_10_width_16k_l0_small", description="must read the residual entering layer_idx")
    n_random_tokens: int = Field(4000, description="random positions for the SAE reconstruction check")
    result_json: Path = Path("sae_comparison.json")


class SaeComparisonResult(BaseModel):
    sae_release: str
    sae_id: str
    layer_idx: int
    target: str
    sae_check: SaeCheck
    neurons: list[NeuronComparison]


def load_config(argv: list[str]) -> SaeComparisonConfig:
    path = config_path_from(argv)
    values = yaml.safe_load(path.read_text()) if path else {}
    return SaeComparisonConfig(_cli_parse_args=argv, **(values or {}))


def token_contexts(
    source: TextSource, tokenizer, batch_idx: np.ndarray, token_idx: np.ndarray, before: int = 12, after: int = 3
) -> tuple[list[str], list[str]]:
    """The hit token, and its text window with the token marked as [[token]]."""
    ids = source.sample_ids()
    rows = {int(s): source.to_model_batch([ids[s]]).display_ids[0].tolist() for s in np.unique(batch_idx)}
    def decode(span: list[int]) -> str:
        return tokenizer.decode(span, skip_special_tokens=True)

    tokens, contexts = [], []
    for b, t in zip(batch_idx, token_idx):
        row = rows[int(b)]
        tokens.append(tokenizer.convert_ids_to_tokens(row[t]))
        contexts.append(f"{decode(row[max(0, t - before) : t])}[[{decode([row[t]])}]]{decode(row[t + 1 : t + 1 + after])}")
    return tokens, contexts


def hit_table(
    neuron: int,
    labels: np.ndarray,
    sae_labels: np.ndarray,
    split: LatentSplit,
    contexts: list[str],
    n_latents: int = 5,
) -> HitTable:
    return HitTable(
        neuron=neuron,
        activation=split.activation.tolist(),
        ours=labels.tolist(),
        sae=sae_labels.tolist(),
        top_latents=split.top_latents[:, :n_latents].tolist(),
        top_contributions=np.round(split.top_contributions[:, :n_latents], 4).tolist(),
        error_share=np.round(split.error_term / split.activation, 4).tolist(),
        context=contexts,
    )


def random_positions(scan: ScanResult, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Skips each sample's first real token, the BOS, whose huge residual SAEs are not trained on."""
    mask = scan.valid_mask.numpy()
    not_bos = mask.cumsum(1) > 1
    valid = np.argwhere(mask & not_bos)
    picked = valid[np.random.default_rng(seed).choice(len(valid), size=min(n, len(valid)), replace=False)]
    picked = picked[np.lexsort((picked[:, 1], picked[:, 0]))]
    return picked[:, 0], picked[:, 1]


def cluster_with_config(points: np.ndarray, config: RunConfig) -> np.ndarray:
    clustering = config.clustering
    return cluster_labels(points, clustering.min_cluster_size, clustering.selection_method, clustering.min_samples)


def main() -> None:
    config = load_config(sys.argv[1:])
    texts = load_texts(config.text, config.seed)
    adapter = Gemma3Adapter.from_pretrained(config.model.value, config.text.max_length)
    run = prepare_run(config, adapter, texts)

    with timed("scan"):
        scan = run.capture.scan()
    selections = {n: select_hits(scan, n, "positive", config.hits, config.seed) for n in config.neuron_idxs}
    merged = merge_positions([(s.batch_idx, s.token_idx) for s in selections.values()], scan.valid_mask.shape[1])
    with timed("gather neuron inputs"):
        gathered = run.capture.gather(merged.batch_idx, merged.token_idx)
    run.capture.check_consistent(scan, gathered, merged.batch_idx, merged.token_idx)

    resid_capture = NeuronCapture(
        run.model, run.source, input_layernorm_getter(config.layer_idx), [0], config.batch_size, config.device
    )
    with timed("gather residuals"):
        resid = resid_capture.gather(merged.batch_idx, merged.token_idx).inputs.to(config.device)
        random_resid = resid_capture.gather(*random_positions(scan, config.n_random_tokens, config.seed)).inputs
    tokens, contexts = token_contexts(run.source, adapter.tokenizer, merged.batch_idx, merged.token_idx)

    sae = SAE.from_pretrained(config.sae_release, config.sae_id, device=config.device)
    sae_check = check_sae(sae, random_resid.to(config.device))
    print(f"SAE on random tokens: {sae_check}")

    layer = run.module.layers[config.layer_idx]
    weight = layer_getter_for(config)(run.module).weight
    comparisons, tables = [], []
    for column, (neuron, index_map) in enumerate(zip(config.neuron_idxs, merged.index_maps)):
        rows = gathered.inputs[index_map]
        labels = cluster_with_config(hadamard_from_rows(rows, weight, neuron), config)
        split = split_over_latents(resid[index_map], weight[neuron].to(config.device), layer.input_layernorm, sae)
        sae_labels = cluster_with_config(latent_contribution_vectors(split), config)
        captured = gathered.activations[index_map, column].numpy()
        comparison = compare_neuron(neuron, labels, split, captured, sae_labels, [tokens[i] for i in index_map])
        print(comparison.model_dump_json(indent=1, exclude={"clusters", "latents"}))
        comparisons.append(comparison)
        tables.append(hit_table(neuron, labels, sae_labels, split, [contexts[i] for i in index_map]))

    result = SaeComparisonResult(
        sae_release=config.sae_release, sae_id=config.sae_id, layer_idx=config.layer_idx,
        target=config.target.value, sae_check=sae_check, neurons=comparisons,
    )
    config.result_json.parent.mkdir(parents=True, exist_ok=True)
    config.result_json.write_text(result.model_dump_json(indent=1))
    config.result_json.with_suffix(".hits.json").write_text(HitTables(neurons=tables).model_dump_json())
    print(f"[done] -> {config.result_json}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
