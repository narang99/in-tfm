#!/usr/bin/env python
"""Neuron reports for a decoder language model, using the same pipeline as the vision path.

Everything between activation capture and clustering is shared with run_neuron_report.py - only
the source (text instead of DICOMs), the presenter (tokens in context instead of pixel overlays)
and the layer getter (down_proj instead of fc2) differ. See in_tfm.sources for why text has to
hand the model embeddings rather than token ids.

Clusters here group tokens by *which intermediate MLP neurons drove the activation*, so a
cluster is a claim about what kind of language the neuron responds to - the report prints the
firing token for each cluster so that claim is readable at a glance.
"""

import argparse
import math
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from nnsight import NNsight
from transformers import AutoModelForCausalLM, AutoTokenizer

from in_tfm.activations import get_activations
from in_tfm.attribution import compute_attnlrp_relevance, patch_gemma3_for_attn_lrp
from in_tfm.clustering import cluster_labels
from in_tfm.device import default_device, empty_cache
from in_tfm.hadamard import hadamard_products, high_activation_hits, near_square_shape
from in_tfm.layers import down_proj_getter
from in_tfm.neuron_report import NeuronClusterHits
from in_tfm.presenters import TextPresenter
from in_tfm.sources import TextSource
from in_tfm.threshold import find_activation_threshold


@contextmanager
def timed(label: str):
    start = time.perf_counter()
    yield
    print(f"[{label}] {time.perf_counter() - start:.2f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="google/gemma-3-270m")
    parser.add_argument("--dataset", default="Salesforce/wikitext", help="hf dataset name")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--layer-idx", type=int, default=10)
    parser.add_argument("--neuron-start", type=int, default=90)
    parser.add_argument("--neuron-end", type=int, default=90, help="inclusive")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-cluster-size", type=int, default=10)
    parser.add_argument("--max-hits-per-cluster", type=int, default=10)
    parser.add_argument("--min-uniq-samples-per-cluster", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=Path("reports_llm"))
    parser.add_argument("--device", default=default_device())
    return parser.parse_args()


def load_texts(args: argparse.Namespace) -> list[str]:
    """Non-empty lines only - wikitext is full of blank lines and bare `= Heading =` markers,
    which tokenize to almost nothing and would pad out every batch."""
    from datasets import load_dataset

    ds = load_dataset(args.dataset, args.dataset_config, split=args.split, streaming=True)
    texts: list[str] = []
    for row in ds:
        text = row["text"].strip()
        if len(text) > 200 and not text.startswith("="):
            texts.append(text)
        if len(texts) >= args.n_samples:
            break
    return texts


def load_model(args: argparse.Namespace) -> tuple[NNsight, torch.nn.Module, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    base = hf_model.model  # the decoder stack; down_proj_getter indexes .layers on it
    return NNsight(base).to(args.device), hf_model, tokenizer


def process_neuron(
    neuron_idx: int,
    sample_ids: list[str],
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    valid_mask: torch.Tensor,
    down_proj_weight: torch.Tensor,
    lrp_model: torch.nn.Module,
    source: TextSource,
    presenter: TextPresenter,
    args: argparse.Namespace,
) -> None:
    threshold, elbow_values, elbow_idx = find_activation_threshold(outputs, neuron_idx, valid_mask)
    print(f"[neuron {neuron_idx}] threshold={threshold:.4f}")

    batch_idx, token_idx = high_activation_hits(outputs, neuron_idx, threshold, valid_mask)
    hdmd = hadamard_products(inputs, batch_idx, token_idx, down_proj_weight, neuron_idx)
    with timed(f"neuron {neuron_idx}: cluster"):
        labels = cluster_labels(hdmd, args.min_cluster_size)
    print(f"[neuron {neuron_idx}] {len(batch_idx)} hits above threshold, "
          f"{len(set(labels) - {-1})} clusters")

    hits = NeuronClusterHits(
        model=lrp_model,
        source=source,
        presenter=presenter,
        sample_ids=sample_ids,
        layer_getter=down_proj_getter(args.layer_idx),
        neuron_idx=neuron_idx,
        token_idx=token_idx,
        batch_idx=batch_idx,
        labels=labels,
        hdmds=hdmd,
        hadamard_shape=near_square_shape(hdmd.shape[1]),
        threshold=threshold,
        elbow_values=elbow_values.numpy(),
        elbow_idx=elbow_idx,
        device=args.device,
    )
    with timed(f"neuron {neuron_idx}: write report"):
        report_path = hits.write_report(
            args.out_dir,
            name=f"neuron_{neuron_idx}",
            attr_fn=compute_attnlrp_relevance,
            max_n=args.max_hits_per_cluster,
            min_uniq_images=args.min_uniq_samples_per_cluster,
        )
    print(f"[neuron {neuron_idx}] report -> {report_path}")


def main() -> None:
    args = parse_args()
    pipeline_start = time.perf_counter()

    with timed("load texts"):
        texts = load_texts(args)
    print(f"loaded {len(texts)} texts")

    with timed("load model"):
        model, hf_model, tokenizer = load_model(args)

    source = TextSource(texts, tokenizer, hf_model.model.embed_tokens, args.max_length)
    presenter = TextPresenter(source)

    with timed("capture activations"):
        sample_ids, inputs, outputs, valid_mask = get_activations(
            model,
            source,
            down_proj_getter(args.layer_idx),
            n_iter=math.ceil(len(texts) / args.batch_size),
            bs=args.batch_size,
            device=args.device,
        )
    print(f"captured activations for {len(sample_ids)} texts, shape {tuple(outputs.shape)}, "
          f"{int(valid_mask.sum())}/{valid_mask.numel()} positions unpadded")

    model.to("cpu")
    empty_cache()
    down_proj_weight = hf_model.model.layers[args.layer_idx].mlp.down_proj.weight

    with timed("patch for attnlrp"):
        patch_gemma3_for_attn_lrp(hf_model.model)

    for neuron_idx in range(args.neuron_start, args.neuron_end + 1):
        process_neuron(
            neuron_idx, sample_ids, inputs, outputs, valid_mask,
            down_proj_weight, hf_model.model, source, presenter, args,
        )

    print(f"[total] {time.perf_counter() - pipeline_start:.2f}s")


if __name__ == "__main__":
    main()
