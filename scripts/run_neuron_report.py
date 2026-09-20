#!/usr/bin/env python
"""Runs the full neuron-report pipeline for a range of neurons in one layer: capture
activations once for a batch of DICOMs, then for each neuron find its activation threshold via
elbow detection, cluster its Hadamard products, and write an AttnLRP report.

Activations are captured once and reused across every neuron in the range (thresholding and
clustering are per-neuron, but the underlying fc2 input/output tensors aren't), and the AttnLRP
model is loaded once and patched a single time since patch_for_attn_lrp mutates process-wide
state (see its docstring) - reloading/repatching per neuron would be redundant.

Clustering picks a GPU backend when one is available - see in_tfm.clustering. On CPU, keep
--n-dicoms small: hit count grows linearly with it while clustering cost grows with the square
of the hit count, so 44 dicoms is ~30x the clustering work of 8, not ~5x.

CLI wrapper around the same steps embs-v2.ipynb walks through interactively - see that
notebook for the exploratory version (elbow plot, spot-checking clusters) this distills.
"""

import argparse
import math
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from nnsight import NNsight
from rad_dino import RadDino
from transformers import AutoImageProcessor
from transformers.image_processing_utils import BaseImageProcessor

from in_tfm.activations import get_activations
from in_tfm.clustering import cluster_labels
from in_tfm.device import default_device, empty_cache
from in_tfm.attribution import compute_attnlrp_relevance, patch_for_attn_lrp
from in_tfm.hadamard import hadamard_products, high_activation_hits
from in_tfm.layers import fc2_getter
from in_tfm.neuron_report import NeuronClusterHits
from in_tfm.presenters import ImagePresenter
from in_tfm.sources import DicomSource
from in_tfm.threshold import find_activation_threshold

ACT_SHAPE = (48, 64)  # 48*64 == mlp intermediate size (3072); reshape target for hdmd plots


@contextmanager
def timed(label: str):
    start = time.perf_counter()
    yield
    print(f"[{label}] {time.perf_counter() - start:.2f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dcm-dir", type=Path, default=Path("dicoms"))
    parser.add_argument("--layer-idx", type=int, default=11)
    parser.add_argument("--neuron-start", type=int, default=90)
    parser.add_argument("--neuron-end", type=int, default=100, help="inclusive")
    parser.add_argument("--n-dicoms", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-cluster-size", type=int, default=10)
    parser.add_argument("--max-hits-per-cluster", type=int, default=10)
    parser.add_argument(
        "--min-uniq-images-per-cluster",
        type=int,
        default=2,
        help="drop clusters drawn from fewer distinct source images than this (a single-image "
        "cluster is likely a repeated local artifact, not a real pattern)",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--device", default=default_device())
    return parser.parse_args()


def neuron_range(args: argparse.Namespace) -> range:
    return range(args.neuron_start, args.neuron_end + 1)


def load_model(device: str) -> tuple[NNsight, RadDino, AutoImageProcessor]:
    processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
    hf_model = RadDino()
    model = NNsight(hf_model.model).to(device)
    return model, hf_model, processor


def clamp_n_dicoms_to_available(args: argparse.Namespace) -> None:
    available = sum(1 for _ in args.dcm_dir.glob("*.dcm"))
    if args.n_dicoms > available:
        print(f"only {available} dicoms found in {args.dcm_dir}, using all of them (requested {args.n_dicoms})")
        args.n_dicoms = available


def capture_activations(args: argparse.Namespace, model: NNsight, source: DicomSource):
    n_iter = math.ceil(args.n_dicoms / args.batch_size)
    return get_activations(
        model,
        source,
        fc2_getter(args.layer_idx),
        n_iter=n_iter,
        bs=args.batch_size,
        device=args.device,
    )


def load_attnlrp_model() -> RadDino:
    """patch_for_attn_lrp mutates LayerNorm/Dropout at the class level (process-wide, see its
    docstring), so this only needs to run once and the same patched model is reused for every
    neuron's report."""
    lrp_model = RadDino()
    patch_for_attn_lrp(lrp_model.model)
    return lrp_model


def cluster_hadamards(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    valid_mask: torch.Tensor,
    fc2_weight: torch.Tensor,
    neuron_idx: int,
    thresh: float,
    min_cluster_size: int,
):
    batch_idx, token_idx = high_activation_hits(outputs, neuron_idx, thresh, valid_mask)
    hdmd = hadamard_products(inputs, batch_idx, token_idx, fc2_weight, neuron_idx)
    return batch_idx, token_idx, hdmd, cluster_labels(hdmd, min_cluster_size)


def build_hits(
    model,
    source: DicomSource,
    presenter: ImagePresenter,
    sample_ids: list[str],
    neuron_idx: int,
    batch_idx,
    token_idx,
    labels,
    hdmd,
    threshold: float,
    elbow_values: torch.Tensor,
    elbow_idx: int,
    args: argparse.Namespace,
) -> NeuronClusterHits:
    return NeuronClusterHits(
        model=model,
        source=source,
        presenter=presenter,
        sample_ids=sample_ids,
        layer_getter=fc2_getter(args.layer_idx),
        neuron_idx=neuron_idx,
        token_idx=token_idx,
        batch_idx=batch_idx,
        labels=labels,
        hdmds=hdmd,
        threshold=threshold,
        elbow_values=elbow_values.numpy(),
        elbow_idx=elbow_idx,
        device=args.device,
    )


def process_neuron(
    neuron_idx: int,
    sample_ids: list[str],
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    valid_mask: torch.Tensor,
    fc2_weight: torch.Tensor,
    lrp_model: RadDino,
    source: DicomSource,
    presenter: ImagePresenter,
    args: argparse.Namespace,
) -> None:
    threshold, elbow_values, elbow_idx = find_activation_threshold(outputs, neuron_idx, valid_mask)
    print(f"[neuron {neuron_idx}] threshold={threshold:.4f}")

    with timed(f"neuron {neuron_idx}: cluster"):
        batch_idx, token_idx, hdmd, labels = cluster_hadamards(
            inputs, outputs, valid_mask, fc2_weight, neuron_idx, threshold, args.min_cluster_size
        )
    n_clusters = len(set(labels) - {-1})
    print(f"[neuron {neuron_idx}] {len(batch_idx)} hits above threshold, {n_clusters} clusters")

    hits = build_hits(
        lrp_model.model,
        source,
        presenter,
        sample_ids,
        neuron_idx,
        batch_idx,
        token_idx,
        labels,
        hdmd,
        threshold,
        elbow_values,
        elbow_idx,
        args,
    )
    with timed(f"neuron {neuron_idx}: write report"):
        report_path = hits.write_report(
            args.out_dir,
            name=f"neuron_{neuron_idx}",
            attr_fn=compute_attnlrp_relevance,
            max_n=args.max_hits_per_cluster,
            min_uniq_images=args.min_uniq_images_per_cluster,
        )
    print(f"[neuron {neuron_idx}] report -> {report_path}")


def main() -> None:
    args = parse_args()
    clamp_n_dicoms_to_available(args)
    pipeline_start = time.perf_counter()

    with timed("load model"):
        model, hf_model, processor = load_model(args.device)

    source = DicomSource(args.dcm_dir, processor)
    presenter = ImagePresenter(processor, ACT_SHAPE)

    with timed("capture activations"):
        sample_ids, inputs, outputs, valid_mask = capture_activations(args, model, source)
    print(f"captured activations for {len(sample_ids)} dicoms")

    # inputs/outputs are already cpu tensors (see get_activations) - the fc2 weight used below
    # for the hadamard product must match, so free the GPU copy of the model first.
    model.to("cpu")
    empty_cache()
    fc2_weight = hf_model.model.encoder.layer[args.layer_idx].mlp.fc2.weight

    with timed("load attnlrp model"):
        lrp_model = load_attnlrp_model()

    for neuron_idx in neuron_range(args):
        process_neuron(
            neuron_idx, sample_ids, inputs, outputs, valid_mask,
            fc2_weight, lrp_model, source, presenter, args,
        )

    print(f"[total] {time.perf_counter() - pipeline_start:.2f}s")


if __name__ == "__main__":
    main()
