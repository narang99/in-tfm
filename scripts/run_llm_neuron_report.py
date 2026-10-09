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
import random
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from jaxtyping import Float, Int
from nnsight import NNsight
from pydantic import BaseModel, ConfigDict

from in_tfm.attribution import compute_attnlrp_relevance
from in_tfm.bucketing import bucketed_sample, merged_bucket_edges
from in_tfm.clustering import cluster_labels
from in_tfm.device import default_device, empty_cache
from in_tfm.hadamard import hadamard_from_rows, high_activation_hits, near_square_shape, normalized_rows
from in_tfm.layers import LayerGetter, down_proj_getter, k_proj_getter, q_proj_getter
from in_tfm.models.attn_only_2l import AttnOnly2LAdapter
from in_tfm.models.gemma3 import Gemma3Adapter
from in_tfm.models import ModelAdapter
from in_tfm.neuron_capture import NeuronCapture, ScanResult, merge_positions
from in_tfm.neuron_report import NeuronClusterHits
from in_tfm.presenters import ClusterPresenter
from in_tfm.sources import SampleSource
from in_tfm.threshold import find_activation_threshold


@contextmanager
def timed(label: str):
    start = time.perf_counter()
    yield
    print(f"[{label}] {time.perf_counter() - start:.2f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="google/gemma-3-270m")
    parser.add_argument(
        "--arch",
        choices=["gemma3", "attn_only_2l"],
        default="gemma3",
        help="attn_only_2l: the 2-layer attention-only shortformer model, which ignores --model "
        "and has no MLP, so only the q_proj / k_proj targets exist",
    )
    parser.add_argument("--dataset", default="Salesforce/wikitext", help="hf dataset name")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--layer-idx", type=int, default=10)
    parser.add_argument(
        "--target",
        choices=["down_proj", "q_proj", "k_proj"],
        default="down_proj",
        help="down_proj: MLP neurons. q_proj / k_proj: query / key coordinates, "
        "neuron index = (kv_)head * head_dim + dim",
    )
    parser.add_argument(
        "--polarity",
        choices=["positive", "negative", "both"],
        default="positive",
        help="which tail of the activation distribution counts as a hit",
    )
    parser.add_argument(
        "--cluster-on",
        choices=["hadamard", "input"],
        default="hadamard",
        help="hadamard: input * weight row (default). input: the raw input at each hit, "
        "L2-normalised the same way - an ablation of the weight row",
    )
    parser.add_argument("--neurons", type=int, nargs="+", help="neuron indices; overrides the range below")
    parser.add_argument("--neuron-start", type=int, default=90)
    parser.add_argument("--neuron-end", type=int, default=90, help="inclusive")
    parser.add_argument(
        "--unit",
        choices=["paragraph", "article"],
        default="paragraph",
        help="article joins a wikitext article's paragraphs, so long --max-length is mostly real tokens",
    )
    parser.add_argument("--min-article-chars", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0, help="corpus shuffle and hit subsampling")
    parser.add_argument("--max-hits", type=int, default=20000, help="hits kept per neuron before clustering")
    parser.add_argument(
        "--hit-selection",
        choices=["elbow", "bucketed"],
        default="elbow",
        help="elbow: everything above the elbow, uniformly subsampled to --max-hits. "
        "bucketed: equal-width activation buckets, an equal share of --max-hits from each",
    )
    parser.add_argument("--n-buckets", type=int, default=10)
    parser.add_argument(
        "--min-bucket-size",
        type=int,
        default=100,
        help="bucketed only: buckets with fewer positions than this are merged with their neighbour",
    )
    parser.add_argument(
        "--bucket-floor-fraction",
        type=float,
        default=0.0,
        help="bucketed only: lowest bucket starts at this fraction of the max activation",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-cluster-size", type=int, default=10)
    parser.add_argument(
        "--min-samples", type=int, default=None, help="HDBSCAN min_samples, defaults to --min-cluster-size"
    )
    parser.add_argument("--cluster-selection-method", choices=["eom", "leaf"], default="leaf")
    parser.add_argument("--max-hits-per-cluster", type=int, default=10)
    parser.add_argument("--min-uniq-samples-per-cluster", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=Path("reports_llm"))
    parser.add_argument("--device", default=default_device())
    return parser.parse_args()


WIKITEXT_TITLE = re.compile(r"^ = [^=].* = \n?$")
"""A level-1 heading; the `= =` of a section heading fails the `[^=]` after the first `= `."""


def paragraphs(rows: list[str]) -> list[str]:
    """Non-empty paragraphs only - wikitext is full of blank lines and bare `= Heading =`
    markers, which tokenize to almost nothing and would pad out every batch."""
    stripped = [row.strip() for row in rows]
    return [text for text in stripped if len(text) > 200 and not text.startswith("=")]


def articles(rows: list[str]) -> list[str]:
    """Rows are single paragraphs; a level-1 heading starts a new article."""
    joined: list[list[str]] = [[]]
    for row in rows:
        if WIKITEXT_TITLE.match(row):
            joined.append([])
        joined[-1].append(row)
    return ["".join(parts).strip() for parts in joined]


def load_texts(args: argparse.Namespace) -> list[str]:
    """Shuffled before truncating to `--n-samples`: the corpus is in file order, so taking the
    first N would be N paragraphs of the same few articles."""
    from datasets import load_dataset

    rows = list(load_dataset(args.dataset, args.dataset_config, split=args.split)["text"])
    if args.unit == "paragraph":
        texts = paragraphs(rows)
    else:
        texts = [a for a in articles(rows) if len(a) >= args.min_article_chars]
    random.Random(args.seed).shuffle(texts)
    return texts[: args.n_samples]


def layer_getter_for(args: argparse.Namespace) -> LayerGetter:
    getters = {"down_proj": down_proj_getter, "q_proj": q_proj_getter, "k_proj": k_proj_getter}
    return getters[args.target](args.layer_idx)


def load_adapter(args: argparse.Namespace) -> ModelAdapter[list[str]]:
    if args.arch == "attn_only_2l":
        return AttnOnly2LAdapter.from_pretrained(args.max_length)
    return Gemma3Adapter.from_pretrained(args.model, args.max_length)


Polarity = Literal["positive", "negative"]


def polarities_for(args: argparse.Namespace) -> list[Polarity]:
    return ["positive", "negative"] if args.polarity == "both" else [args.polarity]


def report_name(neuron_idx: int, polarity: Polarity) -> str:
    """Positive keeps the historical `neuron_{idx}` so existing reports are not renamed."""
    return f"neuron_{neuron_idx}" if polarity == "positive" else f"neuron_{neuron_idx}_neg"


class PickedHits(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    batch_idx: Int[np.ndarray, "n_hits"]
    token_idx: Int[np.ndarray, "n_hits"]
    n_above_threshold: int
    bucket_ids: Int[np.ndarray, "n_hits"] | None = None
    """Merged bucket per hit, bucketed selection only."""
    bucket_edges: Float[np.ndarray, "n_edges"] | None = None
    """Lower edge of every merged bucket, then the max. Bucketed selection only."""


class HitSelection(PickedHits):
    threshold: float
    elbow_values: Float[torch.Tensor, "n_pos"]
    elbow_idx: int


def elbow_hits(column: Float[torch.Tensor, "batch seq 1"], magnitude: float, scan: ScanResult, args: argparse.Namespace) -> PickedHits:
    batch_idx, token_idx = high_activation_hits(column, 0, magnitude, scan.valid_mask)
    n_above = len(batch_idx)
    if n_above > args.max_hits:
        keep = np.sort(np.random.default_rng(args.seed).choice(n_above, args.max_hits, replace=False))
        batch_idx, token_idx = batch_idx[keep], token_idx[keep]
    return PickedHits(batch_idx=batch_idx, token_idx=token_idx, n_above_threshold=n_above)


def bucketed_hits(column: Float[torch.Tensor, "batch seq 1"], floor: float, scan: ScanResult, args: argparse.Namespace) -> PickedHits:
    """Row-major order of `argwhere` on the mask matches boolean-mask indexing, so row i of
    `positions` is the position of `values[i]`."""
    valid = scan.valid_mask.numpy()
    values = column[..., 0].numpy()[valid]
    positions = np.argwhere(valid)
    kept, bucket_ids = bucketed_sample(
        values, args.n_buckets, args.max_hits, floor, args.min_bucket_size, np.random.default_rng(args.seed)
    )
    above = values[values > floor]
    return PickedHits(
        batch_idx=positions[kept, 0],
        token_idx=positions[kept, 1],
        n_above_threshold=len(above),
        bucket_ids=bucket_ids,
        bucket_edges=merged_bucket_edges(above, args.n_buckets, floor, args.min_bucket_size),
    )


def select_hits(scan: ScanResult, neuron_idx: int, polarity: Polarity, args: argparse.Namespace) -> HitSelection:
    """Positions to cluster, capped at `--max-hits`: clustering 640-d vectors is superlinear in
    hit count, and a common token can produce hundreds of thousands.

    - The negative tail is found by negating the column: the elbow and hit helpers only look
      at positive values, so the same code finds the most negative activations.
    - `elbow_values` are then magnitudes, while `threshold` is reported signed.
    - With `--hit-selection bucketed` the elbow is still computed for the plot, but the
      reported threshold is the bucket floor."""
    column = scan.neuron_column(neuron_idx)
    if polarity == "negative":
        column = -column
    magnitude, elbow_values, elbow_idx = find_activation_threshold(column, 0, scan.valid_mask)
    if args.hit_selection == "bucketed":
        magnitude = args.bucket_floor_fraction * elbow_values[-1].item()
        picked = bucketed_hits(column, magnitude, scan, args)
    else:
        picked = elbow_hits(column, magnitude, scan, args)
    threshold = -magnitude if polarity == "negative" else magnitude
    print(f"[neuron {neuron_idx} {polarity}] threshold={threshold:.4f}, {picked.n_above_threshold} hits beyond it, {len(picked.batch_idx)} kept")
    return HitSelection(threshold=threshold, elbow_values=elbow_values, elbow_idx=elbow_idx, **dict(picked))


def report_neuron(
    neuron_idx: int,
    polarity: Polarity,
    selection: HitSelection,
    rows: Float[torch.Tensor, "n_hits hidden"],
    sample_ids: list[str],
    weight: torch.Tensor,
    lrp_model: torch.nn.Module,
    source: SampleSource,
    presenter: ClusterPresenter,
    args: argparse.Namespace,
) -> None:
    tag = f"neuron {neuron_idx} {polarity}"
    hdmd = normalized_rows(rows) if args.cluster_on == "input" else hadamard_from_rows(rows, weight, neuron_idx)
    with timed(f"{tag}: cluster"):
        labels = cluster_labels(hdmd, args.min_cluster_size, args.cluster_selection_method, args.min_samples)
    print(f"[{tag}] {len(set(labels) - {-1})} clusters")

    hits = NeuronClusterHits(
        model=lrp_model,
        source=source,
        presenter=presenter,
        sample_ids=sample_ids,
        layer_getter=layer_getter_for(args),
        neuron_idx=neuron_idx,
        token_idx=selection.token_idx,
        batch_idx=selection.batch_idx,
        labels=labels,
        hdmds=hdmd,
        hadamard_shape=near_square_shape(hdmd.shape[1]),
        threshold=selection.threshold,
        elbow_values=selection.elbow_values.numpy(),
        elbow_idx=selection.elbow_idx,
        bucket_edges=selection.bucket_edges,
        device=args.device,
    )
    with timed(f"{tag}: write report"):
        report_path = hits.write_report(
            args.out_dir,
            name=report_name(neuron_idx, polarity),
            attr_fn=compute_attnlrp_relevance,
            max_n=args.max_hits_per_cluster,
            min_uniq_images=args.min_uniq_samples_per_cluster,
        )
    print(f"[{tag}] report -> {report_path}")


def main() -> None:
    args = parse_args()
    neuron_idxs = args.neurons or list(range(args.neuron_start, args.neuron_end + 1))
    pipeline_start = time.perf_counter()

    with timed("load texts"):
        texts = load_texts(args)
    print(f"loaded {len(texts)} {args.unit}s")

    with timed("load model"):
        adapter = load_adapter(args)

    module = adapter.get_model()
    adapter.patch_for_attn_lrp()  # before NNsight wraps: it stores each module's forward at wrap time
    model = NNsight(module).to(args.device)
    source = adapter.make_source(texts)
    clustered_label = "normalised inputs" if args.cluster_on == "input" else "hadamard products"
    presenter = adapter.make_presenter(source, clustered_label)
    capture = NeuronCapture(model, source, layer_getter_for(args), neuron_idxs, args.batch_size, args.device)

    with timed("pass 1: scan"):
        scan = capture.scan()
    print(f"scanned {len(scan.sample_ids)} texts, activations {tuple(scan.activations.shape)}, "
          f"{int(scan.valid_mask.sum())}/{scan.valid_mask.numel()} positions unpadded")

    targets = [(n, polarity) for n in neuron_idxs for polarity in polarities_for(args)]
    selections = {t: select_hits(scan, *t, args) for t in targets}
    merged = merge_positions([(s.batch_idx, s.token_idx) for s in selections.values()], args.max_length)
    with timed("pass 2: gather"):
        gathered = capture.gather(merged.batch_idx, merged.token_idx)
    worst = capture.check_consistent(scan, gathered, merged.batch_idx, merged.token_idx)
    print(f"gathered {len(merged.batch_idx)} positions; pass 2 matches pass 1 to {worst:.2e}")

    model.to("cpu")
    empty_cache()
    weight = layer_getter_for(args)(module).weight


    for (neuron_idx, polarity), index_map in zip(targets, merged.index_maps):
        report_neuron(
            neuron_idx, polarity, selections[(neuron_idx, polarity)], gathered.inputs[index_map], scan.sample_ids,
            weight, module, source, presenter, args,
        )

    print(f"[total] {time.perf_counter() - pipeline_start:.2f}s")


if __name__ == "__main__":
    main()
