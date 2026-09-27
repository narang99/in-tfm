#!/usr/bin/env python
"""Occlusion test: which way of selecting high-activation q_proj hits finds positions that matter?

Selects hits twice, as run_llm_neuron_report.py does - by the raw q_proj output and by the
q_norm output (`--select-on proj` / `norm`) - then zeroes the neuron's coordinate at the
q_norm output for each set of hits and measures how much the next-token loss changes.
See in_tfm.occlusion for why q_norm is where the zeroing happens.

Reuses the corpus, model loading and hit selection of run_llm_neuron_report.py, so the hits
are the same ones that script would report for the same flags.
"""

import argparse
import time
from collections.abc import Callable
from pathlib import Path

from nnsight import NNsight
from pydantic import BaseModel

from in_tfm.device import default_device
from in_tfm.layers import q_norm_getter, q_proj_getter
from in_tfm.neuron_capture import NeuronCapture, ScanResult
from in_tfm.occlusion import OcclusionRunner, ScoredHits, SetEffect, build_hit_sets, shared_positions
from in_tfm.sources import TextSource
from run_llm_neuron_report import HitSelection, load_model, load_texts, select_hits


class NeuronOcclusion(BaseModel):
    neuron_idx: int
    n_hits_per_set: int
    raw_threshold: float
    norm_threshold: float
    n_raw_hits: int
    n_norm_hits: int
    n_shared_in_top: int
    """How many of `raw_top` are also in `norm_top`: how far apart the two selections are."""
    baseline_repeat_diff: float
    effects: list[SetEffect]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="google/gemma-3-270m")
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--unit", choices=["paragraph", "article"], default="article")
    parser.add_argument("--min-article-chars", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4, help="for the two scans; the occlusion runs one sample at a time")
    parser.add_argument("--layer-idx", type=int, default=11)
    parser.add_argument("--neurons", type=int, nargs="+", default=[90, 300])
    parser.add_argument("--n-hits", type=int, default=1000, help="positions zeroed per set, capped by the smaller selection")
    parser.add_argument("--n-random-controls", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("occlusion_results"))
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()
    # select_hits subsamples above this many hits, which would drop strong hits at random
    # before the top-n is taken
    args.max_hits = 10**9
    return args


def scan_for(model: NNsight, source: TextSource, neurons: list[int], args: argparse.Namespace, on_norm: bool) -> ScanResult:
    capture = NeuronCapture(
        model,
        source,
        q_proj_getter(args.layer_idx),
        neurons,
        args.batch_size,
        args.device,
        head_norm_getter=q_norm_getter(args.layer_idx) if on_norm else None,
    )
    return capture.scan()


def scored_hits(scan: ScanResult, selection: HitSelection, neuron_idx: int) -> ScoredHits:
    column = scan.neuron_idxs.index(neuron_idx)
    activation = scan.activations[selection.batch_idx, selection.token_idx, column]
    return ScoredHits(
        batch_idx=selection.batch_idx, token_idx=selection.token_idx, activation=activation.numpy()
    )


def occlude_neuron(
    neuron_idx: int,
    raw_scan: ScanResult,
    norm_scan: ScanResult,
    runner_factory: Callable[[int], OcclusionRunner],
    args: argparse.Namespace,
) -> NeuronOcclusion:
    raw_selection = select_hits(raw_scan, neuron_idx, "positive", args)
    norm_selection = select_hits(norm_scan, neuron_idx, "positive", args)
    raw_hits = scored_hits(raw_scan, raw_selection, neuron_idx)
    norm_hits = scored_hits(norm_scan, norm_selection, neuron_idx)

    column = norm_scan.neuron_idxs.index(neuron_idx)
    sets = build_hit_sets(
        raw_hits,
        norm_hits,
        valid_mask=norm_scan.valid_mask.numpy(),
        norm_activations=norm_scan.activations[..., column].numpy(),
        n_hits=args.n_hits,
        n_random_controls=args.n_random_controls,
        seed=args.seed,
    )
    by_name = {s.name: s for s in sets}
    n_shared = shared_positions(by_name["raw_top"], by_name["norm_top"], args.max_length)

    runner = runner_factory(neuron_idx)
    repeat_diff = runner.check_baseline_is_repeatable()
    effects = []
    for hit_set in sets:
        start = time.perf_counter()
        effect = runner.effect_of(hit_set)
        effects.append(effect)
        print(
            f"[neuron {neuron_idx}] {effect.name:<10} n={effect.n_hits} samples={effect.n_samples_touched} "
            f"net={effect.net_loss_change:+.3f} per_hit={effect.net_loss_change_per_hit:+.5f} "
            f"abs={effect.abs_sample_loss_change:.3f}  ({time.perf_counter() - start:.0f}s)"
        )
    return NeuronOcclusion(
        neuron_idx=neuron_idx,
        n_hits_per_set=len(by_name["raw_top"].batch_idx),
        raw_threshold=raw_selection.threshold,
        norm_threshold=norm_selection.threshold,
        n_raw_hits=len(raw_hits),
        n_norm_hits=len(norm_hits),
        n_shared_in_top=n_shared,
        baseline_repeat_diff=repeat_diff,
        effects=effects,
    )


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    texts = load_texts(args)
    print(f"loaded {len(texts)} {args.unit}s")

    model, hf_model, tokenizer = load_model(args)
    source = TextSource(texts, tokenizer, hf_model.model.embed_tokens, args.max_length)

    raw_scan = scan_for(model, source, args.neurons, args, on_norm=False)
    norm_scan = scan_for(model, source, args.neurons, args, on_norm=True)
    print(f"scanned {len(raw_scan.sample_ids)} texts twice (raw q_proj, q_norm)")

    hf_model.to(args.device).eval()
    q_norm = hf_model.model.layers[args.layer_idx].self_attn.q_norm

    def runner_for(neuron_idx: int) -> OcclusionRunner:
        return OcclusionRunner(
            hf_model, q_norm, source, raw_scan.sample_ids, neuron_idx, hf_model.config.head_dim, args.device
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for neuron_idx in args.neurons:
        result = occlude_neuron(neuron_idx, raw_scan, norm_scan, runner_for, args)
        (args.out_dir / f"neuron_{neuron_idx}.json").write_text(result.model_dump_json(indent=2))
    print(f"[total] {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
