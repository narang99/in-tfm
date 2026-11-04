#!/usr/bin/env python
"""Residual dims, and so o_proj neurons, that carry one cluster of a logit report.

- Hits are ranked by cosine similarity to the cluster's `mean.pt`.
- For each of the top examples, the dims of use are the terms of its hadamard row that make up `share`
  of the row's sum, largest first.
  - Terms are signed by the row's sum, so a negative-logit cluster keeps its negative terms.
- The final list is the union over examples, with how many examples used each dim.

- Examples are spread through the hits near the mean, one per distinct context.

Needs `hits_cache/` from `the_logit_cluster_summary.py`.

    python scripts/o_proj_neurons_from_cluster.py --report-dir reports/tiny_attn_only_1l_the_logit --cluster 12
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, Int
from pydantic_settings import BaseSettings, SettingsConfigDict
from tokenizers import Tokenizer

PAD = -1


class Args(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)

    report_dir: Path
    cluster: int
    report_name: str = "neuron_263_signed"
    tokenizer: Path = Path("checkpoints/tiny_attn_only_1l/tokenizer.json")
    n_examples: int = 12
    band: float = 0.9
    """Examples are drawn from the hits at least this close to the mean."""
    share: float = 0.9
    context: int = 12


def cosine_to(
    rows: Float[np.ndarray, "n d"], mean: Float[np.ndarray, "d"]
) -> Float[np.ndarray, "n"]:
    norms = np.linalg.norm(rows, axis=1) * np.linalg.norm(mean)
    return rows @ mean / np.maximum(norms, 1e-12)


def dims_of_use(row: Float[np.ndarray, "d"], share: float) -> Int[np.ndarray, "k"]:
    """The largest same-signed terms whose sum reaches `share` of the positive part, the crossing term included."""
    terms = row * np.sign(row.sum())
    order = np.argsort(-terms)
    positive = terms[order].clip(min=0)
    before = np.cumsum(positive) - positive
    kept = (before < share * positive.sum()) & (positive > 0)
    return order[kept]


def context_text(
    ids: Int[np.ndarray, "max_length"], token_idx: int, tokenizer: Tokenizer, width: int
) -> str:
    def text(span) -> str:
        return tokenizer.decode(
            [int(i) for i in span if i != PAD], skip_special_tokens=False
        )

    before = ids[max(0, token_idx - width) : token_idx]
    following = ids[token_idx + 1 : token_idx + 2]
    return f"{text(before)}[[{text(ids[token_idx : token_idx + 1])}]] -> {text(following)!r}"


def spread_examples(
    ranked: Int[np.ndarray, "n"],
    cosine: Float[np.ndarray, "n"],
    hits,
    ids: Int[np.ndarray, "n_texts max_length"],
    args: "Args",
) -> Int[np.ndarray, "k"]:
    """Evenly spaced through the hits at or above `band`, one per distinct context.

    - TinyStories repeats whole stories, so the top hits by cosine are often one text many times.
    """
    seen: set[bytes] = set()
    distinct = []
    for i in ranked[cosine[ranked] >= args.band].tolist():
        b, t = int(hits["batch_idx"][i]), int(hits["token_idx"][i])
        key = ids[b, max(0, t - args.context) : t + 1].tobytes()
        if key not in seen:
            seen.add(key)
            distinct.append(i)
    picks = np.linspace(0, len(distinct) - 1, min(args.n_examples, len(distinct)))
    print(f"  {len(distinct)} distinct contexts at cos >= {args.band}")
    return np.array(distinct)[np.unique(picks.round().astype(int))]


def main() -> None:
    args = Args()  # ty: ignore[missing-argument]
    cache = args.report_dir / "hits_cache"
    rows = np.load(cache / "rows.npy", mmap_mode="r")
    labels = np.load(cache / "labels.npy")
    hits = np.load(cache / "hits.npz")
    ids = np.load(cache / "window_ids.npy")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    mean = (
        torch.load(
            args.report_dir / args.report_name / f"cluster_{args.cluster}" / "mean.pt"
        )
        .numpy()
        .astype(np.float32)
    )

    cosine = cosine_to(np.asarray(rows), mean)
    ranked = np.argsort(-cosine)
    in_cluster = labels == args.cluster
    print(f"cluster {args.cluster}: {int(in_cluster.sum())} hits")
    for threshold in [0.99, 0.95, 0.9, 0.8]:
        near = cosine >= threshold
        print(
            f"  cos >= {threshold}: {int(near.sum())} hits, "
            f"{(in_cluster[near].mean() if near.any() else 0):.1%} of them in the cluster"
        )

    counts: Counter[int] = Counter()
    examples = []
    for i in spread_examples(ranked, cosine, hits, ids, args).tolist():
        used = dims_of_use(np.asarray(rows[i]), args.share)
        counts.update(used.tolist())
        b, t = int(hits["batch_idx"][i]), int(hits["token_idx"][i])
        example = {
            "hit": i,
            "cosine": round(float(cosine[i]), 4),
            "label": int(labels[i]),
            "logit": round(float(hits["activation"][i]), 2),
            "text": context_text(ids[b], t, tokenizer, args.context),
            "dims": used.tolist(),
        }
        examples.append(example)
        print(
            f"  cos {example['cosine']:.3f} c{example['label']:<4} logit {example['logit']:+.2f} "
            f"{len(used):>3} dims | {example['text']}"
        )

    mean_dims = dims_of_use(mean, args.share).tolist()
    final = sorted(counts, key=lambda d: (-counts[d], d))
    print(f"mean vector: {len(mean_dims)} dims of use")
    print(f"union over {len(examples)} examples: {len(final)} dims")
    print("  dim:count", " ".join(f"{d}:{counts[d]}" for d in final))
    print(
        f"  in every example: {sorted(d for d in final if counts[d] == len(examples))}"
    )
    print(f"  union minus the mean's own dims: {sorted(set(final) - set(mean_dims))}")
    out = args.report_dir / f"o_proj_neurons_cluster_{args.cluster}.json"
    out.write_text(
        json.dumps(
            {
                "cluster": args.cluster,
                "share": args.share,
                "examples": examples,
                "mean_dims": mean_dims,
                "dim_counts": {d: counts[d] for d in final},
                "final": final,
            },
            indent=1,
        )
    )
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
