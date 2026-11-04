#!/usr/bin/env python
"""HDBSCAN clusters of one logit of the 1-layer attn-only model.

- A point is `x1 ⊙ W_U[token]`, where `x1` is the final residual.
  - Its dims sum to the logit minus the unembed bias.
- Hits are bucketed per tail, with signed polarity, as in `in_tfm.hit_selection`.
- Windows start at a story's BOS and do not overlap, as in training.

    python scripts/the_logit_clusters.py --checkpoint <dir> --tokens <stream.npy> --out-dir <dir>
"""

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, Int
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from in_tfm.clustering import cluster_labels
from in_tfm.hit_selection.selection import tail_budgets
from in_tfm.hit_selection.strategies.bucketed import bucketed_sample
from in_tfm.models.attn_only_2l.model import (
    AttnOnlyForCausalLM,
    load_trained_attn_only,
)


class Args(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)

    checkpoint: Path
    tokens: Path
    out_dir: Path
    token: str = " the"
    n_tokens: int = 4_000_000
    max_hits: int = 1_000_000
    n_buckets: int = 10
    min_bucket_size: int = 100
    min_cluster_size: int = 50
    min_samples: int | None = None
    batch_size: int = 64
    seed: int = 0


class Residuals(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    x1: Float[torch.Tensor, "n_pos d_model"]
    logit: Float[np.ndarray, "n_pos"]
    current: Int[np.ndarray, "n_pos"]
    next: Int[np.ndarray, "n_pos"]
    position: Int[np.ndarray, "n_pos"]


class SignedHits(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    idx: Int[np.ndarray, "n_hits"]
    """Indices into the flat positions, ascending within each tail."""
    tail: Int[np.ndarray, "n_hits"]
    """-1 negative, +1 positive."""
    bucket: Int[np.ndarray, "n_hits"]
    n_candidates: dict[str, int]


def stage(name: str, started: float) -> None:
    print(f"[{name}] {time.time() - started:.1f}s", flush=True)


def story_aligned_windows(
    stream: Int[np.ndarray, "n_tokens"], bos: int, seq_len: int, n_windows: int
) -> Int[np.ndarray, "n_windows seq_plus_1"]:
    """Greedy, so each window starts at the first story start past the previous window."""
    starts = np.flatnonzero(stream == bos)
    starts = starts[starts + seq_len + 1 <= len(stream)]
    picked: list[int] = []
    next_free = 0
    for start in starts:
        if start >= next_free:
            picked.append(int(start))
            next_free = start + seq_len
        if len(picked) == n_windows:
            break
    if len(picked) < n_windows:
        raise ValueError(f"stream holds only {len(picked)} windows, {n_windows} asked")
    return np.stack([stream[s : s + seq_len + 1] for s in picked]).astype(np.int64)


@torch.no_grad()
def final_residuals(
    model: AttnOnlyForCausalLM,
    windows: Int[np.ndarray, "n_windows seq_plus_1"],
    token_id: int,
    batch_size: int,
    device: str,
) -> Residuals:
    inputs = torch.from_numpy(windows[:, :-1])
    w_u = model.lm_head.weight[token_id].to(device)
    b_u = model.lm_head.bias[token_id].item()
    x1_batches = []
    for batch in inputs.split(batch_size):
        x1 = model.model(input_ids=batch.to(device)).last_hidden_state
        x1_batches.append(x1.reshape(-1, x1.shape[-1]).cpu())
    x1 = torch.cat(x1_batches)
    seq_len = inputs.shape[1]
    return Residuals(
        x1=x1,
        logit=(x1 @ w_u.cpu()).numpy() + b_u,
        current=windows[:, :-1].reshape(-1),
        next=windows[:, 1:].reshape(-1),
        position=np.tile(np.arange(seq_len), len(windows)),
    )


def signed_bucketed_hits(logit: Float[np.ndarray, "n_pos"], args: Args) -> SignedHits:
    """Each tail is bucketed on its own above 0, with the budget split by candidates."""
    signs = {"negative": -1, "positive": 1}
    n_candidates = {name: int((sign * logit > 0).sum()) for name, sign in signs.items()}
    budgets = tail_budgets(list(n_candidates.values()), args.max_hits)
    idx, tail, bucket = [], [], []
    for (name, sign), budget in zip(signs.items(), budgets):
        sample = bucketed_sample(
            sign * logit,
            args.n_buckets,
            budget,
            0.0,
            args.min_bucket_size,
            np.random.default_rng(args.seed),
        )
        idx.append(sample.kept_value_idx)
        tail.append(np.full(len(sample.kept_value_idx), sign))
        bucket.append(sample.group_of_kept)
        print(f"  {name}: {n_candidates[name]} candidates, kept {budget}")
    return SignedHits(
        idx=np.concatenate(idx),
        tail=np.concatenate(tail),
        bucket=np.concatenate(bucket),
        n_candidates=n_candidates,
    )


def top_tokens(
    ids: Int[np.ndarray, "n"], decode, k: int = 8
) -> list[tuple[str, float]]:
    counts = Counter(ids.tolist()).most_common(k)
    return [(decode(token), round(n / len(ids), 3)) for token, n in counts]


def cluster_summaries(
    labels: Int[np.ndarray, "n_hits"],
    hits: SignedHits,
    residuals: Residuals,
    token_id: int,
    decode,
) -> list[dict]:
    rows = []
    for label in np.unique(labels):
        members = hits.idx[labels == label]
        rows.append(
            {
                "label": int(label),
                "size": len(members),
                "positive_share": round(
                    float((hits.tail[labels == label] > 0).mean()), 3
                ),
                "mean_logit": round(float(residuals.logit[members].mean()), 3),
                "next_is_token": round(
                    float((residuals.next[members] == token_id).mean()), 3
                ),
                "median_position": int(np.median(residuals.position[members])),
                "current": top_tokens(residuals.current[members], decode),
                "next": top_tokens(residuals.next[members], decode),
            }
        )
    return sorted(rows, key=lambda row: -row["size"])


def print_summary(rows: list[dict], n_hits: int, k: int = 40) -> None:
    clusters = [row for row in rows if row["label"] >= 0]
    noise = sum(row["size"] for row in rows if row["label"] < 0)
    print(f"{len(clusters)} clusters, noise {noise / n_hits:.1%} of {n_hits}")
    for row in clusters[:k]:
        current = ", ".join(f"{t!r} {s:.2f}" for t, s in row["current"][:4])
        print(
            f"  c{row['label']:<5} n={row['size']:<7} pos={row['positive_share']:.2f} "
            f"logit={row['mean_logit']:+.2f} next=the {row['next_is_token']:.2f} | {current}"
        )


def main() -> None:
    args = Args()  # ty: ignore[missing-argument]
    started = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_trained_attn_only(args.checkpoint)
    model = model.to(device)
    [token_id] = tokenizer.encode(args.token)
    seq_len = model.config.max_position_embeddings
    stream = np.load(args.tokens)
    windows = story_aligned_windows(
        stream, tokenizer.bos_token_id, seq_len, args.n_tokens // seq_len
    )
    stage(f"windows {windows.shape}", started)

    residuals = final_residuals(model, windows, token_id, args.batch_size, device)
    stage(f"forward {len(residuals.logit)} positions", started)

    hits = signed_bucketed_hits(residuals.logit, args)
    stage(f"hits {len(hits.idx)}", started)

    w_u = model.lm_head.weight[token_id].detach().cpu()
    points = (residuals.x1[hits.idx] * w_u).numpy()
    labels = cluster_labels(
        points,
        args.min_cluster_size,
        "leaf",
        args.min_samples,
        "euclidean",
        normalise=False,
    )
    stage("cluster", started)

    decode = tokenizer.decode
    rows = cluster_summaries(labels, hits, residuals, token_id, decode)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_dir / "hits.npz",
        idx=hits.idx,
        tail=hits.tail,
        bucket=hits.bucket,
        labels=labels,
        logit=residuals.logit[hits.idx],
        current=residuals.current[hits.idx],
        next=residuals.next[hits.idx],
        position=residuals.position[hits.idx],
    )
    meta = {
        "args": json.loads(args.model_dump_json()),
        "token_id": token_id,
        "n_positions": len(residuals.logit),
        "n_candidates": hits.n_candidates,
        "clusters": rows,
    }
    (args.out_dir / "clusters.json").write_text(json.dumps(meta, indent=1))
    print_summary(rows, len(hits.idx))
    stage("done, total", started)


if __name__ == "__main__":
    main()
