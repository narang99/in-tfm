#!/usr/bin/env python
"""Writes every non-empty bucket of hits to `buckets.json`, from the arrays `collect_hit_space.py` saved.

    python scripts/hit_buckets.py /content/runs/mixed4e_55/space

- A hit is named `<sample id>:<token idx>`.
- Every non-empty bucket is written, whatever its size. The pages later keep only those with `--min-hits` or more.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from cluster_connectivity import hit_probabilities

from in_tfm.hit_buckets import Bucket, all_buckets


def hit_names(space: Path) -> list[str]:
    arrays = np.load(space / "hits.npz")
    sample_ids = arrays["sample_ids"]
    return [f"{sample_ids[b]}:{t}" for b, t in zip(arrays["batch_idx"], arrays["token_idx"])]


def bucket_record(bucket: Bucket, names: list[str]) -> dict:
    return {"name": bucket.name, "clusters": list(bucket.clusters), "hits": [names[i] for i in bucket.hits]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("space", type=Path)
    parser.add_argument("--min-hits", type=int, default=10, help="only for the printed summary")
    args = parser.parse_args()

    probabilities = hit_probabilities(args.space)
    names = hit_names(args.space)
    buckets = all_buckets(probabilities)
    (args.space / "buckets.json").write_text(json.dumps([bucket_record(b, names) for b in buckets]))

    big = [b for b in buckets if len(b.hits) >= args.min_hits]
    print(f"{len(probabilities)} hits, {probabilities.shape[1]} clusters, {len(buckets)} non-empty buckets, {len(big)} with >= {args.min_hits} hits")
    for bucket in big[:40]:
        print(f"  {len(bucket.hits):6d}  {bucket.name}")


if __name__ == "__main__":
    main()
