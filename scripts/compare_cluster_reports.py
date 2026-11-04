#!/usr/bin/env python
"""Two reports of one neuron, matched cluster to cluster by the cosine of their `mean.pt` prototypes.

- Each cluster's match is the other report's cluster with the closest mean.
- Shares are given per cluster and per clustered hit, since small clusters are the ones likely to vanish.

    python scripts/compare_cluster_reports.py --a <report dir> --b <report dir> --watch 519 304
"""

import json
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, Int
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

THRESHOLDS = [0.99, 0.95, 0.9, 0.8]


class Args(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)

    a: Path
    b: Path
    watch: list[int] = []
    """Clusters of `a` to show their matches for."""


class Report(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    ids: Int[np.ndarray, "k"]
    sizes: Int[np.ndarray, "k"]
    means: Float[np.ndarray, "k d"]
    n_hits: int


def load_report(report_dir: Path) -> Report:
    meta = json.loads((report_dir / "meta.json").read_text())
    ids = [int(cid) for cid in meta["clusters"]]
    means = [
        torch.load(report_dir / f"cluster_{cid}" / "mean.pt").numpy() for cid in ids
    ]
    return Report(
        ids=np.array(ids),
        sizes=np.array([meta["clusters"][str(cid)]["n_hits"] for cid in ids]),
        means=np.stack(means).astype(np.float64),
        n_hits=meta["n_hits"],
    )


def cosines(a: Report, b: Report) -> Float[np.ndarray, "ka kb"]:
    unit_a = a.means / np.linalg.norm(a.means, axis=1, keepdims=True)
    unit_b = b.means / np.linalg.norm(b.means, axis=1, keepdims=True)
    return unit_a @ unit_b.T


def describe(name: str, report: Report) -> None:
    clustered = int(report.sizes.sum())
    print(
        f"{name}: {len(report.ids)} clusters, {report.n_hits:,} hits, "
        f"{1 - clustered / report.n_hits:.1%} noise, median size {int(np.median(report.sizes))}"
    )


def match_shares(
    name: str, best: Float[np.ndarray, "k"], sizes: Int[np.ndarray, "k"]
) -> None:
    for threshold in THRESHOLDS:
        matched = best >= threshold
        print(
            f"  {name} best match >= {threshold}: {matched.mean():.1%} of clusters, "
            f"{sizes[matched].sum() / sizes.sum():.1%} of clustered hits"
        )


def main() -> None:
    args = Args()  # ty: ignore[missing-argument]
    a, b = load_report(args.a), load_report(args.b)
    describe("a", a)
    describe("b", b)
    cos = cosines(a, b)
    match_shares("a -> b", cos.max(1), a.sizes)
    match_shares("b -> a", cos.max(0), b.sizes)
    scale = b.n_hits / a.n_hits
    for cid in args.watch:
        row = int(np.flatnonzero(a.ids == cid)[0])
        order = np.argsort(-cos[row])[:3]
        matches = ", ".join(
            f"c{b.ids[j]} cos {cos[row, j]:.3f} n={b.sizes[j]}" for j in order
        )
        print(
            f"  a c{cid} n={a.sizes[row]} (expect ~{a.sizes[row] * scale:.0f} in b) -> {matches}"
        )


if __name__ == "__main__":
    main()
