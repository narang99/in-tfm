"""A sheet of the hits the cluster report leaves out because they are split between clusters.

- Hits are sampled from several entropy levels, so the sheet runs from mildly to heavily mixed.
- Each row is one hit, drawn by the model's presenter, then each shown cluster's mean Hadamard.
  - A cluster's tile fades toward grey the less likely the hit is to belong to it.
  - The fade is in the image itself, so it never blends with the page.
- It sits next to `index.html` as `mixed_hits.html`, and the index links to it.
"""

from pathlib import Path

import numpy as np
from jaxtyping import Float, Int
from PIL import Image, ImageEnhance
from pydantic import BaseModel

from .attribution import compute_attnlrp_relevance
from .html_report import page, render_template
from .neuron_report import MIXED_HITS_PAGE, AttrFn, NeuronClusterHits
from .presenters import ClusterArtifacts
from .viz import render_hadamard_tiles

ENTROPY_BINS = 4
HITS_PER_BIN = 6
SAMPLE_SEED = 0
MOST_FADED = 0.85
"""How far a cluster the hit is unlikely to belong to is drained of colour and pulled toward grey."""
GREY = (150, 150, 150)


class ClusterTile(BaseModel):
    cluster_id: int
    src: str


class Cell(BaseModel):
    src: str
    probability: str
    likely: bool


class MixedRow(BaseModel):
    entropy: float
    hit_html: str
    cells: list[Cell]


def sample_by_entropy(entropy: Float[np.ndarray, "n_hits"], min_entropy: float) -> Int[np.ndarray, "k"]:
    """Up to `HITS_PER_BIN` hits from each of `ENTROPY_BINS` equal-width bins above `min_entropy`, least mixed first.

    - Even bins keep the rarer, heavily mixed hits on the sheet, which a plain random sample would mostly miss."""
    mixed = np.flatnonzero(entropy >= min_entropy)
    if len(mixed) == 0:
        return mixed
    edges = np.linspace(min_entropy, entropy[mixed].max(), ENTROPY_BINS + 1)
    bins = np.clip(np.searchsorted(edges, entropy[mixed], side="right") - 1, 0, ENTROPY_BINS - 1)
    rng = np.random.default_rng(SAMPLE_SEED)
    picked = np.concatenate([
        rng.permutation(mixed[bins == b])[:HITS_PER_BIN] for b in range(ENTROPY_BINS)
    ])
    return picked[np.argsort(entropy[picked], kind="stable")]


def faded(tile: Image.Image, fade: float) -> Image.Image:
    muted = ImageEnhance.Color(tile).enhance(1 - fade)
    return Image.blend(muted, Image.new("RGB", tile.size, GREY), fade)


def tile_url(tile: Image.Image, out_dir: Path, name: str) -> str:
    tile.save(out_dir / name)
    return f"{out_dir.name}/{name}"


def cells(probabilities: Float[np.ndarray, "n_shown"], tiles: list[Image.Image], out_dir: Path, row: int) -> list[Cell]:
    """The fade is relative to the hit's most likely shown cluster, so that one is always at full colour."""
    top = probabilities.max() if probabilities.max() > 0 else 1.0
    return [
        Cell(
            src=tile_url(faded(tile, MOST_FADED * (1 - p / top)), out_dir, f"row{row}_cluster{column}.png"),
            probability=f"{p:.2f}", likely=bool(p == probabilities.max()),
        )
        for column, (p, tile) in enumerate(zip(probabilities, tiles))
    ]


def write_mixed_hits_page(
    hits: NeuronClusterHits,
    report_dir: Path,
    probabilities: Float[np.ndarray, "n_hits n_clusters"],
    entropy: Float[np.ndarray, "n_hits"],
    min_entropy: float,
    min_uniq_images: int,
    attr_fn: AttrFn = compute_attnlrp_relevance,
) -> Path:
    """`probabilities` are zero for noise. Columns are the clusters the report shows, largest first."""
    report_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir = report_dir / "mixed_hits"
    tiles_dir.mkdir(exist_ok=True)
    cluster_ids = hits.shown_cluster_ids(min_uniq_images)
    means = [hits.cluster_mean_hadamard(cid).reshape(hits.hadamard_shape) for cid in cluster_ids]
    tiles = render_hadamard_tiles(means) if means else []

    rows = []
    for row, hit in enumerate(sample_by_entropy(entropy, min_entropy)):
        artifacts = ClusterArtifacts(cluster_dir=report_dir / f"mixed_hit_{row}")
        artifacts.cluster_dir.mkdir(exist_ok=True)
        hit_html = hits.presenter.render([hits.cluster_hit(int(hit), attr_fn)], artifacts, hits.hadamard_shape)
        rows.append(MixedRow(
            entropy=float(entropy[hit]), hit_html=hit_html,
            cells=cells(probabilities[hit, cluster_ids], tiles, tiles_dir, row),
        ))
    body = render_template(
        "mixed_hits.html",
        title=f"Neuron {hits.neuron_idx}: hits split between clusters",
        n_mixed=int((entropy >= min_entropy).sum()), n_hits=len(entropy), min_entropy=min_entropy,
        bins=ENTROPY_BINS, per_bin=HITS_PER_BIN,
        clusters=[ClusterTile(cluster_id=cid, src=tile_url(t, tiles_dir, f"cluster{cid}.png"))
                  for cid, t in zip(cluster_ids, tiles)],
        rows=rows,
    )
    path = report_dir / MIXED_HITS_PAGE
    path.write_text(page(f"Neuron {hits.neuron_idx}: mixed hits", body))
    return path
