"""Pages of the hit buckets: a map of the sampled graph with the bucket starred, and numbered cards for some of its hits.

- A bucket page is the map with each of the bucket's sampled hits starred, so you see where it sits between the clusters.
- Below it, up to `NUMBERED_PER_BUCKET` of those hits are numbered on the map and shown as cards in the same order.
- A card is the hit drawn by the model's presenter, then the mean Hadamards of its most probable clusters.
  - A cluster fades toward grey the less likely the hit is to belong to it.
  - A hit in several buckets has one card, shared by their pages.
"""

import re
from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float, Int
from PIL import Image
from pydantic import BaseModel

from .attribution import compute_attnlrp_relevance
from .bucket_graph import SampleMap
from .bucket_map_image import draw_base, draw_bucket
from .hit_buckets import Bucket
from .html_report import page, render_template
from .mixed_hits_report import Cell, cells, tile_url
from .neuron_report import AttrFn, NeuronClusterHits
from .pair_mixed_report import cluster_means, spread_by_share
from .presenters import ClusterArtifacts
from .viz import render_hadamard_tiles

NUMBERED_PER_BUCKET = 6
SHOWN_CLUSTERS = 3
MIN_SHOWN_PROBABILITY = 0.05
INDEX_PAGE = "index.html"
BASE_MAP = "base_map.png"


class HitCard(BaseModel):
    hit_html: str
    cluster_ids: list[int]
    cells: list[Cell]


class BucketPage(BaseModel):
    name: str
    n_clusters: int
    n_hits: int
    n_starred: int
    page: str
    map_src: str
    cluster_tiles: list[tuple[int, str]]
    cards: list[HitCard]


def bucket_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def shown_clusters(probabilities: Float[np.ndarray, "n_clusters"]) -> list[int]:
    """The most probable clusters of a hit, always the first and then up to `SHOWN_CLUSTERS` of those above the minimum."""
    ranked = np.argsort(-probabilities)[:SHOWN_CLUSTERS]
    return [int(c) for i, c in enumerate(ranked) if i == 0 or probabilities[c] >= MIN_SHOWN_PROBABILITY]


def render_hit_card(
    hits: NeuronClusterHits, hit: int, report_dir: Path, tiles: dict[int, Image.Image],
    probabilities: Float[np.ndarray, "n_hits n_clusters"], attr_fn: AttrFn,
) -> HitCard:
    artifacts = ClusterArtifacts(cluster_dir=report_dir / f"hit_{hit}")
    artifacts.cluster_dir.mkdir(exist_ok=True)
    hit_html = hits.presenter.render([hits.cluster_hit(hit, attr_fn)], artifacts, hits.hadamard_shape)
    shown = shown_clusters(probabilities[hit])
    return HitCard(
        hit_html=hit_html, cluster_ids=shown,
        cells=cells(probabilities[hit, shown], [tiles[c] for c in shown], report_dir / "bucket_tiles", hit),
    )


def numbered_hits(
    starred_hits: Int[np.ndarray, "k"], probabilities: Float[np.ndarray, "n_hits n_clusters"], bucket: Bucket
) -> list[int]:
    """Spread along the probability of the bucket's leading cluster, as the sample was."""
    picked = spread_by_share(starred_hits, probabilities[starred_hits, bucket.clusters[0]], NUMBERED_PER_BUCKET)
    return [int(h) for h in picked]


def write_bucket_page(
    bucket: Bucket, sample: SampleMap, base: Image.Image, hits: NeuronClusterHits, report_dir: Path,
    tiles: dict[int, Image.Image], probabilities: Float[np.ndarray, "n_hits n_clusters"], attr_fn: AttrFn,
    card_cache: dict[int, HitCard],
) -> BucketPage | None:
    node_of_hit = {int(hit): node for node, hit in enumerate(sample.hits)}
    starred_hits = np.array([h for h in bucket.hits if h in node_of_hit])
    if len(starred_hits) == 0:
        return None
    numbered = numbered_hits(starred_hits, probabilities, bucket)
    for hit in numbered:
        if hit not in card_cache:
            card_cache[hit] = render_hit_card(hits, hit, report_dir, tiles, probabilities, attr_fn)

    slug = bucket_slug(bucket.name)
    map_path = report_dir / "maps" / f"{slug}.png"
    draw_bucket(
        base, sample.points, [node_of_hit[int(h)] for h in starred_hits], [node_of_hit[h] for h in numbered]
    ).save(map_path)
    result = BucketPage(
        name=bucket.name, n_clusters=len(bucket.clusters), n_hits=len(bucket.hits), n_starred=len(starred_hits),
        page=f"{slug}.html", map_src=f"maps/{slug}.png",
        cluster_tiles=[(c, tile_url(tiles[c], report_dir / "bucket_tiles", f"cluster{c}.png")) for c in bucket.clusters],
        cards=[card_cache[h] for h in numbered],
    )
    title = f"Neuron {hits.neuron_idx}: {bucket.name}"
    (report_dir / result.page).write_text(page(title, render_template("bucket_page.html", title=title, b=result)))
    return result


def write_index(pages: list[BucketPage], hits: NeuronClusterHits, report_dir: Path, n_sampled: int, min_hits: int) -> Path:
    sizes = sorted({p.n_clusters for p in pages})
    groups = [(n, [p for p in pages if p.n_clusters == n]) for n in sizes]
    title = f"Neuron {hits.neuron_idx}: buckets of hits by the clusters that carry them"
    body = render_template(
        "bucket_index.html", title=title, groups=groups, base_src=BASE_MAP, n_sampled=n_sampled, min_hits=min_hits,
    )
    path = report_dir / INDEX_PAGE
    path.write_text(page(title, body))
    return path


def write_bucket_report(
    hits: NeuronClusterHits, report_dir: Path, buckets: list[Bucket], sample: SampleMap,
    probabilities: Float[np.ndarray, "n_hits n_clusters"], confident: Bool[np.ndarray, "n_hits"], min_hits: int,
    attr_fn: AttrFn = compute_attnlrp_relevance,
) -> Path:
    for sub in ("maps", "bucket_tiles"):
        (report_dir / sub).mkdir(parents=True, exist_ok=True)
    cluster_ids = list(range(probabilities.shape[1]))
    means = cluster_means(hits.hdmds, probabilities.argmax(axis=1), confident, cluster_ids)
    tiles = dict(zip(cluster_ids, render_hadamard_tiles([m.reshape(hits.hadamard_shape) for m in means])))

    base = draw_base(sample.points, sample.labels, sample.peak_nodes)
    base.save(report_dir / BASE_MAP)
    card_cache: dict[int, HitCard] = {}
    pages = []
    for bucket in buckets:
        if len(bucket.hits) < min_hits:
            continue
        written = write_bucket_page(bucket, sample, base, hits, report_dir, tiles, probabilities, attr_fn, card_cache)
        if written is not None:
            pages.append(written)
    return write_index(pages, hits, report_dir, len(sample.hits), min_hits)
