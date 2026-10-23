"""Pages of the hit buckets: a map of the sampled graph with the bucket starred, and numbered cards for some of its hits.

- A bucket page is the map with each of the bucket's sampled hits starred, so you see where it sits between the clusters.
- Below it, some of those hits are numbered on the map and shown as cards in the same order.
- A card is the hit drawn by the model's presenter, then the mean Hadamards of its most probable clusters.
  - A cluster fades toward grey the less likely the hit is to belong to it.
  - A hit in several buckets has one card, shared by their pages.
- Every cluster named on a page links to its section of the cluster report, and so do the peaks on the map.
  - A numbered star on the map links to its card.
"""

from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float, Int
from PIL import Image
from pydantic import BaseModel, ConfigDict

from .bucket_graph import SampleMap
from .bucket_map_image import HUGE_CANVAS, NORMAL_CANVAS, draw_base, draw_bucket, to_canvas
from .hit_buckets import Bucket
from .html_report import page, render_template
from .mixed_hits_report import Cell, cells, tile_url
from .neuron_report import AttrFn, NeuronClusterHits
from .pair_mixed_report import cluster_means, spread_by_share
from .presenters import ClusterArtifacts
from .run_config import BucketsConfig
from .viz import render_hadamard_tiles

SHOWN_CLUSTERS = 3
MIN_SHOWN_PROBABILITY = 0.05
INDEX_PAGE = "index.html"
BASE_MAP = "base_map.png"
HUGE_SUFFIX = "_huge"
PEAK_MARK_SIZE = 3.0
"""Percent of the map's width, a little larger than the drawn peak so it is easy to click."""
CARD_MARK_SIZE = 2.0


class ClusterTile(BaseModel):
    cluster_id: int
    src: str
    href: str


class MapMark(BaseModel):
    """A link placed over the map image, positioned in percent so it follows the image's scaling."""

    href: str
    title: str
    left: float
    top: float
    size: float


class HitCard(BaseModel):
    hit_html: str
    cells: list[Cell]
    clusters: list[ClusterTile]
    """The clusters of `cells`, in the same order."""


class BucketPage(BaseModel):
    name: str
    n_clusters: int
    n_hits: int
    n_starred: int
    page: str
    map_src: str
    huge_map_src: str
    cluster_tiles: list[ClusterTile]
    marks: list[MapMark]
    cards: list[HitCard]


class BucketReportInputs(BaseModel):
    """What every page needs, so the page writers take one argument instead of ten."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hits: NeuronClusterHits
    probabilities: Float[np.ndarray, "n_hits n_clusters"]
    confident: Bool[np.ndarray, "n_hits"]
    sample: SampleMap
    numbered_per_bucket: int
    attr_fn: AttrFn
    cluster_report: str
    """The cluster report's index, relative to `report_dir`."""
    report_dir: Path

    def cluster_href(self, cluster: int) -> str:
        return f"{self.cluster_report}#cluster-{cluster}"


def shown_clusters(probabilities: Float[np.ndarray, "n_clusters"]) -> list[int]:
    """The most probable clusters of a hit, always the first and then up to `SHOWN_CLUSTERS` of those above the minimum."""
    ranked = np.argsort(-probabilities)[:SHOWN_CLUSTERS]
    return [int(c) for i, c in enumerate(ranked) if i == 0 or probabilities[c] >= MIN_SHOWN_PROBABILITY]


def map_fractions(points: Float[np.ndarray, "m 2"]) -> Float[np.ndarray, "m 2"]:
    """Where each node is drawn, in percent of the map. The same at every canvas size."""
    return to_canvas(points, NORMAL_CANVAS) / NORMAL_CANVAS * 100


def peak_marks(inputs: BucketReportInputs) -> list[MapMark]:
    sample = inputs.sample
    at = map_fractions(sample.points)
    return [
        MapMark(href=inputs.cluster_href(c), title=f"cluster {c}", left=at[node, 0], top=at[node, 1], size=PEAK_MARK_SIZE)
        for c, node in sample.peak_nodes.items()
    ]


def card_marks(sample: SampleMap, numbered_nodes: list[int]) -> list[MapMark]:
    at = map_fractions(sample.points)
    return [
        MapMark(href=f"#card-{n}", title=f"card {n}", left=at[node, 0], top=at[node, 1], size=CARD_MARK_SIZE)
        for n, node in enumerate(numbered_nodes, start=1)
    ]


class ReportTiles(BaseModel):
    """Each cluster's mean Hadamard, once as a plain tile on disk, and as images for the faded card cells."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    images: dict[int, Image.Image]
    tiles: dict[int, ClusterTile]


def report_tiles(inputs: BucketReportInputs) -> ReportTiles:
    hits, probabilities = inputs.hits, inputs.probabilities
    cluster_ids = list(range(probabilities.shape[1]))
    means = cluster_means(hits.hdmds, probabilities.argmax(axis=1), inputs.confident, cluster_ids)
    images = dict(zip(cluster_ids, render_hadamard_tiles([m.reshape(hits.hadamard_shape) for m in means])))
    tiles_dir = inputs.report_dir / "bucket_tiles"
    tiles = {
        c: ClusterTile(cluster_id=c, src=tile_url(image, tiles_dir, f"cluster{c}.png"), href=inputs.cluster_href(c))
        for c, image in images.items()
    }
    return ReportTiles(images=images, tiles=tiles)


def render_hit_card(inputs: BucketReportInputs, hit: int, tiles: ReportTiles) -> HitCard:
    hits = inputs.hits
    artifacts = ClusterArtifacts(cluster_dir=inputs.report_dir / f"hit_{hit}")
    artifacts.cluster_dir.mkdir(exist_ok=True)
    hit_html = hits.presenter.render([hits.cluster_hit(hit, inputs.attr_fn)], artifacts, hits.hadamard_shape)
    shown = shown_clusters(inputs.probabilities[hit])
    hit_cells = cells(
        inputs.probabilities[hit, shown], [tiles.images[c] for c in shown], inputs.report_dir / "bucket_tiles", hit,
    )
    return HitCard(hit_html=hit_html, cells=hit_cells, clusters=[tiles.tiles[c] for c in shown])


def numbered_hits(
    starred_hits: Int[np.ndarray, "k"], probabilities: Float[np.ndarray, "n_hits n_clusters"], bucket: Bucket, n: int,
) -> list[int]:
    """Spread along the probability of the bucket's leading cluster, as the sample was."""
    return [int(h) for h in spread_by_share(starred_hits, probabilities[starred_hits, bucket.clusters[0]], n)]


def save_bucket_maps(
    bases: tuple[Image.Image, Image.Image], sample: SampleMap, starred: list[int], numbered: list[int], slug: str,
    report_dir: Path,
) -> None:
    for base, suffix in zip(bases, ("", HUGE_SUFFIX)):
        draw_bucket(base, sample.points, starred, numbered).save(report_dir / "maps" / f"{slug}{suffix}.png")


def write_bucket_page(
    inputs: BucketReportInputs, bucket: Bucket, bases: tuple[Image.Image, Image.Image], tiles: ReportTiles,
    card_cache: dict[int, HitCard],
) -> BucketPage:
    """Every paged bucket has hits on the map, since the sample takes some from each."""
    sample = inputs.sample
    node_of_hit = {int(hit): node for node, hit in enumerate(sample.hits)}
    starred_hits = np.array([h for h in bucket.hits if h in node_of_hit])
    numbered = numbered_hits(starred_hits, inputs.probabilities, bucket, inputs.numbered_per_bucket)
    for hit in numbered:
        if hit not in card_cache:
            card_cache[hit] = render_hit_card(inputs, hit, tiles)

    numbered_nodes = [node_of_hit[h] for h in numbered]
    save_bucket_maps(bases, sample, [node_of_hit[int(h)] for h in starred_hits], numbered_nodes, bucket.slug, inputs.report_dir)
    result = BucketPage(
        name=bucket.name, n_clusters=len(bucket.clusters), n_hits=len(bucket.hits), n_starred=len(starred_hits),
        page=f"{bucket.slug}.html", map_src=f"maps/{bucket.slug}.png", huge_map_src=f"maps/{bucket.slug}{HUGE_SUFFIX}.png",
        cluster_tiles=[tiles.tiles[c] for c in bucket.clusters],
        marks=peak_marks(inputs) + card_marks(sample, numbered_nodes),
        cards=[card_cache[h] for h in numbered],
    )
    title = f"Neuron {inputs.hits.neuron_idx}: {bucket.name}"
    body = render_template("bucket_page.html", title=title, b=result, cluster_report=inputs.cluster_report)
    (inputs.report_dir / result.page).write_text(page(title, body))
    return result


def write_index(inputs: BucketReportInputs, pages: list[BucketPage], buckets: BucketsConfig) -> Path:
    sizes = sorted({p.n_clusters for p in pages})
    groups = [(n, [p for p in pages if p.n_clusters == n]) for n in sizes]
    title = f"Neuron {inputs.hits.neuron_idx}: buckets of hits by the clusters that carry them"
    body = render_template(
        "bucket_index.html", title=title, groups=groups, base_src=BASE_MAP, huge_base_src=f"base_map{HUGE_SUFFIX}.png",
        marks=peak_marks(inputs), n_sampled=len(inputs.sample.hits), buckets=buckets,
        cluster_report=inputs.cluster_report,
    )
    path = inputs.report_dir / INDEX_PAGE
    path.write_text(page(title, body))
    return path


def write_base_maps(sample: SampleMap, report_dir: Path) -> tuple[Image.Image, Image.Image]:
    bases = (
        draw_base(sample.points, sample.labels, sample.peak_nodes, NORMAL_CANVAS),
        draw_base(sample.points, sample.labels, sample.peak_nodes, HUGE_CANVAS),
    )
    bases[0].save(report_dir / BASE_MAP)
    bases[1].save(report_dir / f"base_map{HUGE_SUFFIX}.png")
    return bases


def write_bucket_report(inputs: BucketReportInputs, paged: list[Bucket], buckets: BucketsConfig) -> Path:
    """`paged` are the buckets that get a page, largest first."""
    for sub in ("maps", "bucket_tiles"):
        (inputs.report_dir / sub).mkdir(parents=True, exist_ok=True)
    tiles = report_tiles(inputs)
    bases = write_base_maps(inputs.sample, inputs.report_dir)
    card_cache: dict[int, HitCard] = {}
    pages = [write_bucket_page(inputs, bucket, bases, tiles, card_cache) for bucket in paged]
    return write_index(inputs, pages, buckets)
