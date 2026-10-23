"""Pages of the hits split between two clusters, one page per pair of clusters and an `index.html` linking them.

- A hit belongs to the pair of its two most probable clusters, if the second one has a real share.
- Each pair is binned by the hit's entropy over every cluster, and each bin shows hits from mostly one cluster to mostly the other.
- A card is the hit, drawn by the model's presenter, then the two clusters' mean Hadamards faded by the hit's probability for each.
"""

from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float, Int
from PIL import Image
from pydantic import BaseModel, ConfigDict

from .attribution import compute_attnlrp_relevance
from .html_report import page, render_template
from .mixed_hits_report import Cell, cells, tile_url
from .neuron_report import AttrFn, NeuronClusterHits
from .presenters import ClusterArtifacts
from .viz import render_hadamard_tiles

MIN_SECOND_PROBABILITY = 0.1
MIN_HITS_PER_PAIR = 30
HITS_PER_BIN = 8
ENTROPY_EDGES = (0.5, 1.0, 1.5, 2.0)
INDEX_PAGE = "index.html"


class PairHits(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    clusters: tuple[int, int]
    hits: Int[np.ndarray, "k"]


class Card(BaseModel):
    entropy: float
    hit_html: str
    cells: list[Cell]


class BinSection(BaseModel):
    label: str
    n_in_bin: int
    cards: list[Card]


class PairSection(BaseModel):
    first: int
    second: int
    n_hits: int
    first_src: str
    second_src: str
    bins: list[BinSection]
    page: str


def two_most_probable(probabilities: Float[np.ndarray, "n_hits n_clusters"]) -> Int[np.ndarray, "n_hits 2"]:
    return np.argsort(probabilities, axis=1)[:, ::-1][:, :2]


def pairs_with_hits(probabilities: Float[np.ndarray, "n_hits n_clusters"]) -> list[PairHits]:
    """Pairs whose hits number at least `MIN_HITS_PER_PAIR`, most hits first. The smaller cluster id comes first."""
    top_two = two_most_probable(probabilities)
    second_share = probabilities[np.arange(len(probabilities)), top_two[:, 1]]
    mixed = np.flatnonzero(second_share >= MIN_SECOND_PROBABILITY)
    pair_of_hit = np.sort(top_two[mixed], axis=1)
    pairs, inverse, counts = np.unique(pair_of_hit, axis=0, return_inverse=True, return_counts=True)
    order = [i for i in np.argsort(-counts, kind="stable") if counts[i] >= MIN_HITS_PER_PAIR]
    return [PairHits(clusters=(int(pairs[i][0]), int(pairs[i][1])), hits=mixed[inverse.ravel() == i]) for i in order]


def entropy_bin(entropy: Float[np.ndarray, "k"]) -> Int[np.ndarray, "k"]:
    return np.digitize(entropy, ENTROPY_EDGES)


def entropy_bin_label(bin_idx: int) -> str:
    if bin_idx == 0:
        return f"below {ENTROPY_EDGES[0]} bits"
    if bin_idx == len(ENTROPY_EDGES):
        return f"{ENTROPY_EDGES[-1]} bits and above"
    return f"{ENTROPY_EDGES[bin_idx - 1]} to {ENTROPY_EDGES[bin_idx]} bits"


def share_of_first(
    probabilities: Float[np.ndarray, "n_hits n_clusters"], hits: Int[np.ndarray, "k"], pair: tuple[int, int]
) -> Float[np.ndarray, "k"]:
    first, second = probabilities[hits, pair[0]], probabilities[hits, pair[1]]
    return first / (first + second)


def spread_by_share(hits: Int[np.ndarray, "k"], share: Float[np.ndarray, "k"], n: int) -> Int[np.ndarray, "m"]:
    """`n` hits at evenly spaced ranks of the share, so a bin runs from mostly the second cluster to mostly the first."""
    ranked = hits[np.argsort(share, kind="stable")]
    if len(ranked) <= n:
        return ranked
    return ranked[np.linspace(0, len(ranked) - 1, n).round().astype(int)]


def cluster_means(
    hadamards: Float[np.ndarray, "n_hits hidden"], most_probable: Int[np.ndarray, "n_hits"],
    confident: Bool[np.ndarray, "n_hits"], cluster_ids: list[int],
) -> list[Float[np.ndarray, "hidden"]]:
    """The mean over the cluster's confident hits, or over all its hits when none is confident."""
    means = []
    for cid in cluster_ids:
        members = (most_probable == cid) & confident
        members = members if members.any() else most_probable == cid
        means.append(hadamards[members].mean(axis=0))
    return means


def pair_page_name(pair: PairHits) -> str:
    return f"pair_{pair.clusters[0]}_{pair.clusters[1]}.html"


def render_card(
    hits: NeuronClusterHits, hit: int, report_dir: Path, card_number: int, pair: PairHits, tiles: dict[int, Image.Image],
    probabilities: Float[np.ndarray, "n_hits n_clusters"], entropy: Float[np.ndarray, "n_hits"], attr_fn: AttrFn,
) -> Card:
    artifacts = ClusterArtifacts(cluster_dir=report_dir / f"pair_hit_{card_number}")
    artifacts.cluster_dir.mkdir(exist_ok=True)
    hit_html = hits.presenter.render([hits.cluster_hit(hit, attr_fn)], artifacts, hits.hadamard_shape)
    return Card(
        entropy=float(entropy[hit]), hit_html=hit_html,
        cells=cells(probabilities[hit, list(pair.clusters)], [tiles[c] for c in pair.clusters], report_dir / "pair_tiles", card_number),
    )


def write_pair_page(
    hits: NeuronClusterHits, pair: PairHits, report_dir: Path, first_card_number: int, tiles: dict[int, Image.Image],
    probabilities: Float[np.ndarray, "n_hits n_clusters"], entropy: Float[np.ndarray, "n_hits"], attr_fn: AttrFn,
) -> PairSection:
    share = share_of_first(probabilities, pair.hits, pair.clusters)
    bin_of_hit = entropy_bin(entropy[pair.hits])
    card_number, bins = first_card_number, []
    for bin_idx in range(len(ENTROPY_EDGES) + 1):
        in_bin = bin_of_hit == bin_idx
        if not in_bin.any():
            continue
        cards = []
        for hit in spread_by_share(pair.hits[in_bin], share[in_bin], HITS_PER_BIN):
            cards.append(render_card(hits, int(hit), report_dir, card_number, pair, tiles, probabilities, entropy, attr_fn))
            card_number += 1
        bins.append(BinSection(label=entropy_bin_label(bin_idx), n_in_bin=int(in_bin.sum()), cards=cards))
    first, second = pair.clusters
    tiles_dir = report_dir / "pair_tiles"
    section = PairSection(
        first=first, second=second, n_hits=len(pair.hits), bins=bins, page=pair_page_name(pair),
        first_src=tile_url(tiles[first], tiles_dir, f"cluster{first}.png"),
        second_src=tile_url(tiles[second], tiles_dir, f"cluster{second}.png"),
    )
    title = f"Neuron {hits.neuron_idx}: clusters {first} and {second}"
    body = render_template("pair_mixed_pair.html", title=title, s=section, per_bin=HITS_PER_BIN)
    (report_dir / section.page).write_text(page(title, body))
    return section


def write_pair_mixed_pages(
    hits: NeuronClusterHits,
    report_dir: Path,
    probabilities: Float[np.ndarray, "n_hits n_clusters"],
    entropy: Float[np.ndarray, "n_hits"],
    confident: Bool[np.ndarray, "n_hits"],
    attr_fn: AttrFn = compute_attnlrp_relevance,
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "pair_tiles").mkdir(exist_ok=True)
    pairs = pairs_with_hits(probabilities)
    cluster_ids = sorted({cid for pair in pairs for cid in pair.clusters})
    means = cluster_means(hits.hdmds, probabilities.argmax(axis=1), confident, cluster_ids)
    tiles = dict(zip(cluster_ids, render_hadamard_tiles([m.reshape(hits.hadamard_shape) for m in means])))

    sections, card_number = [], 0
    for pair in pairs:
        section = write_pair_page(hits, pair, report_dir, card_number, tiles, probabilities, entropy, attr_fn)
        card_number += sum(len(b.cards) for b in section.bins)
        sections.append(section)
    title = f"Neuron {hits.neuron_idx}: hits split between pairs of clusters"
    body = render_template(
        "pair_mixed_index.html", title=title, n_hits=len(entropy), min_second=MIN_SECOND_PROBABILITY,
        min_pair_hits=MIN_HITS_PER_PAIR, entries=sections,
    )
    path = report_dir / INDEX_PAGE
    path.write_text(page(title, body))
    return path
