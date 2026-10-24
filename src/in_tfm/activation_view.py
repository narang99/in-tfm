"""A neuron's activation on every token of every sample, filtered in the page by cluster and probability.

- Every sample is stored, not a sample of them, as compact data next to the page.
  - Tokens point into one vocabulary list, and activations are quantised to whole steps of the neuron's peak.
  - Each hit keeps every cluster it has a real probability for, so any cluster and threshold can be asked for.
- The page filters in the browser: a cluster, a minimum probability for it, a sort and a colouring.
  - The filters live in the URL hash, so the cluster report links straight to a cluster's examples.
  - The link checker treats a `#key=value` anchor as page state, so those links are checked for the file only.
- Text only for now: an image would need its activation grid drawn over the picture instead.
"""

import json
from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel, ConfigDict

from .bucket_map_image import cluster_colour
from .html_report import page, render_template
from .presenters.text.colored_tokens import display_text

EXAMPLES_PAGE = "examples.html"
EXAMPLES_DATA = "examples_data.js"
TOKEN_ACTIVATIONS = "token_activations.npy"
DISPLAY_IDS = "display_ids.npy"
TOKEN_VALID = "token_valid.npy"
ACTIVATION_STEPS = 127
"""Activations are stored as whole steps of peak / 127, signed, so each fits in a small integer."""
MIN_STORED_PROBABILITY = 0.02
"""A hit keeps each cluster at or above this, as a percentage. The page's slider starts above it."""


class TokenGrid(BaseModel):
    """The neuron's activation at every position of every sample, saved beside the space.

    - The space keeps only the hits, so this is what lets the page show each hit in its whole sample.
    - It is an extra of the space rather than part of it, so a space collected before it existed is still reused."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    activations: Float[np.ndarray, "n_samples seq"]
    display_ids: Int[np.ndarray, "n_samples seq"]
    valid: Bool[np.ndarray, "n_samples seq"]


def token_grid_saved(space_dir: Path) -> bool:
    return all((space_dir / name).exists() for name in (TOKEN_ACTIVATIONS, DISPLAY_IDS, TOKEN_VALID))


def save_token_grid(grid: TokenGrid, space_dir: Path) -> None:
    np.save(space_dir / TOKEN_ACTIVATIONS, grid.activations.astype(np.float32))
    np.save(space_dir / DISPLAY_IDS, grid.display_ids.astype(np.int32))
    np.save(space_dir / TOKEN_VALID, grid.valid)


def load_token_grid(space_dir: Path) -> TokenGrid:
    return TokenGrid(
        activations=np.load(space_dir / TOKEN_ACTIVATIONS), display_ids=np.load(space_dir / DISPLAY_IDS),
        valid=np.load(space_dir / TOKEN_VALID),
    )


class ClusterOption(BaseModel):
    """A choice in the page's filter: a cluster, or a group of clusters. `id` is its column of the probabilities."""

    id: int
    name: str
    n_hits: int
    colour: str
    href: str | None


class SampleRecord(BaseModel):
    id: str
    tokens: list[int]
    """Indices into the vocabulary, valid positions only."""
    acts: list[int]
    """Signed steps of peak / `ACTIVATION_STEPS`, one per token."""
    hits: list[tuple[int, list[tuple[int, int]]]]
    """(token position in `tokens`, [(cluster, probability in percent), ...]) for each hit in the sample."""


class ExamplesData(BaseModel):
    neuron_idx: int
    peak: float
    steps: int
    vocab: list[str]
    clusters: list[ClusterOption]
    samples: list[SampleRecord]


def quantised(values: Float[np.ndarray, "n"], peak: float) -> list[int]:
    if peak <= 0:
        return [0] * len(values)
    return np.clip(np.round(values / peak * ACTIVATION_STEPS), -ACTIVATION_STEPS, ACTIVATION_STEPS).astype(int).tolist()


def hit_clusters(probabilities: Float[np.ndarray, "n_clusters"]) -> list[tuple[int, int]]:
    """Every cluster the hit has a real probability for, most probable first, as whole percentages."""
    kept = [c for c in np.argsort(-probabilities) if probabilities[c] >= MIN_STORED_PROBABILITY]
    return [(int(c), int(round(100 * probabilities[c]))) for c in kept]


class Vocabulary:
    def __init__(self) -> None:
        self.words: list[str] = []
        self.index: dict[str, int] = {}

    def ids(self, tokens: list[str]) -> list[int]:
        return [self.id(token) for token in tokens]

    def id(self, token: str) -> int:
        if token not in self.index:
            self.index[token] = len(self.words)
            self.words.append(token)
        return self.index[token]


def sample_records(
    sample_ids: list[str], token_lists: list[list[str]], activations: Float[np.ndarray, "n_samples seq"],
    valid: Bool[np.ndarray, "n_samples seq"], batch_idx: Int[np.ndarray, "n_hits"], token_idx: Int[np.ndarray, "n_hits"],
    probabilities: Float[np.ndarray, "n_hits n_clusters"], peak: float, vocabulary: Vocabulary,
) -> list[SampleRecord]:
    """Hit positions are re-indexed to the valid tokens, since padding is not stored."""
    hits_of_sample: dict[int, list[int]] = {}
    for hit, sample in enumerate(batch_idx.tolist()):
        hits_of_sample.setdefault(sample, []).append(hit)
    records = []
    for i, sample_id in enumerate(sample_ids):
        positions = np.flatnonzero(valid[i])
        stored_at = {int(p): k for k, p in enumerate(positions)}
        hits = [
            (stored_at[int(token_idx[h])], hit_clusters(probabilities[h]))
            for h in hits_of_sample.get(i, []) if int(token_idx[h]) in stored_at
        ]
        records.append(SampleRecord(
            id=sample_id, tokens=vocabulary.ids([display_text(token_lists[i][p]) for p in positions]),
            acts=quantised(activations[i, positions], peak), hits=hits,
        ))
    return records


def cluster_options(probabilities: Float[np.ndarray, "n_hits n_clusters"], cluster_report: str) -> list[ClusterOption]:
    """Ordered by how many hits have the cluster as their most probable."""
    counts = np.bincount(probabilities.argmax(axis=1), minlength=probabilities.shape[1])
    return [
        ClusterOption(id=int(c), name=f"cluster {c}", n_hits=int(counts[c]), colour=hex_colour(cluster_colour(int(c))),
                      href=f"{cluster_report}#cluster-{c}")
        for c in np.argsort(-counts, kind="stable")
    ]


def hex_colour(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


class ClusterGroups(BaseModel):
    """Named groups of clusters, such as islands of the map. Every cluster is in exactly one group."""

    names: list[str]
    members: list[list[int]]


def grouped_probabilities(
    probabilities: Float[np.ndarray, "n_hits n_clusters"], groups: ClusterGroups
) -> Float[np.ndarray, "n_hits n_groups"]:
    """A group's probability is the sum of its clusters', so each hit's group probabilities still sum to 1."""
    placed = sorted(c for members in groups.members for c in members)
    if placed != list(range(probabilities.shape[1])):
        raise ValueError(f"every cluster must be in exactly one group, got {placed}")
    return np.stack([probabilities[:, members].sum(axis=1) for members in groups.members], axis=1)


def group_options(grouped: Float[np.ndarray, "n_hits n_groups"], groups: ClusterGroups) -> list[ClusterOption]:
    """A group has no section of its own to link to, so its hits link nowhere."""
    counts = np.bincount(grouped.argmax(axis=1), minlength=grouped.shape[1])
    return [
        ClusterOption(
            id=g, name=f"{name} (clusters {', '.join(map(str, sorted(groups.members[g])))})", n_hits=int(counts[g]),
            colour=hex_colour(cluster_colour(g)), href=None,
        )
        for g, name in enumerate(groups.names)
    ]


class ExamplesPage(BaseModel):
    """Which files the page is written to, and what it calls the things it filters on."""

    name: str = EXAMPLES_PAGE
    data_name: str = EXAMPLES_DATA
    group_word: str = "cluster"


def write_examples_page(
    neuron_dir: Path, neuron_idx: int, sample_ids: list[str], token_lists: list[list[str]],
    activations: Float[np.ndarray, "n_samples seq"], valid: Bool[np.ndarray, "n_samples seq"],
    batch_idx: Int[np.ndarray, "n_hits"], token_idx: Int[np.ndarray, "n_hits"],
    probabilities: Float[np.ndarray, "n_hits n_options"], options: list[ClusterOption], cluster_report: str,
    target: ExamplesPage = ExamplesPage(),
) -> Path:
    """`probabilities` has a column per option. `cluster_report` is the cluster report's index, relative to `neuron_dir`.

    The colour scale runs to the largest magnitude, so a neuron's negative tail is not clipped."""
    peak = float(np.abs(np.where(valid, activations, 0.0)).max())
    vocabulary = Vocabulary()
    records = sample_records(
        sample_ids, token_lists, activations, valid, batch_idx, token_idx, probabilities, peak, vocabulary,
    )
    data = ExamplesData(
        neuron_idx=neuron_idx, peak=peak, steps=ACTIVATION_STEPS, vocab=vocabulary.words,
        clusters=options, samples=records,
    )
    (neuron_dir / target.data_name).write_text("window.EXAMPLES = " + json.dumps(data.model_dump(), separators=(",", ":")) + ";\n")
    title = f"Neuron {neuron_idx}: activation on every token, by {target.group_word}"
    body = render_template(
        "activation_examples.html", title=title, data_src=target.data_name, cluster_report=cluster_report,
        n_samples=len(records), n_hits=len(batch_idx), group=target.group_word,
    )
    path = neuron_dir / target.name
    path.write_text(page(title, body))
    return path
