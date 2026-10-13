"""Rendering of a modular addition cluster's hits: which of `a`, `b` and `=` the neuron fired on.

- Each hit is three tokens, with the firing one marked, and AttnLRP's relevance of each token to the neuron.
- Relevance is summed over the embedding, as the text path sums it over the hidden size.
"""

from collections import Counter
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel

from ..html_report import page, render_template
from ..viz import render_hadamard_tiles
from .artifacts import ClusterArtifacts
from .base import ClusterHit, HadamardShape

POSITION_NAMES = ("a", "b", "=")
EQUALS_TOKEN = 113
HADAMARD_TILE_GAP = 32


class TokenCell(BaseModel):
    text: str
    firing: bool
    relevance: float


class GrokHitView(BaseModel):
    sample_id: str
    token_idx: int
    tokens: list[TokenCell]
    share_on_firing_token: float
    """Absolute relevance on the firing token over the total. A value neuron reads one token only, so this should be 1."""


class Chip(BaseModel):
    text: str
    count: int


class GrokClusterView(BaseModel):
    firing_positions: list[Chip]
    firing_tokens: list[Chip]
    hits: list[GrokHitView]
    clustered_label: str
    hadamard_url: str


def token_text(token: int) -> str:
    return "=" if token == EQUALS_TOKEN else str(token)


class GrokPresenter:
    def __init__(self, clustered_label: str = "hadamard products", top_tokens: int = 10) -> None:
        self.clustered_label = clustered_label
        self.top_tokens = top_tokens

    def page_controls(self) -> str:
        return ""

    def render(self, hits: Sequence[ClusterHit], artifacts: ClusterArtifacts, hadamard_shape: HadamardShape) -> str:
        view = GrokClusterView(
            firing_positions=_counts([POSITION_NAMES[h.token_idx] for h in hits], limit=3),
            firing_tokens=_counts([token_text(_firing_token(h)) for h in hits], self.top_tokens),
            hits=[_hit_view(h) for h in hits],
            clustered_label=self.clustered_label,
            hadamard_url=self._save_hadamard_tiles(hits, artifacts, hadamard_shape),
        )
        hits_html = "\n".join(render_template("grok_hit.html", hit=hit) for hit in view.hits)
        artifacts.path("hits.html").write_text(page(f"{artifacts.name} hits", hits_html))
        return render_template("grok_cluster.html", view=view, hits_html=hits_html)

    def _save_hadamard_tiles(
        self, hits: Sequence[ClusterHit], artifacts: ClusterArtifacts, hadamard_shape: HadamardShape
    ) -> str:
        height, width = hadamard_shape
        tile_height = 150
        tiles = render_hadamard_tiles(
            [h.hadamard.reshape(hadamard_shape) for h in hits], size=(round(tile_height * width / height), tile_height)
        )
        return artifacts.save_grid(tiles, "hadamard.jpg", cols=4, pad=HADAMARD_TILE_GAP, border=HADAMARD_TILE_GAP)


def _counts(labels: list[str], limit: int) -> list[Chip]:
    return [Chip(text=text, count=count) for text, count in Counter(labels).most_common(limit)]


def _tokens(hit: ClusterHit) -> list[int]:
    if hit.display_ids is None:
        raise ValueError(f"no display_ids on hit for sample {hit.sample_id!r}")
    return [int(t) for t in hit.display_ids]


def _firing_token(hit: ClusterHit) -> int:
    return _tokens(hit)[hit.token_idx]


def _relevance_per_token(hit: ClusterHit) -> np.ndarray:
    relevance = hit.relevance[0] if hit.relevance.ndim == 3 else hit.relevance
    return relevance.sum(-1)


def _hit_view(hit: ClusterHit) -> GrokHitView:
    relevance = _relevance_per_token(hit)
    total = np.abs(relevance).sum()
    return GrokHitView(
        sample_id=str(hit.sample_id),
        token_idx=hit.token_idx,
        tokens=[
            TokenCell(text=token_text(t), firing=i == hit.token_idx, relevance=float(relevance[i]))
            for i, t in enumerate(_tokens(hit))
        ],
        share_on_firing_token=float(abs(relevance[hit.token_idx]) / total) if total > 0 else 0.0,
    )
