"""Text rendering of a cluster's hits: the token heatmap that stands in for the image
path's pixel overlay.

The shading itself lives in `colored_tokens`; this module is only about which tokens get shaded,
and on what scale.
Each hit shows the text from the start up to the firing token.
The attention sink (position 0) counts toward the scale by default, and the report's toggle
switches to the view that excludes it.
"""

from collections import Counter
from collections.abc import Sequence

import numpy as np

from ...html_report import SINK_TOGGLE, page, render_template
from ...viz import render_hadamard_tiles
from ..artifacts import ClusterArtifacts
from ..base import ClusterHit, HadamardShape
from .colored_tokens import display_text, symmetric_scale, token_views
from .views import FiringChip, HitView, RelevanceChip, SinkViews, TextClusterView


HADAMARD_TILE_GAP = 32
"""Pixels between tiles in the composited grid, and around its edge. The default `compose_grid`
pad of 4 reads as one block, since the tiles are dark and the gap is black."""


class TextPresenter:
    """Renders each hit as its token in context, shaded by per-token relevance.

    The image path collapses relevance over channels to get a 2D map; here it collapses over
    the hidden dimension to get one number per token, which is what "where did this neuron look"
    means for text. The colored-token view is that 1D map drawn over the text itself, the way
    `mk_overlay` draws the 2D one over pixels.
    """

    def __init__(
        self,
        source,
        top_tokens: int = 10,
        clustered_label: str = "hadamard products",
    ) -> None:
        self.source = source
        self.clustered_label = clustered_label
        self.top_tokens = top_tokens

    def page_controls(self) -> str:
        return SINK_TOGGLE

    def render(
        self, hits: Sequence[ClusterHit], artifacts: ClusterArtifacts, hadamard_shape: HadamardShape
    ) -> str:
        view = self._cluster_view(hits, artifacts, hadamard_shape)
        hits_html = "\n".join(render_template("text_hit.html", hit=hit) for hit in view.hits)
        artifacts.path("hits.html").write_text(page(f"{artifacts.name} hits", hits_html))
        return render_template("text_cluster.html", view=view, hits_html=hits_html)

    def _cluster_view(
        self, hits: Sequence[ClusterHit], artifacts: ClusterArtifacts, hadamard_shape: HadamardShape
    ) -> TextClusterView:
        relevances = [self._relevance_up_to_firing(h) for h in hits]
        scale = SinkViews(
            with_sink=symmetric_scale(relevances),
            without_sink=symmetric_scale([r[1:] for r in relevances]),
        )
        return TextClusterView(
            firing_tokens=self._firing_token_summary(hits),
            scale=scale,
            hits=[self._hit_view(hit, relevance, scale) for hit, relevance in zip(hits, relevances)],
            clustered_label=self.clustered_label,
            hadamard_url=self._save_hadamard_tiles(hits, artifacts, hadamard_shape),
        )

    def _save_hadamard_tiles(
        self, hits: Sequence[ClusterHit], artifacts: ClusterArtifacts, hadamard_shape: HadamardShape
    ) -> str:
        """The vectors that were clustered, one tile per hit in the same order as the hits above.

        Tiles share one color scale (see `render_hadamard_tiles`), so brightness is comparable
        between hits. Tile width follows the shape's aspect ratio rather than being forced
        square, since a hidden size rarely factors into a square.
        """
        height, width = hadamard_shape
        tile_height = 150
        tiles = render_hadamard_tiles(
            [h.hadamard.reshape(hadamard_shape) for h in hits],
            size=(round(tile_height * width / height), tile_height),
        )
        return artifacts.save_grid(
            tiles, "hadamard.jpg", cols=4, pad=HADAMARD_TILE_GAP, border=HADAMARD_TILE_GAP
        )

    def _firing_token_summary(self, hits: Sequence[ClusterHit]) -> list[FiringChip]:
        """What the cluster is actually grouping, in one line - the payoff of clustering at the
        language level rather than the pixel level.

        Counted over the sampled hits only, not every member of the cluster.
        """
        counts = Counter(self._firing_token(h) for h in hits)
        return [
            FiringChip(text=_chip_text(tok), count=n)
            for tok, n in counts.most_common(self.top_tokens)
        ]

    def _tokens(self, hit: ClusterHit) -> list[str]:
        """Decoded from the ids that were actually fed, not from a fresh tokenization.

        The image path renders inv_tfm(model_input) - the tensor the model consumed. This is
        the same guarantee for text: whatever is printed next to a relevance number came from
        the forward pass that produced it, rather than from an encode that merely ought to
        match.
        """
        if hit.display_ids is None:
            raise ValueError(f"no display_ids on hit for sample {hit.sample_id!r}")
        return self.source.tokenizer.convert_ids_to_tokens(hit.display_ids)

    def _firing_token(self, hit: ClusterHit) -> str:
        tokens = self._tokens(hit)
        return tokens[hit.token_idx] if hit.token_idx < len(tokens) else "<out-of-range>"

    def _relevance_up_to_firing(self, hit: ClusterHit) -> np.ndarray:
        """The whole text from the start, since the firing token only sees what came before it."""
        return self._per_token_relevance(hit)[: hit.token_idx + 1]

    def _hit_view(
        self, hit: ClusterHit, relevance: np.ndarray, scale: SinkViews[float]
    ) -> HitView:
        tokens = self._tokens(hit)[: len(relevance)]
        return HitView(
            sample_id=str(hit.sample_id),
            token_idx=hit.token_idx,
            tokens=token_views(tokens, relevance, scale, firing_idx=len(tokens) - 1),
            top_relevance=SinkViews(
                with_sink=self._top_relevance(tokens, relevance, first=0),
                without_sink=self._top_relevance(tokens, relevance, first=1),
            ),
        )

    def _per_token_relevance(self, hit: ClusterHit) -> np.ndarray:
        """(batch, seq, hidden) -> (seq,): sum over hidden, the text analogue of summing an
        image's relevance over colour channels."""
        relevance = hit.relevance
        if relevance.ndim == 3:
            relevance = relevance[0]
        return relevance.sum(-1)

    def _top_relevance(
        self, tokens: list[str], relevance: np.ndarray, first: int
    ) -> list[RelevanceChip]:
        """Ranked by absolute relevance among positions from `first` on, so `first=1` leaves out
        the attention sink."""
        order = first + np.argsort(-np.abs(relevance[first:]))[: self.top_tokens]
        return [RelevanceChip(text=_chip_text(tokens[i]), value=float(relevance[i])) for i in order]


def _chip_text(token: str) -> str:
    """Stripped, since a chip has its own padding and a leading SentencePiece space would show
    as a stray gap; a bare-space token keeps a visible middle dot instead of vanishing."""
    return display_text(token).strip() or "\u00b7"
