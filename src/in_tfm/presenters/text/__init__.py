"""Text rendering of a cluster's hits: the token heatmap that stands in for the image
path's pixel overlay.

The shading itself lives in `colored_tokens`; this module is only about which tokens get shaded,
and on what scale.
Each hit shows the text from the start up to the firing token.
The attention sink (position 0) counts toward the scale by default, and the report's toggle
switches to the view that excludes it.
"""

import html
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from jaxtyping import Float

from ...html_report import SINK_TOGGLE, details, page
from ...viz import render_hadamard_tiles, save_image_grid
from ..base import ClusterHit, HadamardShape
from .colored_tokens import colored_tokens, display_text, symmetric_scale


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
        self, hits: Sequence[ClusterHit], cluster_dir: Path, hadamard_shape: HadamardShape
    ) -> str:
        relevances = [self._relevance_up_to_firing(h) for h in hits]
        vmax = symmetric_scale(relevances)
        vmax_without_first = symmetric_scale([r[1:] for r in relevances])
        blocks = "\n".join(
            self._hit_block(hit, relevance, vmax, vmax_without_first) for hit, relevance in zip(hits, relevances)
        )
        (cluster_dir / "hits.html").write_text(page(f"{cluster_dir.name} hits", blocks))
        return (
            f'<div class="firing"><span class="label">firing tokens</span>'
            f"{self._firing_token_summary(hits)}</div>\n"
            f'<p class="scale">shading is shared across these hits: '
            + _both_views(_scale_text(vmax), _scale_text(vmax_without_first))
            + ", hover a token for its value</p>\n"
            + details(f"{len(hits)} sampled hits in context", blocks, start_open=True)
            + "\n"
            + self._hadamard_section(hits, cluster_dir, hadamard_shape)
        )

    def _hadamard_section(
        self, hits: Sequence[ClusterHit], cluster_dir: Path, hadamard_shape: HadamardShape
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
        save_image_grid(
            tiles,
            cluster_dir / "hadamard.jpg",
            cols=4,
            pad=HADAMARD_TILE_GAP,
            border=HADAMARD_TILE_GAP,
        )
        return details(
            f"{self.clustered_label} (what was clustered)",
            f'<img src="{cluster_dir.name}/hadamard.jpg" alt="{self.clustered_label}">',
        )

    def _firing_token_summary(self, hits: Sequence[ClusterHit]) -> str:
        """What the cluster is actually grouping, in one line - the payoff of clustering at the
        language level rather than the pixel level.

        Counted over the sampled hits only, not every member of the cluster.
        """
        counts = Counter(self._firing_token(h) for h in hits)
        return "".join(
            f'<span class="chip">{_code(tok)}' + (f'<span class="count">&times;{n}</span>' if n > 1 else "") + "</span>"
            for tok, n in counts.most_common(self.top_tokens)
        )

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

    def _hit_block(
        self, hit: ClusterHit, relevance: np.ndarray, vmax: float, vmax_without_first: float
    ) -> str:
        tokens = self._tokens(hit)[: len(relevance)]
        return (
            '<div class="hit">\n'
            f'<div class="hit-tag">sample {html.escape(str(hit.sample_id))} '
            f"&middot; token {hit.token_idx}</div>\n"
            + colored_tokens(tokens, relevance, vmax, vmax_without_first, firing_idx=len(tokens) - 1)
            + '\n<div class="top-rel"><span class="label">top relevance</span>'
            + _both_views(
                self._top_relevance(tokens, relevance, first=0), self._top_relevance(tokens, relevance, first=1)
            )
            + "</div>\n</div>"
        )

    def _per_token_relevance(self, hit: ClusterHit) -> np.ndarray:
        """(batch, seq, hidden) -> (seq,): sum over hidden, the text analogue of summing an
        image's relevance over colour channels."""
        relevance = hit.relevance
        if relevance.ndim == 3:
            relevance = relevance[0]
        return relevance.sum(-1)

    def _top_relevance(self, tokens: list[str], relevance: np.ndarray, first: int) -> str:
        """Ranked by absolute relevance among positions from `first` on, so `first=1` leaves out
        the attention sink."""
        order = first + np.argsort(-np.abs(relevance[first:]))[: self.top_tokens]
        return "".join(
            f'<span class="chip {"pos" if relevance[i] >= 0 else "neg"}">{_code(tokens[i])}'
            f'<span class="count">{relevance[i]:+.2f}</span></span>'
            for i in order
        )


def _scale_text(vmax: float) -> str:
    return f"red {-vmax:.2f} to green {vmax:+.2f}"


def _both_views(with_sink: str, without_sink: str) -> str:
    """Both versions are in the page, and the stylesheet shows the one the report's sink toggle
    has selected."""
    return f'<span class="when-sink">{with_sink}</span><span class="when-no-sink">{without_sink}</span>'


def _code(token: str) -> str:
    """Stripped, since a chip has its own padding and a leading SentencePiece space would show
    as a stray gap; a bare-space token keeps a visible middle dot instead of vanishing."""
    text = display_text(token).strip()
    return f"<code>{html.escape(text or '\u00b7')}</code>"
