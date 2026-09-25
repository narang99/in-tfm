"""Text rendering of a cluster's hits: the token heatmap that stands in for the image
path's pixel overlay.

The shading itself lives in `colored_tokens`; this module is only about which slice of which
sequence gets shaded, and on what scale.
"""

import html
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
from jaxtyping import Float

from ...html_report import details, page
from ...viz import render_hadamard_tiles, save_image_grid
from ..base import ClusterHit, HadamardShape
from .colored_tokens import colored_tokens, display_text, symmetric_scale


HADAMARD_TILE_GAP = 32
"""Pixels between tiles in the composited grid, and around its edge. The default `compose_grid`
pad of 4 reads as one block, since the tiles are dark and the gap is black."""


class HitWindow(NamedTuple):
    """One hit's context slice, ready to shade."""

    tokens: list[str]
    relevance: Float[np.ndarray, "window"]
    firing_pos: int
    """Index of the firing token within the slice, not within the sequence."""
    starts_at_bos: bool

    @property
    def scale_relevance(self) -> Float[np.ndarray, "..."]:
        """The slice minus `<bos>`, for picking a color scale.

        `<bos>` carries the decoder's attention-sink relevance, an order of magnitude above
        every real token; leaving it in the scale would flatten the rest of the window to grey.
        It is still drawn, just clipped to the limit.
        """
        return self.relevance[1:] if self.starts_at_bos else self.relevance


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
        context_tokens: int = 12,
        top_tokens: int = 10,
        clustered_label: str = "hadamard products",
    ) -> None:
        self.source = source
        self.clustered_label = clustered_label
        self.context_tokens = context_tokens
        self.top_tokens = top_tokens

    def render(
        self, hits: Sequence[ClusterHit], cluster_dir: Path, hadamard_shape: HadamardShape
    ) -> str:
        windows = [self._window(h) for h in hits]
        vmax = symmetric_scale([w.scale_relevance for w in windows])
        blocks = "\n".join(
            self._hit_block(hit, window, vmax) for hit, window in zip(hits, windows)
        )
        (cluster_dir / "hits.html").write_text(page(f"{cluster_dir.name} hits", blocks))
        return (
            f'<div class="firing"><span class="label">firing tokens</span>'
            f"{self._firing_token_summary(hits)}</div>\n"
            f'<p class="scale">shading is shared across these hits: '
            f"red {-vmax:.2f} to green {vmax:+.2f}, hover a token for its value</p>\n"
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

    def _window(self, hit: ClusterHit) -> HitWindow:
        tokens = self._tokens(hit)
        relevance = self._per_token_relevance(hit)
        n = min(len(tokens), len(relevance))
        lo = max(0, hit.token_idx - self.context_tokens)
        hi = min(n, hit.token_idx + self.context_tokens + 1)
        return HitWindow(tokens[lo:hi], relevance[lo:hi], hit.token_idx - lo, starts_at_bos=lo == 0)

    def _hit_block(self, hit: ClusterHit, window: HitWindow, vmax: float) -> str:
        tokens = self._tokens(hit)
        return (
            '<div class="hit">\n'
            f'<div class="hit-tag">sample {html.escape(str(hit.sample_id))} '
            f"&middot; token {hit.token_idx}</div>\n"
            + colored_tokens(
                window.tokens,
                window.relevance,
                vmax,
                firing_idx=window.firing_pos,
                sink_idx=0 if window.starts_at_bos else None,
            )
            + f'\n<div class="top-rel"><span class="label">top relevance</span>'
            f"{self._top_relevance(tokens, self._per_token_relevance(hit))}</div>\n"
            "</div>"
        )

    def _per_token_relevance(self, hit: ClusterHit) -> np.ndarray:
        """(batch, seq, hidden) -> (seq,): sum over hidden, the text analogue of summing an
        image's relevance over colour channels."""
        relevance = hit.relevance
        if relevance.ndim == 3:
            relevance = relevance[0]
        return relevance.sum(-1)

    def _top_relevance(self, tokens: list[str], relevance: np.ndarray) -> str:
        n = min(len(tokens), len(relevance))
        order = np.argsort(-np.abs(relevance[:n]))[: self.top_tokens]
        return "".join(
            f'<span class="chip {"pos" if relevance[i] >= 0 else "neg"}">{_code(tokens[i])}'
            f'<span class="count">{relevance[i]:+.2f}</span></span>'
            for i in order
        )


def _code(token: str) -> str:
    """Stripped, since a chip has its own padding and a leading SentencePiece space would show
    as a stray gap; a bare-space token keeps a visible middle dot instead of vanishing."""
    text = display_text(token).strip()
    return f"<code>{html.escape(text or '\u00b7')}</code>"
