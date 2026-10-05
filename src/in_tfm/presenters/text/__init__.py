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


GAP_MARKER = "…"

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
    """Whether the slice opens with position 0, the attention sink.
    That is `<bos>` for Gemma, and the first word of the text for a model with no `<bos>`.
    False when the hit fires at position 0 itself: nothing after it can affect that activation,
    so it is the only token with any relevance and has to set the scale."""

    @property
    def scale_relevance(self) -> Float[np.ndarray, "..."]:
        """The slice minus the sink, for picking a color scale.

        Position 0 carries the decoder's attention-sink relevance, an order of magnitude above
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
        if lo == 0:
            firing_is_the_sink = hit.token_idx == 0
            return HitWindow(tokens[:hi], relevance[:hi], hit.token_idx, starts_at_bos=not firing_is_the_sink)
        return self._window_behind_sink(tokens, relevance, lo, hi, hit.token_idx)

    def _window_behind_sink(
        self, tokens: list[str], relevance: np.ndarray, lo: int, hi: int, token_idx: int
    ) -> HitWindow:
        """Position 0 is pulled in front of a window that does not reach it.
        It usually holds most of a hit's relevance, so a window without it looks unshaded
        while the top relevance chips name a token that is nowhere on screen.
        A gap marker with no value stands for the skipped tokens."""
        prefix_tokens = [tokens[0], GAP_MARKER]
        prefix_relevance = np.array([relevance[0], 0.0], dtype=relevance.dtype)
        return HitWindow(
            prefix_tokens + tokens[lo:hi],
            np.concatenate([prefix_relevance, relevance[lo:hi]]),
            token_idx - lo + len(prefix_tokens),
            starts_at_bos=True,
        )

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
            f"{self._top_relevance(tokens, self._per_token_relevance(hit), skip_sink=hit.token_idx != 0)}</div>\n"
            "</div>"
        )

    def _per_token_relevance(self, hit: ClusterHit) -> np.ndarray:
        """(batch, seq, hidden) -> (seq,): sum over hidden, the text analogue of summing an
        image's relevance over colour channels."""
        relevance = hit.relevance
        if relevance.ndim == 3:
            relevance = relevance[0]
        return relevance.sum(-1)

    def _top_relevance(self, tokens: list[str], relevance: np.ndarray, skip_sink: bool = True) -> str:
        """Ranked without position 0, which is shown separately as the sink: its relevance
        would otherwise sit first in every hit and push the real tokens out of the list.
        `skip_sink=False` ranks everything, for a hit that fires at position 0."""
        n = min(len(tokens), len(relevance))
        start = 1 if skip_sink else 0
        order = start + np.argsort(-np.abs(relevance[start:n]))[: self.top_tokens]
        chips = "".join(self._relevance_chip(tokens[i], relevance[i]) for i in order)
        if not skip_sink:
            return chips
        return chips + self._relevance_chip(tokens[0], relevance[0], extra_class="sink", label="sink ")

    def _relevance_chip(self, token: str, value: float, extra_class: str = "", label: str = "") -> str:
        sign = "pos" if value >= 0 else "neg"
        return (
            f'<span class="chip {sign} {extra_class}">{label}{_code(token)}'
            f'<span class="count">{value:+.2f}</span></span>'
        )


def _code(token: str) -> str:
    """Stripped, since a chip has its own padding and a leading SentencePiece space would show
    as a stray gap; a bare-space token keeps a visible middle dot instead of vanishing."""
    text = display_text(token).strip()
    return f"<code>{html.escape(text or '\u00b7')}</code>"
