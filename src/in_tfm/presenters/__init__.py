"""Turning a cluster's sampled hits into report artifacts.

The other modality-aware module besides `sources`. A presenter owns two decisions that differ
completely between images and text:

- how to collapse the attribution's feature axis (channels for an image, hidden for a token)
- what a rendered hit even is - a JPEG overlay, or highlighted text
"""

import html
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np
import torch
from jaxtyping import Float
from transformers.image_processing_utils import BaseImageProcessor

from ..dicom import inv_tfm
from ..html_report import colored_tokens, details, display_text, page, symmetric_scale
from ..sources import SampleId
from ..viz import mk_overlay, render_hadamard_tiles, save_image_grid, to_pil


class ClusterHit(NamedTuple):
    """One sampled member of a cluster, with everything needed to render it."""

    sample_id: SampleId
    token_idx: int
    """Which position in the sequence crossed the threshold - the token a text report
    highlights, and the patch an image overlay is centred on."""
    relevance: Float[np.ndarray, "..."]
    """Unreduced, straight from compute_attnlrp_relevance - the presenter reduces it."""
    model_input: torch.Tensor
    """The leaf that produced it, for presenters that show the input alongside the overlay."""
    hadamard: Float[np.ndarray, "hidden"]
    """Token ids as fed, when the leaf is not invertible - see sources.ModelBatch."""
    display_ids: torch.Tensor | None = None


@runtime_checkable
class ClusterPresenter(Protocol):
    def render(self, hits: Sequence[ClusterHit], cluster_dir: Path) -> str:
        """Writes artifact files into cluster_dir; returns the html fragment linking them."""
        ...


class ImagePresenter:
    """Pixel-space overlays, as in the original DICOM reports."""

    def __init__(
        self, processor: BaseImageProcessor, act_shape: tuple[int, int], alpha: float = 0.7
    ) -> None:
        self.processor = processor
        self.act_shape = act_shape
        self.alpha = alpha

    def render(self, hits: Sequence[ClusterHit], cluster_dir: Path) -> str:
        originals = [to_pil(self._denormalized(h), (500, 500)) for h in hits]
        overlays = [
            mk_overlay(self._denormalized(h), self._relevance_map(h), (500, 500), alpha=self.alpha)
            for h in hits
        ]
        tiles = render_hadamard_tiles([h.hadamard.reshape(self.act_shape) for h in hits])

        save_image_grid(originals, cluster_dir / "original.jpg")
        save_image_grid(overlays, cluster_dir / "overlay.jpg")
        save_image_grid(tiles, cluster_dir / "hadamard.jpg")

        name = cluster_dir.name
        return (
            f'<img src="{name}/original.jpg" alt="original">\n'
            f'<img src="{name}/overlay.jpg" alt="overlay">\n'
            + details("hadamard patterns", f'<img src="{name}/hadamard.jpg" alt="hadamard">')
        )

    def _denormalized(self, hit: ClusterHit) -> np.ndarray:
        """The model input with the processor's mean/std undone - what the overlay is blended
        onto, so the two line up pixel for pixel."""
        return inv_tfm(self.processor, hit.model_input)

    def _relevance_map(self, hit: ClusterHit) -> torch.Tensor:
        """Collapse the channel axis of a (C, H, W) relevance array down to (H, W)."""
        relevance = hit.relevance
        if relevance.ndim == 4:
            relevance = relevance[0]
        return torch.from_numpy(relevance.sum(0))


class HitWindow(NamedTuple):
    """One hit's context slice, ready to shade."""

    tokens: list[str]
    relevance: Float[np.ndarray, "window"]
    """Index of the firing token within the slice, not within the sequence."""
    firing_pos: int
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

    def __init__(self, source, context_tokens: int = 12, top_tokens: int = 10) -> None:
        self.source = source
        self.context_tokens = context_tokens
        self.top_tokens = top_tokens

    def render(self, hits: Sequence[ClusterHit], cluster_dir: Path) -> str:
        windows = [self._window(h) for h in hits]
        vmax = symmetric_scale([w.scale_relevance for w in windows])
        blocks = "\n".join(
            self._hit_block(hit, window, vmax) for hit, window in zip(hits, windows)
        )
        (cluster_dir / "hits.html").write_text(page(f"{cluster_dir.name} hits", blocks))
        return (
            f'<p class="firing">firing tokens: {self._firing_token_summary(hits)}</p>\n'
            f'<p class="scale">shading is shared across these hits: '
            f"red {-vmax:.2f} to green {vmax:+.2f}, hover a token for its value</p>\n"
            + details(f"{len(hits)} sampled hits in context", blocks, start_open=True)
        )

    def _firing_token_summary(self, hits: Sequence[ClusterHit]) -> str:
        """What the cluster is actually grouping, in one line - the payoff of clustering at the
        language level rather than the pixel level.

        Counted over the sampled hits only, not every member of the cluster.
        """
        counts = Counter(self._firing_token(h) for h in hits)
        return ", ".join(
            f"{_code(tok)}x{n}" if n > 1 else _code(tok)
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
            + colored_tokens(window.tokens, window.relevance, vmax, firing_idx=window.firing_pos)
            + f'\n<div class="scale">top relevance over the whole sequence: '
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
        return ", ".join(f"{_code(tokens[i])}({relevance[i]:+.2f})" for i in order)


def _code(token: str) -> str:
    return f"<code>{html.escape(display_text(token))}</code>"
