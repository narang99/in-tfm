"""Turning a cluster's sampled hits into report artifacts.

The other modality-aware module besides `sources`. A presenter owns two decisions that differ
completely between images and text:

- how to collapse the attribution's feature axis (channels for an image, hidden for a token)
- what a rendered hit even is - a JPEG overlay, or highlighted text
"""

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np
import torch
from jaxtyping import Float
from transformers.image_processing_utils import BaseImageProcessor

from .dicom import inv_tfm
from .sources import SampleId
from .viz import mk_overlay, render_hadamard_tiles, save_image_grid, to_pil


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


@runtime_checkable
class ClusterPresenter(Protocol):
    def render(self, hits: Sequence[ClusterHit], cluster_dir: Path) -> str:
        """Writes artifact files into cluster_dir; returns the markdown fragment linking them."""
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
            f"![original]({name}/original.jpg)\n\n"
            f"![overlay]({name}/overlay.jpg)\n\n"
            f"<details>\n<summary>hadamard patterns</summary>\n\n"
            f"![hadamard]({name}/hadamard.jpg)\n\n"
            f"</details>\n"
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


class TextPresenter:
    """Renders each hit as its token in context, with per-token relevance.

    The image path collapses relevance over channels to get a 2D map; here it collapses over
    the hidden dimension to get one number per token, which is what "where did this neuron look"
    means for text.
    """

    def __init__(self, source, context_tokens: int = 12, top_tokens: int = 10) -> None:
        self.source = source
        self.context_tokens = context_tokens
        self.top_tokens = top_tokens

    def render(self, hits: Sequence[ClusterHit], cluster_dir: Path) -> str:
        lines = [self._hit_block(h) for h in hits]
        (cluster_dir / "hits.md").write_text("\n\n".join(lines))
        return (
            f"firing tokens: {self._firing_token_summary(hits)}\n\n"
            f"<details>\n<summary>{len(hits)} sampled hits in context</summary>\n\n"
            + "\n\n".join(lines)
            + f"\n\n</details>\n"
        )

    def _firing_token_summary(self, hits: Sequence[ClusterHit]) -> str:
        """What the cluster is actually grouping, in one line - the payoff of clustering at the
        language level rather than the pixel level.

        Counted over the sampled hits only, not every member of the cluster.
        """
        counts = Counter(self._firing_token(h) for h in hits)
        return ", ".join(
            f"`{tok}`x{n}" if n > 1 else f"`{tok}`"
            for tok, n in counts.most_common(self.top_tokens)
        )

    def _firing_token(self, hit: ClusterHit) -> str:
        tokens = self.source.token_strings(hit.sample_id)
        return tokens[hit.token_idx] if hit.token_idx < len(tokens) else "<out-of-range>"

    def _hit_block(self, hit: ClusterHit) -> str:
        tokens = self.source.token_strings(hit.sample_id)
        relevance = self._per_token_relevance(hit)
        lo = max(0, hit.token_idx - self.context_tokens)
        hi = min(len(tokens), hit.token_idx + self.context_tokens + 1)

        rendered = [
            f"**[{tokens[i]}]**" if i == hit.token_idx else tokens[i]
            for i in range(lo, hi)
        ]
        return (
            f"- sample `{hit.sample_id}` token {hit.token_idx}: "
            + " ".join(rendered).replace(chr(9601), " ")
            + f"\n  - top relevance: {self._top_relevance(tokens, relevance)}"
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
        return ", ".join(f"`{tokens[i]}`({relevance[i]:+.2f})" for i in order)
