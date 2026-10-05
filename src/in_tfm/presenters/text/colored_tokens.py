"""A heatmap drawn over running text: one span per token, shaded by that token's value.

This is circuitsvis' `ColoredTokens` idea - shaded tokens, hover for the number - but emitted
as static html instead of rendered by a CDN javascript import. These reports get zipped off a
Colab box and read offline, and a report that needs the network to show its main content is not
worth the tooltip polish.

Colors keep the image overlays' red/green convention, so a green token and a green pixel mean
the same thing across both report types - see `token_cmap_light` for why the neutral midpoint
differs. The span class names are styled by `html_report.REPORT_CSS`.
"""

import html
from collections.abc import Sequence

import numpy as np
from jaxtyping import Float
from matplotlib.colors import Colormap

from ...colormaps import token_cmap_dark, token_cmap_light

SENTENCEPIECE_SPACE = "\u2581"
BYTE_LEVEL_BPE_SPACE = "\u0120"
BYTE_LEVEL_BPE_NEWLINE = "\u010a"
BYTE_LEVEL_BPE_TAB = "\u0109"


def symmetric_scale(values: Sequence[Float[np.ndarray, "n"]]) -> float:
    """One +/- limit shared by every token view that will be compared side by side, so relative
    shading stays honest across hits instead of each one renormalizing to its own maximum.

    Values outside the limit are clipped by `colored_tokens`, so a caller can leave an outlier
    out of `values` to keep it from flattening everything else.
    """
    peaks = [float(np.abs(v).max()) for v in values if v.size]
    return max(peaks) if peaks else 1.0


def display_text(token: str) -> str:
    """SentencePiece marks a leading space with U+2581, and byte-level BPE (GPT-2, GPT-NeoX) with
    U+0120 (`Ġ`), a newline with `Ċ` and a tab with `ĉ`.
    Spans render them literally, so show the characters they stand for."""
    spaced = token.replace(SENTENCEPIECE_SPACE, " ").replace(BYTE_LEVEL_BPE_SPACE, " ")
    return spaced.replace(BYTE_LEVEL_BPE_NEWLINE, "\n").replace(BYTE_LEVEL_BPE_TAB, "\t").replace("\n", "↵")


def colored_tokens(
    tokens: Sequence[str],
    values: Float[np.ndarray, "n"],
    vmax: float,
    vmax_without_first: float,
    firing_idx: int | None = None,
) -> str:
    """Tokens in reading order, each shaded by its value on a shared +/-`vmax` scale.

    Every span carries colors for two scales, and the stylesheet picks one:
    - the default `vmax`, which every token counts toward
    - `vmax_without_first`, for the view that excludes the attention sink (position 0), which is
      then drawn unshaded
    """
    spans = [
        _token_span(tok, float(val), vmax, vmax_without_first, firing=(i == firing_idx), first=(i == 0))
        for i, (tok, val) in enumerate(zip(tokens, values))
    ]
    return f'<div class="tokens">{"".join(spans)}</div>'


def _token_span(token: str, value: float, vmax: float, vmax_without_first: float, firing: bool, first: bool) -> str:
    classes = ["tok"] + (["tok-firing"] if firing else []) + (["tok-first"] if first else [])
    style = f"{_token_colors(value, vmax)};{_token_colors(value, vmax_without_first, prefix='x-')}"
    title = f"{display_text(token)}  {value:+.3f}"
    return (
        f'<span class="{" ".join(classes)}" style="{style}"'
        f' title="{html.escape(title, quote=True)}">{html.escape(display_text(token))}</span>'
    )


def _token_colors(value: float, vmax: float, prefix: str = "") -> str:
    """Light- and dark-mode colors as custom properties, picked by a media query in REPORT_CSS.

    The color has to live on the element (it is per token), but a media query cannot reach into
    an inline `background`, so the span ships both and the stylesheet chooses.
    """
    light = _hex(token_cmap_light, value, vmax)
    dark = _hex(token_cmap_dark, value, vmax)
    return (
        f"--{prefix}bg-l:{light};--{prefix}fg-l:{_readable_fg(light)};"
        f"--{prefix}bg-d:{dark};--{prefix}fg-d:{_readable_fg(dark)}"
    )


def _hex(cmap: Colormap, value: float, vmax: float) -> str:
    normed = 0.5 if vmax <= 0 else float(np.clip((value + vmax) / (2 * vmax), 0.0, 1.0))
    r, g, b, _ = cmap(normed)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def _readable_fg(bg_hex: str) -> str:
    r, g, b = (int(bg_hex[i : i + 2], 16) / 255 for i in (1, 3, 5))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#111" if luminance > 0.55 else "#fff"
