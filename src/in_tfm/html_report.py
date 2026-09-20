"""HTML rendering for neuron reports: the page shell, and a colored-token heatmap over text.

The token view is circuitsvis' `ColoredTokens` idea - one span per token, background shaded by
that token's value, hover for the number - but baked into static html rather than rendered by a
CDN javascript import. Reports are zipped off a Colab box and read offline, so a report that
needs the network to show its main content is not worth the tooltip polish.

Colors keep the image overlays' red/green convention, so a green token and a green pixel mean
the same thing across both report types - see `token_cmap_light` for why the neutral midpoint
differs.
"""

import html
from collections.abc import Sequence

import numpy as np
from jaxtyping import Float
from matplotlib.colors import Colormap

from .colormaps import token_cmap_dark, token_cmap_light

SENTENCEPIECE_SPACE = "▁"

REPORT_CSS = """
:root { color-scheme: light dark;
  --bg:#fff; --fg:#1a1a1a; --muted:#777; --line:#e3e3e3; --panel:#fafafa; }
@media (prefers-color-scheme: dark) { :root {
  --bg:#161616; --fg:#e6e6e6; --muted:#9a9a9a; --line:#333; --panel:#1e1e1e; } }
body { background:var(--bg); color:var(--fg); max-width:980px; margin:0 auto;
  padding:32px 20px 64px; line-height:1.5;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif; }
h1 { font-size:1.5rem; margin:0 0 4px; }
h2 { font-size:1.05rem; margin:0 0 10px; }
.meta { color:var(--muted); font-size:0.85rem; margin:0 0 24px; }
.cluster { border:1px solid var(--line); border-radius:8px; background:var(--panel);
  padding:14px 18px; margin:22px 0; }
.firing { font-size:0.85rem; margin:0 0 4px; }
.hit { padding:10px 0; border-top:1px solid var(--line); }
.hit:first-of-type { border-top:none; }
.hit-tag { font-size:0.65rem; font-weight:600; letter-spacing:0.04em; text-transform:uppercase;
  color:var(--muted); margin-bottom:4px; }
.tokens { line-height:2; font-size:0.85rem; }
.tok { display:inline-block; white-space:pre-wrap; padding:3px 0; border-radius:2px;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  background:var(--bg-l); color:var(--fg-l); }
@media (prefers-color-scheme: dark) { .tok { background:var(--bg-d); color:var(--fg-d); } }
.tok-firing { outline:2px solid var(--fg); outline-offset:1px; }
.scale { color:var(--muted); font-size:0.7rem; margin-top:4px; }
img { max-width:100%; border-radius:4px; }
details { margin-top:10px; }
details summary { cursor:pointer; color:var(--muted); font-size:0.85rem; }
"""


def page(title: str, body: str) -> str:
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(title)}</title>\n"
        f"<style>{REPORT_CSS}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n"
    )


def details(summary: str, body: str, start_open: bool = False) -> str:
    tag = "<details open>" if start_open else "<details>"
    return f"{tag}\n<summary>{html.escape(summary)}</summary>\n{body}\n</details>"
