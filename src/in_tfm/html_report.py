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
.meta { color:var(--muted); font-size:0.85rem; margin:0 0 24px; }
.accent-0 { --accent:#3b82f6; } .accent-1 { --accent:#8b5cf6; } .accent-2 { --accent:#f59e0b; }
.accent-3 { --accent:#06b6d4; } .accent-4 { --accent:#ec4899; } .accent-5 { --accent:#64748b; }
.cluster { border:1px solid var(--line); border-left:6px solid var(--accent);
  border-radius:8px; background:var(--panel); margin:64px 0; }
.cluster-head { position:sticky; top:0; z-index:5; display:flex; align-items:baseline;
  gap:14px; padding:12px 18px; background:var(--accent); color:#fff;
  border-radius:2px 6px 0 0; box-shadow:0 2px 6px rgba(0,0,0,0.25); }
.cluster-badge { font-size:1.25rem; font-weight:700; }
.cluster-stats { font-size:0.85rem; opacity:0.9; }
.cluster-body { padding:14px 18px; }
.cluster-nav { display:flex; flex-wrap:wrap; gap:8px; margin:16px 0 0; }
.cluster-nav a { color:var(--fg); text-decoration:none; font-size:0.85rem; padding:3px 12px;
  border:1px solid var(--line); border-left:4px solid var(--accent); border-radius:6px; }
.cluster-nav a:hover { background:var(--panel); }
.cluster-nav .n { color:var(--muted); margin-left:8px; font-size:0.75rem; }
html { scroll-behavior:smooth; scroll-padding-top:16px; }
.firing, .top-rel { display:flex; flex-wrap:wrap; align-items:center; gap:6px; margin:0 0 6px; }
.label { color:var(--muted); font-size:0.65rem; font-weight:600; letter-spacing:0.04em;
  text-transform:uppercase; margin-right:2px; }
.chip { display:inline-flex; align-items:baseline; gap:5px; padding:1px 8px; font-size:0.8rem;
  border:1px solid var(--line); border-radius:999px; background:var(--bg); }
.chip code { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; white-space:pre; }
.chip .count { color:var(--muted); font-size:0.7rem; }
.chip.pos .count { color:#00A550; }
.chip.neg .count { color:#d33; }
.top-rel { margin:6px 0 0; }
.top-rel .chip { font-size:0.7rem; padding:0 6px; }
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
.tok-sink { background:transparent; color:var(--muted); border:1px dashed var(--line);
  font-size:0.7rem; padding:2px 4px; margin-right:6px; }
.scale { color:var(--muted); font-size:0.7rem; margin:2px 0 4px; }
img { max-width:100%; border-radius:4px; }
details > img { display:block; margin-top:8px; }
img.elbow { max-width:420px; }
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
