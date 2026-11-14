"""One page for HDBSCAN's merges: the condensed tree drawn as a dendrogram, with a cluster card at every node.

- Height is distance, so a merge sits at the distance where its two groups join, and leaves sit on the floor.
- Leaves are spread along x with the smaller group first, and a merge sits between its two groups.
- Hovering a node shows its card beside the graph and lights up everything under it. A click pins the card.
"""

import json
from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np
from jaxtyping import Float
from markupsafe import Markup, escape

from .hdbscan_merges import MergeNode

TOP_TOKENS = 4
LEAF_GAP = 16
PLOT_HEIGHT = 620
MARGIN_TOP, MARGIN_LEFT, MARGIN_BOTTOM = 24, 56, 110
DISTANCE_TICKS = 5
EOM = "#e0a800"
LEAF = "#3987e5"
INNER = "#8a8a8a"
LIT = "#eb6834"

STYLE = f"""
<style>
.merge-layout {{ display: flex; gap: 16px; align-items: flex-start; }}
.merge-graph {{ flex: 7 1 0; min-width: 0; overflow-x: auto; }}
.merge-panel {{ flex: 4 1 0; min-width: 0; position: sticky; top: 0; max-height: 100vh; overflow-y: auto; }}
@media (max-width: 900px) {{ .merge-layout {{ flex-direction: column; }} .merge-panel {{ position: static; }} }}
.merge-graph svg text {{ fill: {INNER}; font-size: 11px; }}
.merge-graph .edge {{ fill: none; stroke: {INNER}; stroke-opacity: 0.55; stroke-width: 1.2; }}
.merge-graph .edge.lit {{ stroke: {LIT}; stroke-opacity: 1; stroke-width: 2; }}
.merge-graph .node {{ cursor: pointer; stroke-width: 0; }}
.merge-graph .node.eom {{ stroke: {EOM}; stroke-width: 2.5; }}
.merge-graph .node.lit {{ fill: {LIT}; }}
.merge-graph .node.pinned {{ stroke: {LIT}; stroke-width: 3; }}
.merge-graph .tick {{ stroke: {INNER}; stroke-opacity: 0.2; }}
.node-card {{ display: none; }} .node-card.shown {{ display: block; }}
.badge {{ font-size: 11px; padding: 1px 6px; border-radius: 8px; margin-left: 4px; border: 1px solid currentColor; }}
.badge.eom {{ color: {EOM}; }} .badge.leaf {{ color: {LEAF}; }}
.merge-meta {{ opacity: 0.7; }}
.legend span {{ margin-right: 14px; }}
</style>
"""

SCRIPT = """
<script>
(() => {
  const under = JSON.parse(document.getElementById('merge-under').textContent);
  let pinned = null;
  function show(name, light = true) {
    document.querySelectorAll('.node-card.shown').forEach((c) => c.classList.remove('shown'));
    document.getElementById('card-' + name).classList.add('shown');
    const lit = new Set(light ? under[name] : []);
    document.querySelectorAll('.merge-graph .node').forEach((n) => n.classList.toggle('lit', lit.has(n.dataset.name)));
    document.querySelectorAll('.merge-graph .edge').forEach((e) => e.classList.toggle('lit', lit.has(e.dataset.child)));
  }
  document.querySelectorAll('.merge-graph .node').forEach((node) => {
    node.addEventListener('mouseenter', () => { if (!pinned) show(node.dataset.name); });
    node.addEventListener('click', () => {
      document.querySelectorAll('.merge-graph .node.pinned').forEach((n) => n.classList.remove('pinned'));
      pinned = pinned === node.dataset.name ? null : node.dataset.name;
      if (pinned) node.classList.add('pinned');
      show(node.dataset.name);
    });
  });
  show('N1', false);
})();
</script>
"""


def tops(nodes: dict[int, MergeNode]) -> list[int]:
    """The full tree has one top, its root. A trimmed tree has one per eom cluster, largest first."""
    return sorted(
        (n for n, node in nodes.items() if node.parent is None),
        key=lambda n: -len(nodes[n].members),
    )


def node_names(nodes: dict[int, MergeNode]) -> dict[int, str]:
    """N1 is the largest top, then depth first with the smaller group first, the order the graph spreads leaves in."""
    names: dict[int, str] = {}
    stack = tops(nodes)[::-1]
    while stack:
        node = stack.pop()
        names[node] = f"N{len(names) + 1}"
        stack.extend(sorted(nodes[node].children, key=lambda c: -len(nodes[c].members)))
    return names


def subtree(nodes: dict[int, MergeNode], node: int) -> list[int]:
    stack, out = [node], []
    while stack:
        current = stack.pop()
        out.append(current)
        stack.extend(nodes[current].children)
    return out


def x_positions(nodes: dict[int, MergeNode]) -> dict[int, float]:
    """Leaves one gap apart in depth-first order, smaller group first. A merge sits midway between its groups.

    Separate trees get one extra gap between them."""
    x: dict[int, float] = {}
    slot = [0]

    def place(node: int) -> float:
        kids = sorted(nodes[node].children, key=lambda c: len(nodes[c].members))
        if not kids:
            x[node] = slot[0] * LEAF_GAP
            slot[0] += 1
        else:
            x[node] = float(np.mean([place(k) for k in kids]))
        return x[node]

    for top in tops(nodes):
        place(top)
        slot[0] += 1
    return x


def node_heights(nodes: dict[int, MergeNode]) -> dict[int, float]:
    """A merge sits at the distance where its two groups join. Leaves sit on the floor, at 0."""
    return {n: node.merge_distance or 0.0 for n, node in nodes.items()}


def token_summary(tokens: Sequence[str], members: np.ndarray) -> str:
    counts = Counter(tokens[i] for i in members)
    return "  ".join(
        f"<code>{escape(repr(token))}</code> {count / len(members):.0%}"
        for token, count in counts.most_common(TOP_TOKENS)
    )


def top_token(tokens: Sequence[str], members: np.ndarray) -> str:
    return Counter(tokens[i] for i in members).most_common(1)[0][0]


def relation(nodes: dict[int, MergeNode], node: int, names: dict[int, str]) -> str:
    parent = nodes[node].parent
    return "" if parent is None else f" | under {names[parent]}"


def summary_line(
    node: MergeNode,
    name: str,
    from_parent: str,
    tokens: Sequence[str],
    activations: Float[np.ndarray, "n_hits"],
) -> str:
    joins = (
        f" | its two groups join at {node.merge_distance:.3f}"
        if node.merge_distance is not None
        else ""
    )
    badges = ('<span class="badge eom">eom cluster</span>' if node.is_eom else "") + (
        '<span class="badge leaf">leaf cluster</span>' if node.is_leaf else ""
    )
    median = float(np.median(activations[node.members]))
    return (
        f"<b>{name}</b> {len(node.members)} hits{badges} "
        f'<span class="merge-meta">{from_parent} | activation median {median:+.4f}{joins} '
        f"| {len(node.dropped)} drop out here</span><br>{token_summary(tokens, node.members)}"
    )


class Canvas:
    """Distance to pixels. The top of the plot is the root's merge distance."""

    def __init__(self, top_distance: float, width: float) -> None:
        self.top_distance = top_distance
        self.width = width + MARGIN_LEFT + LEAF_GAP
        self.height = MARGIN_TOP + PLOT_HEIGHT + MARGIN_BOTTOM

    def y(self, distance: float) -> float:
        return MARGIN_TOP + (1 - distance / self.top_distance) * PLOT_HEIGHT

    def x(self, x: float) -> float:
        return MARGIN_LEFT + x


def axis_svg(canvas: Canvas) -> str:
    parts = []
    for d in np.linspace(0, canvas.top_distance, DISTANCE_TICKS):
        y = canvas.y(d)
        parts.append(
            f'<line class="tick" x1="{MARGIN_LEFT - 6}" x2="{canvas.width}" y1="{y:.1f}" y2="{y:.1f}"/>'
            f'<text x="{MARGIN_LEFT - 10}" y="{y + 4:.1f}" text-anchor="end">{d:.2f}</text>'
        )
    parts.append(
        f'<text transform="rotate(-90)" x="{-MARGIN_TOP - PLOT_HEIGHT / 2}" y="14" '
        'text-anchor="middle">distance where groups join</text>'
    )
    return "".join(parts)


def edges_svg(
    nodes: dict[int, MergeNode],
    names: dict[int, str],
    x: dict[int, float],
    height: dict[int, float],
    canvas: Canvas,
) -> str:
    """A right angle from each merge down to each of its groups."""
    return "".join(
        f'<path class="edge" data-child="{names[child]}" d="M{canvas.x(x[n]):.1f},{canvas.y(height[n]):.1f} '
        f'H{canvas.x(x[child]):.1f} V{canvas.y(height[child]):.1f}"/>'
        for n, node in nodes.items()
        for child in node.children
    )


def nodes_svg(
    nodes: dict[int, MergeNode],
    names: dict[int, str],
    x: dict[int, float],
    height: dict[int, float],
    canvas: Canvas,
    tokens: Sequence[str],
) -> str:
    n_max = max(len(node.members) for node in nodes.values())
    parts = []
    for n, node in nodes.items():
        cx, cy = canvas.x(x[n]), canvas.y(height[n])
        radius = 2.5 + 7 * np.sqrt(len(node.members) / n_max)
        fill = LEAF if node.is_leaf else INNER
        classes = "node eom" if node.is_eom else "node"
        parts.append(
            f'<circle class="{classes}" data-name="{names[n]}" cx="{cx:.1f}" cy="{cy:.1f}" '
            f'r="{radius:.1f}" fill="{fill}"><title>{names[n]}</title></circle>'
        )
        if node.is_leaf:
            label = escape(repr(top_token(tokens, node.members)))
            parts.append(
                f'<text transform="translate({cx + 3:.1f},{cy + 12:.1f}) rotate(60)">{label}</text>'
            )
    return "".join(parts)


def graph_svg(
    nodes: dict[int, MergeNode], names: dict[int, str], tokens: Sequence[str]
) -> str:
    x, height = x_positions(nodes), node_heights(nodes)
    canvas = Canvas(max(height.values()), max(x.values()))
    return (
        f'<svg width="{canvas.width:.0f}" height="{canvas.height:.0f}" role="img" '
        'aria-label="HDBSCAN merge tree. Hover a node for its card.">'
        f"{axis_svg(canvas)}{edges_svg(nodes, names, x, height, canvas)}"
        f"{nodes_svg(nodes, names, x, height, canvas, tokens)}</svg>"
    )


def cards_html(
    nodes: dict[int, MergeNode],
    names: dict[int, str],
    card: Callable[[int], str],
    tokens: Sequence[str],
    activations: Float[np.ndarray, "n_hits"],
) -> str:
    return "".join(
        f'<div class="node-card" id="card-{names[n]}">'
        f"<p>{summary_line(node, names[n], relation(nodes, n, names), tokens, activations)}</p>"
        f"{card(n)}</div>"
        for n, node in nodes.items()
    )


INTRO = (
    "<p>HDBSCAN's tree over the hits, from one fit. Each dot is a group of hits, sized by how many. A dot's height is "
    "the distance where its two groups join, so groups that join late sit high. Leaves sit on the floor, labelled "
    "with their most common token. <b>Hover</b> a dot for its card on the right, and everything under it lights up. "
    "<b>Click</b> to pin the card. A card shows the hits that stay in the group longest, one per set of identical "
    "hits, shaded by AttnLRP relevance for the neuron. A merge's card shows hits from both groups that join.</p>"
)


TRIMMED_NOTE = (
    "<p>Only the leaf and eom clusters are drawn. Each leaf hangs straight off the eom cluster it sits in, and the "
    "merges between them are left out. Each eom cluster is the top of its own tree. Its height is still the distance "
    "where its own two groups first join, and its card shows hits from both of those groups.</p>"
)


def legend() -> str:
    return (
        '<p class="legend">'
        f'<span><svg width="12" height="12"><circle cx="6" cy="6" r="5" fill="{LEAF}"/></svg> leaf cluster</span>'
        f'<span><svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="{INNER}" stroke="{EOM}" '
        'stroke-width="2"/></svg> eom cluster (ringed)</span>'
        f'<span><svg width="12" height="12"><circle cx="6" cy="6" r="5" fill="{INNER}"/></svg> other merge</span>'
        "</p>"
    )


def merge_page_body(
    nodes: dict[int, MergeNode],
    card: Callable[[int], str],
    tokens: Sequence[str],
    activations: Float[np.ndarray, "n_hits"],
    heading: str,
    controls: str,
    note: str = "",
) -> Markup:
    names = node_names(nodes)
    under = {names[n]: [names[s] for s in subtree(nodes, n)] for n in nodes}
    n_leaves = sum(n.is_leaf for n in nodes.values())
    n_eom = sum(n.is_eom for n in nodes.values())
    counts = (
        f"<p>{len(nodes)} nodes, {n_leaves} leaf clusters, {n_eom} eom clusters.</p>"
    )
    return Markup(
        f"{STYLE}<h1>{escape(heading)}</h1>{controls}{INTRO}{note}{counts}{legend()}"
        '<div class="merge-layout">'
        f'<div class="merge-graph">{graph_svg(nodes, names, tokens)}</div>'
        f'<div class="merge-panel">{cards_html(nodes, names, card, tokens, activations)}</div>'
        "</div>"
        f'<script type="application/json" id="merge-under">{json.dumps(under)}</script>'
        f"{SCRIPT}"
    )
