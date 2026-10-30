#!/usr/bin/env python
"""One HTML page for one head block of one neuron, from a `head_block_scale.py` run's saved points.

    python scripts/head_block_report.py --neuron 0 --head 3 --size 32000000 --run-size 32000000

- It reads `tree_*.npy` and `entropy_*.npy`, and rebuilds the run's windows from its seed, so no model runs.
  - `run_size` is the largest size of that run. The windows were drawn for it, and smaller sizes are prefixes.
- The full report pipeline wants a points x trees probability table, which millions of points cannot have.
  - So this page shows tree sizes, the entropy, a table of every tree, and cards for the largest trees.
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from jaxtyping import Float, Int
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from head_block_forests import draw_windows
from in_tfm.html_report import ClusterLink, cluster_nav, cluster_section, page, render_template
from in_tfm.models.attn_only_2l.model import BOS_TOKEN, load_trained_attn_only
from in_tfm.report_page.figures import PLOTLY_JS, embed
from in_tfm.tiny_attn_only_training import TOKENS_FILE, story_starts


class HeadBlockReportConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", cli_kebab_case=True, cli_parse_args=True)

    checkpoint_dir: Path = Path("checkpoints/tiny_attn_only_2l_v4")
    results_dir: Path = Path("reports/head_block_entropy")
    neuron: int = 0
    head: int = 3
    size: int = 32_000_000
    run_size: int = 32_000_000
    kind: str = "probe64"
    seed: int = 0
    confident_bits: float = 0.5
    cards: int = 60
    examples: int = 12
    context: int = 12
    """Tokens shown before the current one."""
    table_rows: int = 2000
    out: Path | None = None


class TreeStats(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    sizes: Int[np.ndarray, "n_trees"]
    median_entropy: Float[np.ndarray, "n_trees"]
    confident: Float[np.ndarray, "n_trees"]
    order: Int[np.ndarray, "n_points"]
    """Point ids grouped by tree, so tree t's points are order[starts[t]:starts[t + 1]]."""
    starts: Int[np.ndarray, "n_trees_plus_one"]


class TokenChip(BaseModel):
    text: str
    share: float


def tree_stats(tree: Int[np.ndarray, "n"], entropy: Float[np.ndarray, "n"], confident_bits: float) -> TreeStats:
    order = np.lexsort((entropy, tree))
    sizes = np.bincount(tree)
    starts = np.concatenate([[0], np.cumsum(sizes)])
    sorted_entropy = entropy[order]
    medians = sorted_entropy[starts[:-1] + sizes // 2]
    confident = np.add.reduceat((sorted_entropy < confident_bits).astype(float), starts[:-1]) / sizes
    return TreeStats(sizes=sizes, median_entropy=medians, confident=confident, order=order, starts=starts)


def token_table(tokenizer) -> list[str]:
    return [tokenizer.decode([i]) for i in range(len(tokenizer))]


def top_tokens(ids: Int[np.ndarray, "m"], texts: list[str], limit: int = 8) -> list[TokenChip]:
    values, counts = np.unique(ids, return_counts=True)
    best = np.argsort(-counts)[:limit]
    return [TokenChip(text="(start)" if values[i] < 0 else texts[values[i]], share=counts[i] / len(ids)) for i in best]


def example_points(members: Int[np.ndarray, "m"], count: int, rng: np.random.Generator) -> list[tuple[str, int]]:
    """`members` are sorted by entropy, so the first ones are the tree's most confident points."""
    confident = [("most confident", int(p)) for p in members[:count // 2]]
    rest = members[count // 2:]
    drawn = rng.choice(rest, size=min(count - len(confident), len(rest)), replace=False) if len(rest) else []
    return confident + [("random", int(p)) for p in drawn]


def example(point: int, tag: str, windows, entropy, texts: list[str], context: int) -> dict:
    window, position = divmod(point, windows.shape[1])
    first = max(0, position - context)
    tokens = [{"text": texts[t], "current": i == position, "previous": i == position - 1}
              for i, t in zip(range(first, position + 1), windows[window, first:position + 1])]
    return {"tag": tag, "window": window, "position": position, "entropy": float(entropy[point]), "tokens": tokens}


def card(tree: int, stats: TreeStats, windows, previous, entropy, texts, config: HeadBlockReportConfig, rng) -> str:
    members = stats.order[stats.starts[tree]:stats.starts[tree + 1]]
    body = render_template(
        "head_block_cluster.html",
        share=stats.sizes[tree] / len(entropy), median_entropy=float(stats.median_entropy[tree]),
        confident=float(stats.confident[tree]),
        previous=top_tokens(previous[members], texts), current=top_tokens(windows.reshape(-1)[members], texts),
        examples=[example(p, tag, windows, entropy, texts, config.context)
                  for tag, p in example_points(members, config.examples, rng)],
    )
    return cluster_section(tree, int(stats.sizes[tree]), int(stats.sizes[tree]), body)


def size_figure(stats: TreeStats) -> go.Figure:
    sizes = np.sort(stats.sizes)[::-1]
    fig = go.Figure(go.Scatter(x=np.arange(1, len(sizes) + 1), y=sizes, mode="lines"))
    fig.update_xaxes(type="log", title="tree rank, largest first")
    fig.update_yaxes(type="log", title="points in tree")
    return fig


def entropy_figure(entropy: Float[np.ndarray, "n"], confident_bits: float) -> go.Figure:
    counts, edges = np.histogram(entropy, bins=np.arange(0, entropy.max() + 0.25, 0.25))
    fig = go.Figure(go.Bar(x=edges[:-1] + 0.125, y=counts, width=0.25))
    fig.add_vline(x=confident_bits, line_dash="dash", annotation_text="confident below")
    fig.update_xaxes(title="entropy over trees (bits)")
    fig.update_yaxes(title="points")
    return fig


def entropy_by_size_figure(stats: TreeStats) -> go.Figure:
    fig = go.Figure(go.Scattergl(
        x=stats.sizes, y=stats.median_entropy, mode="markers", marker=dict(size=4, opacity=0.5),
        text=[f"c{t}" for t in range(len(stats.sizes))], hovertemplate="%{text}: %{x} points, %{y:.2f} bits",
    ))
    fig.update_xaxes(type="log", title="points in tree")
    fig.update_yaxes(title="median entropy (bits)")
    return fig


def tree_table(stats: TreeStats, windows, previous, texts, limit: int) -> str:
    """The largest trees, each with its most common previous and current token."""
    rows = []
    flat = windows.reshape(-1)
    for tree in np.argsort(-stats.sizes)[:limit]:
        members = stats.order[stats.starts[tree]:stats.starts[tree + 1]]
        prev, cur = top_tokens(previous[members], texts, 1)[0], top_tokens(flat[members], texts, 1)[0]
        rows.append(f"<tr><td>c{tree}</td><td>{stats.sizes[tree]:,}</td><td>{stats.median_entropy[tree]:.2f}</td>"
                    f"<td>{100 * stats.confident[tree]:.0f}%</td><td><code>{escape(prev.text)}</code> "
                    f"{100 * prev.share:.0f}%</td><td><code>{escape(cur.text)}</code> {100 * cur.share:.0f}%</td></tr>")
    head = ("<tr><th>tree</th><th>points</th><th>median bits</th><th>confident</th><th>top previous</th>"
            "<th>top current</th></tr>")
    return f'<div style="max-height:60vh;overflow:auto"><table class="borders">{head}{"".join(rows)}</table></div>'


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def section(title: str, lede: str, body: str) -> str:
    return f'<section class="page-section"><h2>{title}</h2><p class="meta">{lede}</p>{body}</section>'


def main() -> None:
    config = HeadBlockReportConfig()
    name = f"neuron{config.neuron}_head{config.head}_n{config.size}_{config.kind}"
    tree = np.load(config.results_dir / f"tree_{name}.npy")
    entropy = np.load(config.results_dir / f"entropy_{name}.npy").astype(np.float64)
    _, tokenizer = load_trained_attn_only(config.checkpoint_dir)
    tokens = np.load(config.checkpoint_dir / TOKENS_FILE.format(split="train"), mmap_mode="r")
    seq_len = 256
    starts = story_starts(tokens, tokenizer.convert_tokens_to_ids(BOS_TOKEN), seq_len)
    windows, _ = draw_windows(tokens, starts, -(-config.run_size // seq_len), seq_len,
                              np.random.default_rng(config.seed))
    windows = windows[:config.size // seq_len]
    previous = np.concatenate([np.full((len(windows), 1), -1), windows[:, :-1]], axis=1).reshape(-1)
    texts = token_table(tokenizer)
    stats = tree_stats(tree, entropy, config.confident_bits)
    rng = np.random.default_rng(config.seed)
    largest = np.argsort(-stats.sizes)[:config.cards]

    tiles = [("points", f"{len(tree):,}"), ("trees", f"{len(stats.sizes):,}"),
             ("confident", f"{100 * (entropy < config.confident_bits).mean():.1f}%"),
             ("median entropy", f"{np.median(entropy):.2f} bits"),
             ("median / log2(trees)", f"{np.median(entropy) / np.log2(len(stats.sizes)):.3f}")]
    stats_html = '<div class="stats">' + "".join(f"<div class='stat'><b>{v}</b><span>{k}</span></div>"
                                                for k, v in tiles) + "</div>"
    body = "".join([
        f"<h1>Neuron {config.neuron}, head {config.head} block</h1>",
        f'<p class="meta">Layer 0 o_proj. Every position of {len(windows):,} TinyStories windows is a point. '
        "Underlined: the previous token. Outlined: the current token.</p>",
        stats_html,
        section("Tree sizes", "Each tree is a cluster. Sizes on log axes, largest first.", embed(size_figure(stats), "45vh")),
        section("Entropy over trees", f"Per point. Confident is below {config.confident_bits} bits.",
                embed(entropy_figure(entropy, config.confident_bits), "45vh")),
        section("Entropy by tree size", "Each dot is a tree, at its median point.",
                embed(entropy_by_size_figure(stats), "50vh")),
        section("Every tree", f"The {min(config.table_rows, len(stats.sizes)):,} largest trees.",
                tree_table(stats, windows, previous, texts, config.table_rows)),
        f'<h2 class="clusters-head">The {len(largest)} largest trees</h2>',
        cluster_nav([ClusterLink(cluster_id=int(t), n_hits=int(stats.sizes[t])) for t in largest]),
        *[card(int(t), stats, windows, previous, entropy, texts, config, rng) for t in largest],
    ])
    out = config.out or config.results_dir / f"report_{name}.html"
    out.write_text(page(f"Neuron {config.neuron} head {config.head}", body, PLOTLY_JS))
    print(out)


if __name__ == "__main__":
    main()
