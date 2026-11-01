#!/usr/bin/env python
"""Attribution graphs for one (layer 1 neuron, head) root, on a text and on the dataset positions nearest to it.

    python scripts/attribution_graph_report.py --neuron 215 --head 1 --text "Once upon a time, there was a little girl named Lily. She"

- The root is one head's block of one layer 1 o_proj neuron, at the text's last position.
  - Its Hadamard product is that head's 32 terms z[h, j] W_O[n, hj].
- Neighbours are the dataset positions whose Hadamard product is nearest in euclidean distance.
  - Positions come from training windows, one per story, and each neighbour is from a different window.
  - TinyStories repeats stories, so the nearest positions are often copies of the query.
    - A position whose last `distinct_tokens` tokens match the query's, or an earlier pick's, is skipped.
  - Raw rows, not normalised, so a neighbour matches the size of each term as well as its shape.
- Each example gets a graph page, and an index page lines them up for comparison.
- Graph edges are exact OV splits with attention held fixed.
  - Each node keeps its largest positive parents up to 70% of its own positive sum, so the full graph is large.
  - The page walks it one node at a time instead of drawing it, with each node's own AttnLRP on the tokens.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, Int
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from in_tfm.colormaps import rd_bk_gn, rd_wht_gn
from in_tfm.device import default_device
from in_tfm.html_report import page, render_template
from in_tfm.models.attn_only_2l.adapter import patch_attn_only_for_attn_lrp
from in_tfm.models.attn_only_2l.attribution_graph import (
    AttributionGraph, Coordinate, Embed, GraphNode, HeadBlock, HeadOut, OProjOut, OvState, PruneRule, Resid, ValueNeuron,
    build_graph, ov_state, parts, value,
)
from in_tfm.models.attn_only_2l.model import BOS_TOKEN, AttnOnlyForCausalLM, load_trained_attn_only
from in_tfm.models.attn_only_2l.node_attnlrp import node_token_relevance
from in_tfm.presenters.text.colored_tokens import diverging_colors, display_text
from in_tfm.presenters.text.concept_cards import concept_card_context
from in_tfm.presenters.text.views import TokenColors
from in_tfm.tiny_attn_only_training import TOKENS_FILE, story_starts

LAYER = 1
STAGE_NAMES = ["root: head block", "L1 head out z", "L1 value neuron", "resid into L1", "L0 o_proj neuron",
               "L0 head out z", "L0 value neuron", "embedding"]
SHOWN_BEFORE = 30
"""Tokens shown before the root's position, unless a node sits earlier."""
MAX_SHOWN = 120


class AttributionGraphConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", cli_kebab_case=True, cli_parse_args=True)

    checkpoint_dir: Path = Path("checkpoints/tiny_attn_only_2l_v4")
    text: str = "Once upon a time, there was a little girl named Lily. She"
    neuron: int = 215
    head: int | None = None
    """By default, the head adding the most to the neuron at the text's last position."""
    neighbours: int = 6
    distinct_tokens: int = 8
    n_windows: int = 1000
    split: str = "train"
    batch_size: int = 32
    seed: int = 0
    device: str = default_device()
    prune: PruneRule = PruneRule()
    out_dir: Path = Path("reports/attribution_graph")


class Example(BaseModel):
    name: str
    title: str
    input_ids: list[int]
    distance: float = 0.0
    window: int | None = None


class Cell(BaseModel):
    value: float
    colors: TokenColors


class ExampleSummary(BaseModel):
    name: str
    title: str
    distance: float
    root_value: float
    context: list[str]
    hadamard: list[Cell]
    stage_counts: list[int]
    top_sources: list[str]
    top_leaves: list[str]
    n_nodes: int


def hadamard_cells(row: Float[np.ndarray, "head_dim"], vmax: float) -> list[Cell]:
    return [Cell(value=float(v), colors=diverging_colors(float(v), vmax, rd_wht_gn, rd_bk_gn)) for v in row]


@torch.no_grad()
def dataset_hadamards(model: AttnOnlyForCausalLM, windows: np.ndarray, neuron: int, head: int, batch_size: int,
                      device: str) -> Float[np.ndarray, "n seq head_dim"]:
    attn = model.model.layers[LAYER].self_attn
    block = slice(head * attn.head_dim, (head + 1) * attn.head_dim)
    weight = attn.o_proj.weight[neuron, block]
    captured = {}
    handle = attn.o_proj.register_forward_hook(lambda _m, inp, _o: captured.__setitem__("z", inp[0]))
    rows = []
    try:
        for start in range(0, len(windows), batch_size):
            model(input_ids=torch.from_numpy(windows[start:start + batch_size]).to(device))
            rows.append((captured["z"][..., block] * weight).float().cpu().numpy())
    finally:
        handle.remove()
    return np.concatenate(rows)


def nearest_positions(hadamards: Float[np.ndarray, "n seq head_dim"], query: Float[np.ndarray, "head_dim"],
                      windows: np.ndarray, query_ids: list[int], k: int, distinct: int) -> list[tuple[int, int, float]]:
    """(window, position, distance), nearest first, one per window, each with its own last `distinct` tokens."""
    distances = np.linalg.norm(hadamards - query, axis=-1)
    picked, used, seen_contexts = [], set(), {tuple(query_ids[-distinct:])}
    for flat in np.argsort(distances, axis=None):
        window, position = np.unravel_index(flat, distances.shape)
        context = tuple(windows[window, max(0, position + 1 - distinct):position + 1].tolist())
        if window in used or context in seen_contexts:
            continue
        used.add(int(window))
        seen_contexts.add(context)
        picked.append((int(window), int(position), float(distances[window, position])))
        if len(picked) == k:
            break
    return picked


def query_hadamard(model: AttnOnlyForCausalLM, input_ids: list[int], neuron: int, head: int) -> np.ndarray:
    state = ov_state(model, torch.tensor([input_ids]))
    z = state.head_outputs[LAYER][len(input_ids) - 1, head]
    return (z * state.w_o[LAYER][neuron, head * state.head_dim:(head + 1) * state.head_dim]).numpy()


def node_label(node: Coordinate, tokens: list[str]) -> str:
    def at(pos: int) -> str:
        return f"{tokens[pos]!r}@{pos}"
    match node:
        case HeadBlock(layer=layer, neuron=neuron, head=head, pos=pos):
            return f"L{layer} n{neuron} h{head} {at(pos)}"
        case HeadOut(layer=layer, pos=pos, head=head, dim=dim):
            return f"L{layer} z h{head}.{dim} {at(pos)}"
        case ValueNeuron(layer=layer, pos=pos, head=head, dim=dim):
            return f"L{layer} v h{head}.{dim} {at(pos)}"
        case Resid(pos=pos, dim=dim):
            return f"resid d{dim} {at(pos)}"
        case OProjOut(layer=layer, pos=pos, neuron=neuron):
            return f"L{layer} o n{neuron} {at(pos)}"
        case Embed(pos=pos, dim=dim):
            return f"E d{dim} {at(pos)}"
    raise ValueError(node)


def shown_range(graph: AttributionGraph, t: int) -> tuple[int, int]:
    earliest = min(n.node.pos for n in graph.nodes)
    low = max(0, min(earliest, t - SHOWN_BEFORE), t - MAX_SHOWN)
    return low, t + 1


def positive_relevance(state: OvState, node: GraphNode) -> float:
    """What the node's positive pieces hold, the total its kept parents are measured against."""
    if not node.value:
        return 0.0
    scale = node.relevance / node.value
    return sum(max(0.0, part * scale) for _, part in parts(state, node.node))


def graph_page_data(model: AttnOnlyForCausalLM, state: OvState, example: Example, graph: AttributionGraph,
                    tokens: list[str]) -> dict:
    t = len(example.input_ids) - 1
    low, high = shown_range(graph, t)
    lrp = node_token_relevance(model, torch.tensor([example.input_ids]),
                               [n.node for n in graph.nodes if not isinstance(n.node, Embed)])
    index = {n.node: i for i, n in enumerate(graph.nodes)}
    parents, children = defaultdict(list), defaultdict(list)
    for e in sorted(graph.edges, key=lambda e: -e.relevance):
        parents[index[e.child]].append([index[e.parent], round(e.relevance, 6)])
        children[index[e.parent]].append([index[e.child], round(e.relevance, 6)])
    return {
        "tokens": tokens[low:high],
        "offset": low,
        "root": graph.nodes[index[graph.root]].relevance,
        "stages": STAGE_NAMES,
        "nodes": [node_data(state, n, tokens, lrp.get(n.node), low, high) | {"parents": parents[i], "children": children[i]}
                  for i, n in enumerate(graph.nodes)],
    }


def node_data(state: OvState, n: GraphNode, tokens: list[str], lrp: np.ndarray | None, low: int, high: int) -> dict:
    data = {"stage": n.stage, "label": node_label(n.node, tokens), "value": round(n.value, 5),
            "relevance": round(n.relevance, 6), "pos": n.node.pos, "nParts": len(parts(state, n.node)),
            "positiveSum": round(positive_relevance(state, n), 6), "lrp": None, "lrpOutside": 0.0}
    if lrp is not None:
        data |= {"lrp": [round(float(x), 5) for x in lrp[low:high]], "lrpOutside": round(float(lrp[:low].sum()), 5)}
    return data


def summary(example: Example, graph: AttributionGraph, tokens: list[str], hadamard: np.ndarray,
            vmax: float) -> ExampleSummary:
    t = len(example.input_ids) - 1
    by_stage = defaultdict(int)
    for n in graph.nodes:
        by_stage[n.stage] += 1
    sources = sorted((n for n in graph.nodes if isinstance(n.node, ValueNeuron) and n.node.layer == LAYER),
                     key=lambda n: -abs(n.relevance))
    leaves = defaultdict(float)
    for n in graph.nodes:
        if isinstance(n.node, Embed):
            leaves[(n.node.pos, tokens[n.node.pos])] += n.relevance
    top_leaves = sorted(leaves.items(), key=lambda kv: -abs(kv[1]))[:5]
    return ExampleSummary(
        name=example.name, title=example.title, distance=example.distance,
        root_value=graph.nodes[0].relevance,
        context=tokens[max(0, t - 15):t + 1],
        hadamard=hadamard_cells(hadamard, vmax),
        stage_counts=[by_stage[s] for s in range(len(STAGE_NAMES))],
        top_sources=[f"{node_label(n.node, tokens)} {n.relevance:+.3f}" for n in sources[:4]],
        top_leaves=[f"{tok!r}@{pos} ({pos - t:+d}) {r:+.3f}" for (pos, tok), r in top_leaves],
        n_nodes=len(graph.nodes),
    )


def draw_windows(tokens: np.ndarray, bos: int, n: int, seq_len: int, rng: np.random.Generator) -> np.ndarray:
    chosen = rng.choice(story_starts(tokens, bos, seq_len), size=n, replace=False)
    return np.stack([tokens[s:s + seq_len] for s in np.sort(chosen)]).astype(np.int64)


def main() -> None:
    config = AttributionGraphConfig()
    print(config)
    model, tokenizer = load_trained_attn_only(config.checkpoint_dir)
    bos = tokenizer.convert_tokens_to_ids(BOS_TOKEN)
    query_ids = [bos, *tokenizer.encode(config.text)]
    t = len(query_ids) - 1
    state = ov_state(model, torch.tensor([query_ids]))
    head_values = [value(state, HeadBlock(layer=LAYER, pos=t, neuron=config.neuron, head=h)) for h in range(8)]
    head = config.head if config.head is not None else int(np.argmax(head_values))
    print(f"neuron {config.neuron} head blocks at the last position: {[round(v, 3) for v in head_values]}, head {head}")
    query = query_hadamard(model, query_ids, config.neuron, head)

    rng = np.random.default_rng(config.seed)
    tokens = np.load(config.checkpoint_dir / TOKENS_FILE.format(split=config.split), mmap_mode="r")
    windows = draw_windows(tokens, bos, config.n_windows, model.config.max_position_embeddings, rng)
    hadamards = dataset_hadamards(model.to(config.device), windows, config.neuron, head, config.batch_size,
                                  config.device)
    model = model.cpu()
    examples = [Example(name="query", title="the text", input_ids=query_ids)] + [
        Example(name=f"neighbour_{i}", title=f"neighbour {i}: window {w}, position {p}",
                input_ids=windows[w, :p + 1].tolist(), distance=d, window=w)
        for i, (w, p, d) in enumerate(nearest_positions(hadamards, query, windows, query_ids, config.neighbours,
                                                               config.distinct_tokens), start=1)
    ]

    patch_attn_only_for_attn_lrp(model)
    out_dir = config.out_dir / f"neuron_{config.neuron}_head_{head}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [query_hadamard(model, ex.input_ids, config.neuron, head) for ex in examples]
    vmax = float(np.abs(np.stack(rows)).max())
    summaries = []
    for example, row in zip(examples, rows):
        token_texts = [display_text(tokenizer.decode([tok])) for tok in example.input_ids]
        root = HeadBlock(layer=LAYER, pos=len(example.input_ids) - 1, neuron=config.neuron, head=head)
        state = ov_state(model, torch.tensor([example.input_ids]))
        graph = build_graph(state, root, config.prune)
        summaries.append(summary(example, graph, token_texts, row, vmax))
        body = render_template(
            "attribution_graph.html", title=example.title, neuron=config.neuron, head=head, prune=config.prune,
            graph_data=graph_page_data(model, state, example, graph, token_texts), **concept_card_context(),
        )
        (out_dir / f"{example.name}.html").write_text(page(f"Graph n{config.neuron} h{head} {example.name}", body))
        print(f"{example.name}: distance {example.distance:.3f}, root {graph.nodes[0].relevance:+.3f}, "
              f"{len(graph.nodes)} nodes, {len(graph.edges)} edges")

    index = render_template("attribution_graph_index.html", neuron=config.neuron, head=head, text=config.text,
                            n_positions=int(hadamards.shape[0] * hadamards.shape[1]), stages=STAGE_NAMES,
                            summaries=summaries, prune=config.prune, **concept_card_context())
    (out_dir / "index.html").write_text(page(f"Graphs n{config.neuron} h{head}", index))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
