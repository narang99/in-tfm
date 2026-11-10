# OV attribution graphs on the attn-only 2L model

Built 2026-11-01.

## Question

- Where does a layer 1 o_proj neuron's head block get its input from, traced back through layer 0 to the embedding?

## Setup

- The root is one head's block of one layer 1 o_proj neuron, at one position.
- The chain, from the root down:
  - the head block
  - its z dims
  - value neurons
  - the residual junction, kept as its own node
  - a layer 0 o_proj neuron
  - layer 0 z dims and value neurons
  - embedding dims
- Edges are exact OV splits, with attention held fixed.
  - Hovering a node shows its own AttnLRP, which includes QK.
- Pruning: each node keeps its positive parents only, up to 70% of the node's own positive sum.
  - There is no floor at the root and no cap.
  - An earlier rule (80% per sign, at most 5 parents, a 1% floor at the root) left dead ends at layer 0 o_proj.
    - It made layer 0 look idle. Unpruned, layer 0 carries about half the root.
  - Positive-only sums exceed the root, so leaf totals are not shares of it.
- Graphs have 4k to 6k nodes.
  - So the page is a drill-down navigator with breadcrumbs, not a drawn graph.

## Code

- Branch [`feat/attribution-graph`](https://github.com/narang99/in-tfm/tree/feat/attribution-graph), on top of [`feat/residual-concepts`](https://github.com/narang99/in-tfm/tree/feat/residual-concepts).
  - `feat/residual-concepts` was closed PR #132.
    - It adds concept-card pages for what layer 0 writes into the residual, and a layer 1 neuron page.
  - Script: [`scripts/attribution_graph_report.py`](https://github.com/narang99/in-tfm/blob/989eac9a3c23664545ad7b624e8946f684f8e768/scripts/attribution_graph_report.py).
  - Graph code: [`models/attn_only_2l/attribution_graph.py`](https://github.com/narang99/in-tfm/blob/989eac9a3c23664545ad7b624e8946f684f8e768/src/in_tfm/models/attn_only_2l/attribution_graph.py).

```
python scripts/attribution_graph_report.py --neuron 215 --head 1 --text "Once upon a time, there was a little girl named Lily. She"
```

## Findings

- Each head's W_V reads all of d_model.
  - About 12.5% of its weight is on "its own" slice.
  - Only W_O has real per-head blocks.
- The nearest dataset positions by Hadamard are often exact copies of the query, at distance 0.
  - Skip positions whose last 8 tokens match the query.
  - Even then, the neighbours of a story-opening query are the same template with another name.
