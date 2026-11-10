# " the" logit dependencies on the 1L d256 model, as token x o_proj cluster

Run on a Colab L4 on 2026-11-05.

## Question

- What are the clusters c519 and c647 of the " the" logit made of?
  - c519 is the obvious one, " in" after " walk".
  - c647 is the diffuse one, " \n" after ".".
- Each fired residual dim is modelled as its two writers' clusters.
  - The embedding's cluster is the token itself.
  - The o_proj dim's cluster is the nearest `mean.pt` by cosine, from the 64 o_proj batch reports in S3.
  - No magnitudes are compared. A dependency is the (token, o_proj cluster) pair and its share of the dim's firings.

## Code

- Branch `exp/the-logit-dependency-d256`, on top of the dependency stack (#147 to #161), at [`55b10bd`](https://github.com/narang99/in-tfm/tree/55b10bd).
- Commands, on Colab, after syncing only `*/mean.pt` of the o_proj reports into `reports/tiny-attn-only-1l/model.layers.0.self_attn.o_proj/`:

```
python scripts/run_neuron_report.py --config configs/tiny_attn_only_1l_the_logit.yaml
python scripts/run_dependency_report.py --config configs/tiny_attn_only_1l_the_logit.yaml
```

- Settings: dim mass 0.9, writer mass 1.0, so both writers are kept on every fired dim.

## Setup

- The neuron report was re-run, since the 2026-11-04 run kept no hit positions.
  - Same config: 19,600 stories, about 4M positions, 1M signed bucketed hits, HDBSCAN leaf at 50.
  - It gave 651 clusters again, in 980 s.
  - c519 has 234 hits and c647 has 1,477, as before. Their firing tokens match too, so the ids carried over.
- The dependency run took 300 s for both clusters.

## Results

- Outputs are in S3, under `s3://narang99-private/in-tfm/tiny-attn-only-1l/lm_head/`.
  - `263/index.html` is the neuron report. Its clusters link to their dependency pages.
  - `263/dependencies/cluster_519/index.html` and `cluster_647/index.html` are the pages.
  - `dependency_graph.html` is the graph page.
  - The `firings.json` files were not kept, 28 MB and 126 MB.

| cluster | examples | firings | dims fired | frequent | o_proj firings with a report | page |
|---|---|---|---|---|---|---|
| 519 | 234 | 30,414 | 209 | 172 | 44% | 3.5 MB |
| 647 | 1,477 | 197,155 | 226 | 199 | 41% | 11.4 MB |

- Each cluster fires at a single token, so the embedding side of every pair is that token.
- c519's top dims each land in one o_proj cluster.
  - Dims 117, 225, 123 and 24 match one o_proj cluster in 100% of firings. Dim 46 in 74%.
- c647's top dims spread over 4 to 8 pairs.
  - The top pair holds 62% to 73% of the firings.

## Caveats

- Only 64 of 256 o_proj dims have reports, so 56% to 59% of o_proj firings show "(no report)".
- Mean matching has no cutoff. A far match still counts, and the page does not show its distance yet.
- Both clusters fire at one token, so the embedding side says nothing beyond the cluster's token.
