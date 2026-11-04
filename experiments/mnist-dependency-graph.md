# MNIST 2/4 dependency graph: every logit's clusters, matched to conv2's own clusters

Run locally on 2026-11-04, on the Mac, CPU only.

## Question

- With phase B built, what does the full dependency graph of the 2/4 MNIST net look like?
  - Origin: every cluster of all 10 logits of `head.1`.
  - Writers: the 4 conv2 channels, each matched against its own neuron report's clusters.
- Do logit clusters lean on a few conv2 clusters, or on many?

## Code

- Branch `feat/dep-graph`, the top of the dependency report stack, at [`54b6951`](https://github.com/narang99/in-tfm/tree/54b6951).
- Commands:

```
uv run python scripts/run_neuron_report.py --config configs/mnist_cnn_small_conv2_writers.yaml
uv run python scripts/run_neuron_report.py --config configs/mnist_cnn_small_logits.yaml
uv run python scripts/run_dependency_report.py --config configs/mnist_cnn_small_logits.yaml
```

- Settings are the raw defaults, plus `min_cluster_size` 5 for the logits.
  - dim mass 0.9, writer mass 0.9, frequent at 10% and at least 2, threshold 0.8, outlier ratio 0.1
  - view trimmed at 90% cumulative share, for cards and for graph edges

## Setup

- Data: the first 100 test images per digit, 1000 images, for every report.
- conv2 writer reports: elbow hits, euclidean HDBSCAN on L2-normalised Hadamard rows, leaf selection, `min_cluster_size` 10.

| conv2 channel | hits | clusters | hits in shown clusters |
|---|---|---|---|
| 0 | 5,812 | 51 | 1,196 (21%) |
| 1 | 14,420 | 52 | 1,315 (9%) |
| 2 | 6,278 | 15 | 423 (7%) |
| 3 | 3,824 | 10 | 487 (13%) |

- Logit reports: every positive logit is a hit, 114 to 262 per logit, 2 to 6 clusters each, 32 in all.

## Results

- Outputs are in `reports/mnist_cnn_small_conv2_writers/` and `reports/mnist_cnn_small_logits/` (git-ignored).
  - `reports/mnist_cnn_small_logits/dependency_graph.html` is the graph of all 32 origin clusters.
  - Each logit report links every cluster to its dependency page.
- Time: conv2 reports 37 s, logit reports 15 s, all 32 dependency reports with matching 37 s.
- 20,107 writer firings were matched.
  - 7% at distance 0: the firing is itself one of conv2's clustered hits, since every report used the same images.
  - Median distance 0.23 and 90th percentile 0.53, euclidean between L2-normalised rows.
  - 12% landed in a writer cluster holding under 10% of that writer's matches, so it was folded as an outlier.
- The graph is wide.
  - An origin cluster needs 18 to 62 writer-cluster edges to reach 90% of its matched |contribution|, median 40.
  - Its top edge holds 5% to 30%.
- By writer channel, conv2:1 holds 39% of the summed edge share, conv2:2 30%, conv2:3 17%, conv2:0 13%.
- A few writer clusters are hubs, near the top for many logits.
  - conv2:2 c7, the largest of its 15 clusters at 134 hits, tops logits 0, 1, 4, 5 and 8.
  - conv2:1 c41 is second for logits 1, 3 and 8.
  - The logits that lean hardest on them are logit 1 c3 (30% and 21%) and logit 8 c1 (19% and 19%).

## Caveats

- Most conv2 hits are HDBSCAN noise. Only 7% to 21% are in shown clusters.
  - Matching has no cutoff, so a firing is assigned to the nearest of these few hundred hits, even if it sits where HDBSCAN found noise.
  - The distance beside each match is the only guard. A noise-aware rule, or peak geodesic membership on a density-peak forest, would be the next step.
- conv2's elbow hits are its strong activations. A weak conv2 firing under a logit is still matched to clusters of strong ones.
- Matching is in-sample, since the writers were clustered on the same 1000 images.
- The logit clusters are small, 5 to 55 examples, so fired ratios move in steps of 2% to 20%.
