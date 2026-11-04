# First dependency reports: MNIST logit 9 and the 1L " the" logit

Run locally on 2026-11-04, on the Mac, CPU only.

## Question

- Does the dependency report, ported from hiccup-ide `hariom/hp-do`, run end to end on our two cheapest targets?
  - A direct map: MNIST 2/4 `head.1`, reading `relu2(conv2)` flattened.
  - A residual map: the 1L d128 attn-only `lm_head`, reading the embeddings plus 4 `o_proj` heads.
- How wide is a dependency in each case?

## Code

- Branch `feat/dep-driver`, the top of the dependency report stack, at [`39756ed`](https://github.com/narang99/in-tfm/tree/39756ed).
- Commands:

```
uv run python scripts/run_neuron_report.py --config configs/mnist_cnn_small_logit9.yaml
uv run python scripts/run_dependency_report.py --config configs/mnist_cnn_small_logit9.yaml

uv run python scripts/run_neuron_report.py --config configs/tiny_attn_only_1l_d128_the_logit.yaml
uv run python scripts/run_dependency_report.py --config configs/tiny_attn_only_1l_d128_the_logit.yaml --device cpu
```

- Settings are the raw defaults:
  - dim mass 0.9, writer mass 0.9
  - frequent at 10% of examples and at least 2
  - threshold 0.8 noise units
  - view trimmed at 90% cumulative share

## Setup

- MNIST: the first 100 test images per digit, 1000 images.
  - Logit 9 has 221 positive hits, all kept by bucketed sampling at floor 0.
  - HDBSCAN on normalised euclidean, `min_cluster_size` 5, gives 2 clusters.
  - The default of 10 gave none, and raw weighted Jaccard at 5 gave none.
- 1L: 400 TinyStories validation stories of up to 128 tokens, 50,881 real positions.
  - " the" is token 263. Signed bucketed sampling kept 2000 hits.
  - Raw euclidean HDBSCAN, `min_cluster_size` 20, gives 5 clusters.
  - Density peaks with weighted Jaccard left every hit too mixed.

## Results

- Outputs are in `reports/mnist_cnn_small_logit9/` and `reports/tiny_attn_only_1l_d128_the_logit/` (git-ignored).
  - Each cluster has `neuron_*/dependencies/cluster_{id}/firings.json` and `index.html`.
- Time: MNIST 6 s for both clusters. 1L 27 s for all 5, nearly all of it the per-cluster noise pass.
- Peak RSS of a 1L collection was 580 to 680 MB.

### MNIST logit 9

| cluster | examples | dependencies | frequent | dims per example (median) | writer firings past 0.8 |
|---|---|---|---|---|---|
| 0 | 15 | 52 | 45 | 33 | 43% |
| 1 | 7 | 50 | 43 | 36 | 33% |

- Both clusters lead with the same two dims, conv2 channel 1 at cells 16 and 9.
  - So on these two clusters the dependency view does not tell them apart at the top.
- Every dependency has one writer, as the direct map says.

### 1L " the" logit

| cluster | examples | dependencies | frequent | dims per example (median) | writer firings past 0.8 |
|---|---|---|---|---|---|
| 3 | 419 | 119 | 111 | 70 | 10% |
| 2 | 73 | 117 | 104 | 67 | 17% |
| 0 | 40 | 76 | 76 | 76 | 36% |
| 4 | 31 | 108 | 102 | 67 | 17% |
| 1 | 23 | 85 | 85 | 69 | 33% |

- An example needs about 70 of 128 stream dims to reach 90% of the logit's |input x weight|.
- Most dims keep 4 of their 5 writers to reach 90%.
- Head 3 holds the largest writer share in every cluster, from 24% in cluster 3 to 50% in cluster 4.
  - The embedding holds 7% to 19%.
- Dim 44 is the top dependency in 4 of the 5 clusters.
- Residual writers mostly sit inside their noise band.
  - Only 10% to 36% of writer firings are past 0.8 noise units, against 33% to 43% on MNIST.
- The largest cluster's page has 79 cards and weighs 2.8 MB.

## Caveats

- These are smoke runs, small enough for the laptop. The clusters are not tuned.
- Phase B is not built yet, so no firing is matched to a writer's own clusters.
  - The writer reports for `o_proj.heads.h` do not exist yet either.
- MPS was not tried for the 1L run, only CPU.
