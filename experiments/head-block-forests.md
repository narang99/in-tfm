# Density-peak forests of attn-only head blocks, scaled to 32M points

Run 2026-10-30 to 2026-10-31, on Colab T4 and A100.

## Question

- Do the trees of one o_proj neuron's head block settle as the data grows?
- How does the forest scale past the exact method's limit?

## Setup

- Model: our tiny attn-only 2L TinyStories model, run 4.
- A point is one layer 0 head's block of one o_proj neuron: z of that head times the neuron's block weights.
- The forest runs on raw rows, without normalisation.
  - Weighted Jaccard is sensitive to scale here.
- Up to 256k points the forest is exact.
- Past that, the forest uses IVF neighbours.
  - The IVF lists are k-means cells.
  - Exact weighted Jaccard is computed inside the searched lists.
  - Searching every list gives the exact forest.
  - Final settings: fixed 1,000-point lists, 64 lists searched per point, and a fused Triton search kernel.
- Tree membership entropy is computed without the points x trees table.
  - On the GPU with cuGraph, or by batched frontier relaxation in torch.

## Code

- Branch [`feat/ivf-forest`](https://github.com/narang99/in-tfm/tree/feat/ivf-forest), closed PR #130.
  - Forest: [`density_peaks/ivf_forest.py`](https://github.com/narang99/in-tfm/blob/0e4e41f11e8886c25545372d5b57d44480abecd2/src/in_tfm/density_peaks/ivf_forest.py) and [`triton_ivf.py`](https://github.com/narang99/in-tfm/blob/0e4e41f11e8886c25545372d5b57d44480abecd2/src/in_tfm/density_peaks/triton_ivf.py).
  - Entropy: [`cugraph_entropy.py`](https://github.com/narang99/in-tfm/blob/0e4e41f11e8886c25545372d5b57d44480abecd2/src/in_tfm/density_peaks/probabilities/cugraph_entropy.py) and [`frontier_entropy.py`](https://github.com/narang99/in-tfm/blob/0e4e41f11e8886c25545372d5b57d44480abecd2/src/in_tfm/density_peaks/probabilities/frontier_entropy.py).
  - Scripts: [`head_block_forests.py`](https://github.com/narang99/in-tfm/blob/0e4e41f11e8886c25545372d5b57d44480abecd2/scripts/head_block_forests.py), [`head_block_elbow.py`](https://github.com/narang99/in-tfm/blob/0e4e41f11e8886c25545372d5b57d44480abecd2/scripts/head_block_elbow.py), [`head_block_scale.py`](https://github.com/narang99/in-tfm/blob/0e4e41f11e8886c25545372d5b57d44480abecd2/scripts/head_block_scale.py), [`ivf_sweep.py`](https://github.com/narang99/in-tfm/blob/0e4e41f11e8886c25545372d5b57d44480abecd2/scripts/ivf_sweep.py), [`head_block_batch.py`](https://github.com/narang99/in-tfm/blob/0e4e41f11e8886c25545372d5b57d44480abecd2/scripts/head_block_batch.py), [`head_block_report.py`](https://github.com/narang99/in-tfm/blob/0e4e41f11e8886c25545372d5b57d44480abecd2/scripts/head_block_report.py).
- The IVF forest itself is moving to main through issue #171. The entropy code and the scripts stay on this branch.

```
python scripts/head_block_forests.py --checkpoint-dir checkpoints/tiny_attn_only_2l --neurons "[0,1,2,3]"
python scripts/head_block_scale.py --neuron 0 --head 2 --sizes '[256000,1000000,4000000]'
python scripts/head_block_batch.py --neurons '[0,1,2,3,4,5,6,7,8,9]' --size 4000000
python scripts/head_block_report.py --neuron 0 --head 3 --size 32000000 --run-size 32000000
```

## Results: trees as the data grows

- Previous-token head (h2): the confident share goes from 50% at 1k points to 97% at 256k, with median entropy 0.00 bits.
  - Its trees are groups of previous tokens: subject pronouns, sentence ends, possessives, names, be-verbs.
- Mixed heads (h1, h3, h5): the confident share rises slowly, from about 10% to 23-29% at 256k.
- Diffuse heads (h0, h4, h7): the confident share stays at 5-8% up to 256k.
- Tree counts keep growing with n, roughly with its square root.
  - So trees don't settle into a fixed set. They split further as the data grows.
- From 256k to 4M points:
  - h2 goes from 97% confident to 98%, h3 from 32% to 41%, h0 from 11% to 14%, and h7 stays at about 7%.
- At 32M points, on neuron 0: h3 is 45.8% confident with 4,892 trees, and h0 16.7% with 5,966 trees.
- A diffuse result describes this neuron's block of the head, not the head itself.
  - The neuron's block weights can hide structure that z has.

## Results: 10 neurons x 8 heads at 4M

- The confident share is set by the head, not the neuron. The spread is under 6 points per head.
  - h2 98%, h3 39-43%, h1 32-37%, h5 32-37%, h0 13-14%, h4 10-11%, h6 5-7%, h7 5-7%.
- Points confident in two neurons fall in the same trees for h1, h3, h4 and h5, at ARI 0.82 to 0.96.
  - On all points the agreement is only 0.3 to 0.5. Each neuron splits the diffuse mass its own way.
- In every head except h2, the best key for a tree is the token the head attends to most.
  - h3 trees key on a token 1 to 3 back: "and", "said", "Once", "had", "The".
  - Token chips alone made h3 look uninterpretable. Marking the attended token explains it.

## Results: IVF accuracy and speed

- h2, neuron 0, 16 lists searched: 8.6 s against 114 s exact at 256k, and 18.6 s against 657 s at 1M.
  - Recall 99.3% and 99.6%, ARI 0.91 and 0.94, with the same tree count.
- h0 is harder.
  - 16 lists searched give recall 92.7% and ARI 0.76.
  - 64 lists searched give recall 98.5% and ARI 0.93.
- At 32M points on an A100, a pair takes 9.5 min for h3 and 11 min for h0, forest and entropy together.
- On a T4 at 4M points, the batch of 80 pairs took 4.0 h, 145 to 250 s per pair.
- Tried and dropped:
  - cuVS L1 candidates: recall 0.94, ARI 0.79, and twice the trees.
  - Probes per cell (86% recall), probes from a pool around each list's centroid (97.5% recall).
  - Union of probes was accurate but slower.

## Outputs

- `s3://narang99-private/tmp/colab_transfer/tiny_attn_only/`, under `head_block_forests/`, `head_block_entropy/`, `head_block_scale/`, `head_block_fast/` and `head_block_batch/`.
- Locally, git-ignored: `reports/head_block_forests/`, `reports/head_block_entropy/`, `reports/head_block_fast/`, `reports/head_block_batch/`.

## Traps

- A fresh VM's `results.jsonl` synced to S3 overwrote earlier rows. The bucket has no versioning.
  - Copy S3 results down before appending, or write each run to its own file.
- Colab recycled the VM after about 80 minutes with no cell run.
