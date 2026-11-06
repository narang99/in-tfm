# A density-peak forest of one o_proj neuron, from a scan of the whole training stream

Run 2026-11-06 on a Colab L4 (52 GB of RAM).

## Setup

- Model: `checkpoints/tiny_attn_only_1l` (d_model 256, 8 heads).
- Neuron: 6 of `model.layers.0.self_attn.o_proj`, signed.
- Corpus: `train_tokens.npy`, read as story-aligned windows of 256 tokens with no overlap.
  - 1,239,605 windows, so 317,338,880 positions.
  - That is 65% of the 486M stream. The rest is the tails of stories that a window boundary cut.
- Hits: bucketed per tail, 250k in total, so the whole activation range is sampled.
- Forest: IVF, weighted Jaccard on raw Hadamards, 1,000 lists, 64 lists probed.
  - Copies are not merged, unlike the exact forest.

## Code

- Branch `exp/o-proj-full-scan`, config [tiny_attn_only_1l_o_proj_full_scan.yaml](https://github.com/narang99/in-tfm/blob/89871e2/configs/tiny_attn_only_1l_o_proj_full_scan.yaml).
- Command:

```
python scripts/run_neuron_report.py --config configs/tiny_attn_only_1l_o_proj_full_scan.yaml
```

- Needs `checkpoints/tiny_attn_only_1l` and `data/tiny_attn_only/train_tokens.npy`, from `s3://narang99-private/tmp/colab_transfer/tiny_attn_only/`.

## Results

- Time: 27 min in total. Scan 16 min, gather 3 min, clustering 2.3 min, report 5 min.
  - The IVF forest itself took 32 s.
- Peak RAM: 11.6 GB, during clustering.
  - A T4 has 12 GB. An earlier T4 run was killed at this step, before the scan's memory was freed.
- A rerun picked identical hits and rows to the killed run.
- 93 trees. The largest holds 11.4% of the points and the median tree 754.
- 189,294 of 250k hits are too mixed to show (entropy of 0.5 bits or more).

## Copies inflate the confident share

- 24.3% of points are confident, but 12.3% have a top probability of exactly 1.
  - Those are not confident in the usual sense. Most sit in groups of identical rows.
  - 216,360 of 250k rows are distinct, and the largest group of copies holds 5,240 points.
- A group larger than k lists only itself as neighbours, so it splits off from the graph.
  - The graph has 231 pieces. The largest holds 219,166 points.
  - 24,348 points (9.7%) sit in a piece with no root, so they reach no peak and fall back to their forest tree.
- Without those 30,834 points, about 12% of points are confident.
- Every one of the 30,834 lies outside the largest piece.

## Links between trees join sparse points

- Measured over every link at k=5, labelled by forest tree. A point's density share is its density rank inside its tree, from 0 to 1.

| Links | Count | Median sparser end | Both ends in the top half | Both ends in the top tenth |
|---|---|---|---|---|
| Within a tree | 900,025 | 0.57 | 56.4% | 7.33% |
| Between trees, either-way | 96,374 | 0.10 | 4.1% | 0.02% |
| Between trees, mutual | 11,877 | 0.13 | 5.3% | 0.01% |

- 5.8% of points in the top tenth of their tree have a link to another tree, against 34.0% of the rest.
- Mixed hits are rarer than links, and not sparse.
  - 1,059 hits (0.4%) have their top two probabilities both at 0.4 or more.
  - Their median density share is 0.47, close to the 0.50 of all points.

## Outputs

- `reports/o_proj_full_scan_neuron_6/` locally, git-ignored, without the `hits.pt` files.
- `neuron_6_signed_gathered.npz` holds the hits, rows, noise sample, curves and bucket edges, so clustering can change with no forward pass.
