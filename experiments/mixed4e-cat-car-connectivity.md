# Are the cat and car clusters of mixed4e:55 connected?

Run around 2026-10-23.

## Question

- On InceptionV1 mixed4e:55, is there a direct route between the cars cluster (0) and the cats cluster (1)?

## Setup

- 14,240 hits of mixed4e:55, with the density-peak forest.
- Two checks:
  - Edges between clusters in the mutual kNN graph of the hits, at k = 5, 10 and 20.
  - A page for every pair of clusters that share mixed hits.
    - A hit counts for a pair when those are its top two clusters and the second has at least 0.1.

## Code

- Branch [`feat/inception-cluster-space`](https://github.com/narang99/in-tfm/tree/feat/inception-cluster-space), closed PR #113.
  - [`scripts/collect_hit_space.py`](https://github.com/narang99/in-tfm/blob/42b1282a7099720737d2137dccf1a5c85ad9f225/scripts/collect_hit_space.py) saves the hits, distances, forest and probabilities.
  - [`scripts/cluster_connectivity.py`](https://github.com/narang99/in-tfm/blob/42b1282a7099720737d2137dccf1a5c85ad9f225/scripts/cluster_connectivity.py) counts mutual kNN edges between clusters.
  - [`scripts/pair_mixed_report.py`](https://github.com/narang99/in-tfm/blob/42b1282a7099720737d2137dccf1a5c85ad9f225/scripts/pair_mixed_report.py) writes a page per pair.
  - Config: [`inception_v1_mixed4e_peaks.yaml`](https://github.com/narang99/in-tfm/blob/42b1282a7099720737d2137dccf1a5c85ad9f225/configs/inception_v1_mixed4e_peaks.yaml).

```
python scripts/collect_hit_space.py --config configs/inception_v1_mixed4e_peaks.yaml \
    --inception.image-dir /content/data/inception_pool --out-dir /content/runs/mixed4e_55
python scripts/cluster_connectivity.py /content/runs/mixed4e_55/space --ks 5 10 20
python scripts/pair_mixed_report.py --config configs/inception_v1_mixed4e_peaks.yaml \
    --inception.image-dir /content/data/inception_pool --out-dir /content/runs/mixed4e_55
```

## Results

- Cars and cats have zero mutual kNN edges at k = 5, 10 and 20.
  - That holds on confident hits and on all hits.
- There is no `pair_0_1` page.
  - So fewer than 30 hits have 0 and 1 as their top two clusters.
- Going by page filenames, 0 and 1 share no mixing partner.
  - A shortest route goes 0, 3, 17, 7, 1.
- The reading: a hit that is half cat and half car does not exist. Any route between them goes through the background.

## Caveats

- The test is mutual-only.
  - The membership graph (`probabilities.knn_graph`) keeps an edge if either end lists the other.
  - Nobody has counted cat-car edges on that graph.
- `edge_0_1_k20.png` in the outputs is probably a false edge.
  - It comes from a labelling bug fixed in [`f874f21`](https://github.com/narang99/in-tfm/commit/f874f21).

## Outputs

- `s3://narang99-private/inception_pool/outputs/` holds `space.tar` and `pair_report.tar`.
