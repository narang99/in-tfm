# Clustering one writer neuron's hits, picked by the stream-writers walk

## Setup

- TinyStories-1M, layer 6 MLP write, dim 43 (`h.6.mlp.c_proj`, neuron 43).
- 80 TinyStories validation stories, up to 128 tokens each.
- A token is a hit only where the stream-writers walk keeps that write, at share 0.9.
  - The walk starts at every logit and splits each stream dim into every earlier write.
  - Scores start no walks, since the QK circuit is studied separately. Nothing is ablated.
- Hadamards over the 256 MLP dims feeding `c_proj`, weighted Jaccard, no normalisation.

```
python scripts/writer_neuron_report.py --config configs/tinystories_1m_l6_mlp_write.yaml --n-stories 80
```

## Hits

- The walk keeps 9,450 of 10,031 tokens: 7,177 negative and 2,273 positive.
  - Dim 43 is kept at 96% of tokens, so at share 0.9 the walk barely filters this write.
- The walk took 20 minutes on 6 CPU workers. Its picks are saved in `walk_picks.pt` and reused.

## HDBSCAN, min_cluster_size 10

| selection | clusters shown | hits in them |
|---|---|---|
| leaf | 121 | 4,277 (45%) |
| eom | 99 | 5,179 (55%) |

- Clusters are mostly token by token: " the", names, " and", " was", ",", "\n".
- 71 of the 80 stories open with "Once upon a time, there was a".
  - Each token of that opening is the same hit in every story, so it becomes its own cluster of 71 copies.
  - leaf splits some of those copies into several clusters, which can only come from how ties break.

## HDBSCAN merges

- `scripts/hdbscan_merge_report.py` fits HDBSCAN once and draws its tree, with a cluster card at every node.
  - Leaf and eom both come from that one tree: 221 nodes, 111 leaf clusters and 99 eom clusters.
- The top of the tree sheds the shared story opening one group at a time: " upon", "Once", " named".
- Lower down, merges read as word groups. " boy" and " girl" join first, then " man".
- HDBSCAN joins two groups at their closest pair of hits, so one close pair can join groups that look different.

```
python scripts/hdbscan_merge_report.py --config configs/tinystories_1m_l6_mlp_write.yaml \
    --picks-from reports/tinystories-1m/h.6.mlp.c_proj --min-cluster-size 10 \
    --out reports/tinystories-1m/h.6.mlp.c_proj_hdbscan_merges
```
