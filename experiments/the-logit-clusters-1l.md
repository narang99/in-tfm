# HDBSCAN clusters of the " the" logit, 1L attn-only d256

Run 2026-11-04 on a Colab L4 with cuml 26.06 HDBSCAN (GPU).

## Setup

- Model: `checkpoints/tiny_attn_only_1l` (d_model 256, 8 heads).
- Logit: `" the"`, token id 263.
- Point per hit: `x1 ⊙ W_U[263]`, 256 dims, raw (no normalisation), euclidean.
  - `x1` is the final residual, so the dims sum to the logit minus the unembed bias.
- Corpus: 4M positions from the start of the training stream.
  - 15,625 windows of 256 tokens, each starting at a story's BOS, with no overlap.
  - The val stream only holds 3.2M positions this way, so it was not used.
  - The stream slice is `s3://narang99-private/tmp/colab_transfer/tiny_attn_only/tiny_attn_only_1l/train_head_8m.npy`.
- Hits: signed bucketed sampling, 1M budget, 10 buckets per tail, floor 0, min bucket size 100.
  - Negative tail: 616,560 candidates, 154,140 kept.
  - Positive tail: 3,383,440 candidates, 845,860 kept.
- HDBSCAN: leaf selection, min_cluster_size 50, min_samples left at its default (50).

## Code

- Script: [scripts/the_logit_clusters.py](https://github.com/narang99/in-tfm/blob/c044704/scripts/the_logit_clusters.py) on branch `exp/the-logit-clusters`.
- Command:

```
python scripts/the_logit_clusters.py --checkpoint <1l ckpt dir> --tokens train_head_8m.npy --out-dir <out>
```

## Results

- Outputs are in `s3://narang99-private/tmp/colab_transfer/tiny_attn_only/the_logit_clusters/`, with a local copy in `reports/the_logit_clusters/` (git-ignored).
  - `hits.npz` holds, per hit, the flat position, tail, bucket, label, logit, current and next token.
  - `clusters.json` holds a summary per cluster.
- Time: forward pass 6 s, HDBSCAN 408 s, nearly all of it in the brute-force kNN.
- 646 clusters, with 81.5% of hits as noise.
  - Noise is 75.9% in the negative tail and 82.6% in the positive tail.
  - Cluster sizes range from 50 to 8,348, with a median of 140.
  - 125 clusters have no single current token at 90% or more.
- The largest clusters key on the current token:
  - `" upon"`: two clusters, 8.3k and 7.8k, logit +11.
  - `" a"` and `" little"`: negative logit clusters, -5 to -8.
  - `" to"`, `" in"`, `" opened"`: logit +11 to +12, and next token is " the" 55-80% of the time.

## Follow-ups on the same branch

- Runs at 1M and 250k hits, min_cluster_size 50.
  - 1M gave 651 clusters. 250k gave 185.
  - 250k kept the confident clusters whole, but lost 36% of the 1M run's clustered hits, mostly from diffuse clusters.
  - [`hdbscan_predict_coverage.py`](https://github.com/narang99/in-tfm/blob/5e272532aab6c38a798b6ec59d48d817f23daeef/scripts/hdbscan_predict_coverage.py) measures that share.
  - [`compare_cluster_reports.py`](https://github.com/narang99/in-tfm/blob/5e272532aab6c38a798b6ec59d48d817f23daeef/scripts/compare_cluster_reports.py) matches two reports' clusters by the cosine of their mean prototypes.
- Soft HDBSCAN made more noise than hard: 920k against 824k hits.
  - The vectors are near one-hot or near uniform.
  - Script: [`hdbscan_soft_compare.py`](https://github.com/narang99/in-tfm/blob/5e272532aab6c38a798b6ec59d48d817f23daeef/scripts/hdbscan_soft_compare.py).
- Clusters picked for a closer look, by 1M report id:
  - c519, the obvious one: " in" after "went for a walk".
  - c647, the diffuse one: a newline.
  - c537 ("and said"), c304 ("went to", too easy) and c525 ("wanted").
- o_proj neurons of a logit cluster: the residual dims covering 90% of the Hadamard sum, over 12 spread and deduplicated examples near the cluster mean.
  - c519 gives 46 neurons. c647 adds 18 more.
  - A dim's term includes the token embedding, not only the heads. That split is not done here.
  - Script: [`o_proj_neurons_from_cluster.py`](https://github.com/narang99/in-tfm/blob/5e272532aab6c38a798b6ec59d48d817f23daeef/scripts/o_proj_neurons_from_cluster.py).
- A batch of reports for those 64 o_proj neurons.
  - 250k hits each, min_cluster_size 12, 4M positions.
  - 700 to 1,200 clusters per neuron, about 9 min each on an L4.
  - Outputs are in `s3://narang99-private/in-tfm/tiny-attn-only-1l/model.layers.0.self_attn.o_proj/`, 11 GB in total.
    - Each neuron keeps `index.html`, `cluster_*/mean.pt` and `cluster_*/hadamard.jpg`.
  - Runner: [`s3_report_batch.py`](https://github.com/narang99/in-tfm/blob/5e272532aab6c38a798b6ec59d48d817f23daeef/scripts/s3_report_batch.py), with config [`tiny_attn_only_1l_o_proj_l0.yaml`](https://github.com/narang99/in-tfm/blob/5e272532aab6c38a798b6ec59d48d817f23daeef/configs/tiny_attn_only_1l_o_proj_l0.yaml).

```
python scripts/s3_report_batch.py --config configs/tiny_attn_only_1l_o_proj_l0.yaml --neurons '[6, 8]' --s3-prefix <s3 prefix>
```

- The dependency pages of c519 and c647 are in [the-logit-dependencies-d256.md](the-logit-dependencies-d256.md).
