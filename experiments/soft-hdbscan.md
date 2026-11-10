# Soft HDBSCAN on the attn-only 2L q_proj pile run

Run 2026-10-29.

## Question

- Does soft HDBSCAN give a cleaner picture than the density-peak forest on the same hits?

## Setup

- Same corpus, distance and normalisation as the density-peak run of attn-only 2L `q_proj` neuron 3, signed, on pile-10k.
- Every hit gets a probability per cluster, with weighted Jaccard as the distance.
- 12,000 hits, not 20,000.
  - The hdbscan package keeps several n x n float64 matrices.
  - At 20,000 hits they ran out of a standard T4 VM's 12 GB of RAM.

## Code

- Branch [`feat/soft-hdbscan`](https://github.com/narang99/in-tfm/tree/feat/soft-hdbscan).
  - Config: [`attn_only_2l_signed_q_proj_pile_soft_hdbscan.yaml`](https://github.com/narang99/in-tfm/blob/531020426648e54a7b8e6bc35462e24644180a8c/configs/attn_only_2l_signed_q_proj_pile_soft_hdbscan.yaml).

```
python scripts/run_neuron_report.py --config configs/attn_only_2l_signed_q_proj_pile_soft_hdbscan.yaml
```

## Results

- About 59% of hits went to noise.
  - The density-peak forest put them in one large background cluster instead.
- The biggest clusters were piles of one (token, position) pair, such as `.` at position 70.
  - Layer 0 q sees only the token and its position.
