# Do Hadamard clusters survive a rotation of the residual stream?

Run 2026-10-29.

## Question

- Do a neuron's per-coordinate Hadamard clusters depend on the residual stream's basis?

## Setup

- Model: attn-only 2L, layer 1 `q_proj` neuron 3, signed.
- A random rotation is applied to every writer and reader of the residual stream.
  - The rotated model computes the same function.
- Corpus: pile-10k. 20k hits are picked once, and both models report the same hits.

## Code

- Branch [`exp/rotated-model`](https://github.com/narang99/in-tfm/tree/exp/rotated-model).
  - Script: [`scripts/rotated_model_report.py`](https://github.com/narang99/in-tfm/blob/4e00fa29685b2212999a714f6a784dd9e5fc659a/scripts/rotated_model_report.py).
  - Config: [`attn_only_2l_signed_q_proj_l1_pile.yaml`](https://github.com/narang99/in-tfm/blob/4e00fa29685b2212999a714f6a784dd9e5fc659a/configs/attn_only_2l_signed_q_proj_l1_pile.yaml).

```
python scripts/rotated_model_report.py --rotation-seed 1 --config configs/attn_only_2l_signed_q_proj_l1_pile.yaml
```

## Results

- Control, on raw inputs with euclidean distance: ARI 0.98 to 1.0, density correlation 0.99999.
  - So the setup is sound.
- Hadamard rows with weighted Jaccard:
  - Density per hit correlates at 0.99989.
  - Tree ARI is only 0.21, and the most-probable-tree ARI 0.14.
- Hits confident in both models (590 of them) agree at ARI 0.88.
  - Distinctive clusters with large activations match at Jaccard 0.83 to 1.0.
    - Examples are licence boilerplate and citation markup.
  - Generic tokens (newline, whitespace, " I") split, merge or turn mixed.
- Sparse points spread over 14 to 19 trees, with a top probability of about 0.2.
  - A tiny change in distance flips their tree.

## Takeaway

- Confident clusters can be trusted across bases.
- The low-confidence mass depends on the basis.

## Note

- Sampling twice gave different hits, because float32 rounding moves the bucket edges.
  - Reports on this branch save `hit_positions.npz`, so a selection can be reused.
