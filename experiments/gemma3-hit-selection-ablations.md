# Gemma 3 270m: hit selection and clustering ablations

Run in 2026-10 on Colab.
Layer 11 (a full-attention layer) of `google/gemma-3-270m`'s `q_proj` / `k_proj`, unless noted.

## Findings

- **ablate-hadamard-vs-input**: does the Hadamard weighting change what clusters?
  - Layer 0: barely. The residual there is essentially the token embedding.
  - Layer 11: it differs more, but clusters are still mostly single-token.
  - Still runnable on `main` with `--clustering.on input`.
- **ablate-select-on-q-norm**: `q_proj` is hooked before Gemma's `q_norm` and rope.
  - Does selecting hits on the post-norm value change which hits get picked?
  - Neuron 300 went from one 4,313-hit blob to 28 well-formed clusters.
  - Neuron 90 lost its cleanest cluster (`of`/`in` after a place noun).
  - Inconclusive on its own, since inspection gives no ground truth for the right hits.
- **occlusion-test**: zero the neuron's coordinate at the `q_norm` output, at each selection's hits.
  - Then measure the change in next-token loss.
  - Both selections' top hits matter about equally.
  - The hits only raw selection finds matter more than the hits only norm selection finds.
  - So raw `q_proj` selection is at least as good functionally.
    - It also keeps the exact Hadamard decomposition.
- **hdbscan-leaf**: does `cluster_selection_method=leaf` split neuron 300's blob without changing selection?
  - Yes, into 23 clusters.
  - 12 of the top 20 pairs match the norm-selection clusters almost exactly.
    - Matched by linear sum assignment on cosine similarity between `mean.pt` vectors.
  - `leaf` became the default.
- **elbow vs bucketed**: the same scan and budget under both hit selections.

## Reproduce

- Branches: [`exp/select-on-q-norm`](https://github.com/narang99/in-tfm/tree/exp/select-on-q-norm), [`exp/occlusion-test`](https://github.com/narang99/in-tfm/tree/exp/occlusion-test).
- Elbow vs bucketed: [`scripts/ablate_hit_selection.py`](https://github.com/narang99/in-tfm/blob/0c792524ada4568bd24eac7bfea2795820e6a864/scripts/ablate_hit_selection.py), removed from `main` after [`0c79252`](https://github.com/narang99/in-tfm/tree/0c792524ada4568bd24eac7bfea2795820e6a864).
  - `git checkout 0c792524a`, then `uv run python scripts/ablate_hit_selection.py --config configs/gemma3_down_proj.yaml --out-dir experiments/ablate-hit-selection`
- The per-ablation READMEs with exact numbers were local only, in git-ignored `experiments/<name>/` folders.
