# Finding the circuit of the modular addition network with our pipeline

Branch `exp/grokking-neuron-clusters`.
Network: the one-layer transformer of Nanda et al. (P = 113, 4 heads, d_mlp 512), final state `checkpoints/grokking/states/state_40000.pt`.

## Question

- Can the pipeline find circuits (subgraphs) of this network automatically, without reading what each weight does?
- Ground truth is the paper: the network uses a few key frequencies, and heads and MLP neurons work on those frequencies.

## Answer in short

- The pipeline's per-neuron hit clusters do not find the circuit (E1, E2).
  - Clusters of hits are local neighbourhoods, not frequency groups.
- Clustering the neurons themselves does find it (E3).
  - Each neuron is described by its frequency spectrum over the residues, and `cluster_labels` (HDBSCAN, `eom`, min size 10) groups the neurons.
  - The 512 MLP neurons fall into 4 big groups, one per key frequency (9, 34, 38, 53), each 100% frequency-pure.
- The groups are causally real (E4).
  - Mean-ablating the 136 neurons of the frequency 9 group drops accuracy on all pairs from 1.00 to 0.49, and a random set of the same size drops it to 1.00.
- Tracked over training, the groups appear before test accuracy rises (E5).
  - The paper's three phases show up as group count and purity, with no use of the loss.

## What is new in the code

- `Target.UP_PROJ` for the toy model, which makes the 512 MLP pre-activation neurons a normal pipeline target (`src/in_tfm/layers.py`, `neuron_run.py`, `run_config.py`).
  - `q_proj` and `k_proj` are also enabled for it.
- `scripts/cluster_grokking_neurons.py`: scan, hit selection, gather and clustering for every neuron, saved as npz with no HTML.
- `src/in_tfm/grokking_fourier.py`: the paper's Fourier view, used as ground truth and as the neuron description.
- `src/in_tfm/grokking_circuits.py`, `grokking_groups.py`: cluster frequencies, neuron groups, ablation.
- `scripts/grokking_neuron_groups.py`, `scripts/plot_grokking_spectra.py`: E3 to E5 and the figures.
- `scripts/probe_up_proj_hit_clusters.py`, `scripts/probe_token_hit_clusters.py`: the probes behind E1 and E2.

## Ground truth used throughout

- Key frequencies from the token embedding at the final state: 9, 38, 34, 53 (45 is a weak fifth).
- Each MLP neuron has a median 97% of its (a, b) variance in terms where `a` or `b` has one frequency.
  - Strongest frequency per neuron: 9 for 159 neurons, 38 for 146, 34 for 105, 53 for 102.
- The first attempt at this number gave 3% because it dropped the terms that are constant in one operand.
  - Those terms are where most of a pre-activation's variance is.

## Experiments

### E0: value neurons, full reports (aborted)

- `run_neuron_report.py` on neurons 0 to 127 of `v_proj`, output in `exp1_final_all_vproj/`.
- Stopped after neuron 8, because you redirected the goal from reports to circuit discovery.
- Nothing was concluded from it, and E7 later ran reports for every neuron.

### E1: clusters of the hits of MLP neurons

- Setup: `up_proj`, all 12769 pairs, 3000 bucketed hits per neuron, Hadamard product with the neuron's weight row, `leaf` and min size 3 (the pipeline defaults).
  - Output of the first 32 neurons: `final_up_proj.npz`.
- Result: each neuron gets about 60 clusters of 3 to 27 hits.
  - Of the hits at `=`, 66 to 74% are noise.
  - Hits at the `a` and `b` positions form one cluster per token, since a token always gives the same hit.
- Probe on 6 neurons, with `leaf`/3, `eom`/30, `eom`/100 and `leaf`/50, on the `=` hits only (`scripts/probe_up_proj_hit_clusters.py`).
  - Almost every setting gives no cluster of 20 hits or more.
  - The few that exist have at most 17% of their energy at their strongest frequency, and it is often not the neuron's own frequency.
- Check across all clusters, asked after the first write-up: does any hit cluster carry the key frequencies?
  - Over the 1567 clusters of 5 hits or more of the 32 neurons (mean size 6.5, none reaches 20), the energy of the cluster's pair map at the four key frequencies is 0.167.
  - Random pair sets of the same sizes give 0.140, so the enrichment is small.
  - The cluster's strongest frequency is a key frequency for 16% of clusters, and the neuron's own frequency for 7%.
  - For the 118 clusters of 10 hits or more, the own-frequency match is 25%, which is what picking one of the four key frequencies at random gives.
  - So some clusters sit on key frequencies, but no more than chance allows, and none is a clean frequency group.
- Conclusion: the hits of an MLP neuron form no clusters that mean a frequency.
  - A neuron fires on a Fourier-structured set of pairs, and the Hadamard products of those pairs are a continuum, not modes.

### E2: clusters of the hit tokens of value and key coordinates

- Setup: `distinct` samples `(a, a)`, 128 coordinates each of `v_proj`, `q_proj`, `k_proj`.
  - Outputs: `final_v_proj_distinct.npz`, `final_k_proj_distinct.npz`, `final_q_proj_distinct.npz`, and `final_v_proj.npz` for all pairs.
- Result: about 11 to 12 clusters per coordinate, but small (mean 3.6 tokens), so they are neighbourhoods in token space.
  - The strongest frequency of a cluster's token set equals the coordinate's own frequency for 5% of clusters (`v_proj`) and 10% (`k_proj`).
  - With all pairs each cluster is exactly one token, because the 113 copies of a token are identical.
- `q_proj` at `=` does not depend on the pair, so it has nothing to cluster.
- Conclusion: same as E1, hit clusters do not carry the frequency.

### E3: clustering the neurons by spectrum

- Description of a neuron: its energy at each frequency 1 to 56, as unit-norm square roots (Hellinger).
  - MLP neuron: energy of its (a, b) map.
  - `v_proj` and `k_proj` coordinate: energy of its value over the token residues, summed over the two operand positions.
- Grouping: `cluster_labels` with `eom` and min size 10.
  - `leaf` with min size 5 gives dozens of fragments, so `eom` is the setting that matters.
- Result at the final state (`groups_final.json`, `figures/mlp_spectra_state_40000.png`):

| layer | groups | content |
|---|---|---|
| `up_proj` (512) | 6 | sizes 102 (f53), 105 (f34), 128 (f38), 136 (f9), plus two small groups of 18 (f38) and 19 (f9), all purity 1.00, 4 noise |
| `k_proj` (128) | 4 | four groups of 32, two pure at f9 and two at f38 |
| `v_proj` (128) | 2 | about 60 each, f9 and f38 with purity about 0.77 |

- Conclusion: the pipeline's clustering step, applied to neurons, recovers the paper's key frequencies and which neurons serve which one.
  - The group lists are a set of subgraph candidates, one per frequency, over the MLP and the attention coordinates.
- Caveat: the neuron description is the Fourier spectrum, so the grouping uses the paper's insight about what to look at.
  - The key frequencies themselves are not given, the clustering finds the four.

### E4: are the groups causal

- Mean-ablate the post-ReLU activations of a group's neurons at `=`, loss and accuracy over all 12769 pairs (`groups_final.json`).

| group | size | accuracy, group ablated | accuracy, random set of same size |
|---|---|---|---|
| f9 | 136 | 0.49 | 1.00 |
| f38 | 128 | 0.92 | 1.00 |
| f34 | 105 | 1.00 | 1.00 |
| f53 | 102 | 1.00 | 1.00 |
| f38 small | 18 | 1.00 | 1.00 |
| f9 small | 19 | 1.00 | 1.00 |

- Conclusion: the two largest groups are load-bearing, and the small ones do nothing.
  - Each of f34 and f53 can be removed alone with no loss of accuracy, but the loss rises (0.015 and 0.005, against 0.0005 for random sets), so they contribute and are redundant with the rest.

### E5: tracking the groups through training

- 17 states from 0 to 40000, same method each time (`groups_tracking.json`, `figures/mlp_spectra_state_{03000,04400,06000,40000}.png`).
- Test accuracy rises between epochs 4400 and 6000, with a memorisation phase before it.

| state | embedding key freqs | `up_proj` groups (freq, size, purity) | median share of top frequency |
|---|---|---|---|
| 0 to 2000 | none | none | 0.08 to 0.11 |
| 3000 | 38 | 34 (11, 0.73), 38 (23, 0.70) | 0.13 |
| 4400 | 38, 9, 53 | 53 (17), 9 (24), 38 (61) | 0.17 |
| 5200 | 38, 9, 53, 34 | 4 groups, 3 with purity 1.00 | 0.29 |
| 6000 | 4 | 38 (99), 9 (86), 34 (73), 53 (70), purity 0.99 to 1.00 | 0.71 |
| 8000 | 4 | 53 (103), 34 (103), 38 (127), 9 (141) | 0.97 |
| 40000 | 5 | the final 6 groups | 0.97 |

- Group ablation accuracy when the group is removed (random set in brackets):
  - 5600: f9 0.55 (0.85), f38 0.44 (0.84).
  - 6000: f38 0.52 (0.94), f9 0.72 (0.94).
  - 10000: f9 0.45 (0.97), f38 0.73 (1.00).
  - At 4400 and earlier the ablated and random numbers are equal, since the network still memorises.
- Conclusions:
  - Frequency groups exist from state 3000, before test accuracy moves (circuit formation).
  - Groups become pure around 4800 to 5200, and each of the 4 key frequencies has its group by 5600.
  - From 6000 to 8000 neurons keep joining and the spectra sharpen (median share 0.71 to 0.97), which is the cleanup phase.
  - The groups become causal at 5600, as test accuracy climbs, and not before.
  - After 10000 the groups are stable, and two small groups split off (20000, 40000).

### E6: the edges, which heads feed which MLP group

- For every MLP neuron, each head's contribution to its pre-activation at `=` (`o_proj` and `up_proj` are linear), and the share of the neuron's variance it explains (`edges_final.json`, `scripts/grokking_circuit_edges.py`).
- Mean share per head, by neuron group:

| group | size | head 0 | head 1 | head 2 | head 3 |
|---|---|---|---|---|---|
| f9 | 136 | 0.50 | 0.00 | 0.50 | 0.00 |
| f9 small | 19 | 0.31 | 0.03 | 0.63 | 0.03 |
| f38 | 128 | 0.00 | 0.48 | 0.00 | 0.51 |
| f38 small | 18 | 0.06 | 0.39 | 0.08 | 0.48 |
| f34 | 105 | 0.25 | 0.24 | 0.25 | 0.26 |
| f53 | 102 | 0.26 | 0.23 | 0.26 | 0.25 |

- Heads and frequencies, from the coordinates of each head:
  - All 32 `k_proj` coordinates of heads 0 and 2 have strongest frequency 9, and all 32 of heads 1 and 3 have 38.
  - `v_proj`: heads 0 and 2 have 24 of 32 coordinates at 9, heads 1 and 3 have 23 and 26 of 32 at 38, the rest at 34 and 53.
- Averaged over all pairs, every head attends 0.5 to `a`, 0.5 to `b` and 0 to `=` from the `=` position.
  - The 0.5 split is forced by the symmetry between `a` and `b`, so it says nothing about how attention varies per pair.
  - Only the 0 on `=` is informative.
- Traced circuit:
  - Heads 0 and 2 (f9) feed the f9 neuron group, and heads 1 and 3 (f38) feed the f38 group.
  - The f34 and f53 groups read all four heads equally, so no head owns those frequencies.
  - The groups write to the logits, and E4 shows the f9 and f38 groups are the load-bearing ones.
- This matches the paper's picture of neurons working on key frequencies, with heads reading the operands.
  - The specialisation of heads to frequencies (0/2 to f9, 1/3 to f38) is something I only saw in these edges, I did not check it against the paper.

### E7: neuron reports for every neuron

- `experiments_grokking/run_all_reports.sh` runs `scripts/run_neuron_report.py` on every neuron of the final state, output in `experiments_grokking/reports/<target>/neuron_<i>/index.html` (619 MB, not committed).
  - `v_proj`, `k_proj`, `q_proj`: 128 each, `distinct` samples, the default `leaf` and min size 3.
  - `up_proj`: 512, all pairs, 2000 hits, min cluster size 20 to keep the reports readable and fast.
  - At most 3 hits are shown per cluster.
- All 896 reports were written (`index.html` counted), and every run printed its total time.

| target | neurons | clusters, median (range) | hits, median | run time |
|---|---|---|---|---|
| `v_proj` | 128 | 13 (6 to 19) | 218 | 605 s |
| `k_proj` | 128 | 11 (8 to 15) | 229 | 591 s |
| `q_proj` | 128 | 14 (14 to 23) | 90 | 913 s |
| `up_proj` | 512 | 6 (0 to 13) | 2000 | 1545 s |

- These are the per-neuron views behind E1 and E2.
  - The circuit trace (E3 to E6) does not depend on them.

## Not yet done

- The edges from the groups to the logits are only measured by ablation (E4), not by a path decomposition.
- Edges are computed at the final state only, not tracked through training.
- The neuron description uses the Fourier spectrum, so it is not a clustering with no prior at all.
  - A prior-free version would describe a neuron by its raw activation map over the pairs, and was not tried.
- `up_proj` hit clustering was run for 32 neurons only, then stopped, since E1 already showed noise.
- Everything is one training run, with one seed.
