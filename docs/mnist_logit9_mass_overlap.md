# MNIST 2/4 logit 9: a shared-dimension score for Hadamard products

Run locally on 2026-10-19 and 2026-10-20, on a Mac.

- Most numbers here came from throwaway scripts, which are not kept.
  - "How it was calculated" below says what each one did.
- The clustering run used `--clustering.distance mass_overlap`.
  - That option lives on the local branch `feat/overlap-distance`, commit `e36bc7cef`, and is not part of any PR.

## Question

- Cosine HDBSCAN leaves most of logit 9's hits as noise.
- Can a score that asks "which inputs drive these two hits" group them better?
- What scale does that score have, and can it set per-cluster thresholds?

## Setup

- Model: the 2/4 MNIST net, `checkpoints/mnist_cnn_2_4/final.pt`.
- Layer: `head.1`, the linear layer, logit 9.
  - Its input is conv2's output, 4 channels of 6x6, flattened channel by channel into 144 dims.
- Data: the first 100 MNIST test images of each digit, 1000 images.
- Hits: bucketed sampling with a floor of 0.
  - A logit has one activation per image, so every positive logit is a hit.
  - Logit 9 has 221 hits.
- Baseline report: `reports/mnist_cnn_small_all/linear/neuron_9`.
  - Euclidean distance on L2-normalised Hadamard products.
  - `min_cluster_size` 5, `min_samples` 3, leaf selection.
  - 4 clusters, 178 of 221 hits are noise.

| Digit | Hits | In a cluster | Median activation |
|---|---|---|---|
| 9 | 99 | 22 | 4.00 |
| 7 | 56 | 20 | 1.65 |
| 4 | 42 | 1 | 0.86 |
| others | 24 | 0 | below 1.8 |

## The score

- For each hit's Hadamard product:
  - The positive set is the dims with the largest positive values, taken until they reach 90% of the positive total.
  - The negative set is the same for negative values, by magnitude.
- The score of two hits is the number of dims both keep.
  - Positive dims count against positive dims, negative against negative.

### The sign split adds nothing on this layer

- conv2's output passes a ReLU, so every input is 0 or above.
- So a dim's sign in the Hadamard product is always its weight's sign.
  - Logit 9 has 57 positive-weight dims and 87 negative-weight dims.
  - Every positive set held only positive-weight dims, and every negative set only negative-weight dims.
- So the score is simply the count of shared dims.
- Layers whose inputs can be negative, such as a transformer's `q_proj`, would still need the split.

### Scale

- A hit's set holds 29 to 44 dims, with a median of 37.
- Over all 24,310 pairs of the 221 hits, the score is one bell, with a median of 24 and a range of 10 to 37.
  - Plot: `reports/mnist_cnn_small_all/linear/neuron_9/overlap_score_hist.png`.

| Reference | Median score | p10 to p90 |
|---|---|---|
| Chance: random sets of the same sizes | 10.2 | |
| Different digits | 24 | 19 to 28 |
| Same digit, not the same cluster | 25 | 20 to 30 |
| Same cosine cluster | 29 | 26 to 32 |
| The same image shifted 1 px | 31 | 28 to 35 |
| A hit against itself | 37 | |

- The useful band is narrow, from about 24 for an ordinary pair to about 31 for the same image shifted.
- A typical pair already shares about two thirds of its dims.
  - A likely cause is a core of dims active in nearly every image, such as conv2 channel 0's background cells.
  - This was not checked.

## Clustering with the score

- Distance: the highest score between two different hits, minus each pair's score, with a zero diagonal.
  - It is not a metric.
- It runs on sklearn's HDBSCAN with a precomputed matrix.
  - sklearn counts a point as its own neighbour, so `min_samples` gets +1.
  - With that, sklearn and the `hdbscan` package gave the same labels on euclidean distances at the report's settings.
- Run: `--clustering.distance mass_overlap` with the report's other settings.
  - Report: `reports/mnist_cnn_small_all/linear_mass_overlap/neuron_9`.

| | Cosine clusters | Score clusters |
|---|---|---|
| Clusters | 4 | 22 |
| Noise | 178 of 221 | 45 of 221 |
| Fours in a cluster | 1 of 42 | 32 of 42 |
| Within-cluster mean cosine | 0.915 to 0.948 | 0.586 to 0.844 |

- Most score clusters mix 9s, 7s and 4s.
- Six score clusters (0, 1, 2, 3, 4 and 7) are looser than a random pair by both measures.
  - Their mean cosine is 0.59 to 0.67, against 0.724 over all pairs.
  - Their mean score is 18.8 to 22.2, against 24 over all pairs.
  - Clusters 1, 2 and 4 are also the lowest-activation clusters, with medians of 0.24 to 0.45.
- Likely cause: the raw count rewards large sets.
  - Hits with small sets score low against everyone, so they end up near each other.
  - This was not checked.
- Clusters with high scores also have high cosine.
  - Clusters 16, 18, 19 and 20 score 28 to 30 and have cosines of 0.83 to 0.84.
- The report's grouping step still uses cosine of cluster means at 0.98, plus the median activation gap.
  - So clustering and grouping measure similarity differently in this report.

## Per-cluster thresholds on the cosine clusters

- Idea: each cosine cluster has its own similarity cutoff, and noise hits above it could join.
- A member's best match is its highest score with another member of the same cluster.
- A cluster's threshold is the lowest best match among its members.

| Group (cluster) | Hits | Threshold | Median best match |
|---|---|---|---|
| 0 (cluster 1, the 7s) | 19 | 31 | 33 |
| 1 (cluster 2, 9s) | 11 | 28 | 31 |
| 2 (cluster 3, 9s) | 8 | 29 | 31 |
| 3 (cluster 0, 9s) | 5 | 33 | 34 |

- In groups 1 and 2, the lowest best match belongs to the one member of another digit.
  - The lone 4 in group 1 and the lone 7 in group 2 both score 29.
- Noise hits look just as connected by this measure.
  - Each noise hit's best score with any other noise hit has a median of 32.

### Noise hits that pass a threshold

- A noise hit passes a cluster if its best score with any member reaches that cluster's threshold.

| Group | Threshold | Noise hits that pass | Digits |
|---|---|---|---|
| 0 | 31 | 18 | 15 sevens, 2 fours, 1 nine |
| 1 | 28 | 101 | 54 nines, 28 fours, 12 sevens, 7 others |
| 2 | 29 | 71 | 42 nines, 23 fours, 3 sevens, 3 others |
| 3 | 33 | 2 | 2 nines |

- Of the 178 noise hits:
  - 42 pass exactly one cluster.
  - 72 pass two or more, and 58 of those pass both group 1 and group 2.
  - 64 pass none, with a median activation of 0.83.
- The 7s cluster stays clean.
- Groups 1 and 2 take in most of the 9s and many 4s.

### Scores of 28 to 30 in groups 1 and 2

| Group | Count of | 28 | 29 | 30 | 31 or more |
|---|---|---|---|---|---|
| 1 | member pairs (55) | 12 | 10 | 12 | 8 |
| 1 | members' best match (11) | 1 | 1 | 0 | 9 |
| 1 | noise hits' best score against the group | 26 | 20 | 24 | 31 |
| 2 | member pairs (28) | 0 | 7 | 7 | 9 |
| 2 | members' best match (8) | 0 | 1 | 1 | 6 |
| 2 | noise hits' best score against the group | 13 | 23 | 14 | 34 |

- 28 to 30 is an ordinary score for two members of these groups.
- Many noise hits score the same against the groups.
  - So no single threshold separates members from noise.

## Conclusion

- The raw count of shared dims is too brittle to cluster with or to threshold on.
  - It depends on set size, which varies from 29 to 44 dims per hit.
  - Its useful range is about 7 points, from 24 to 31.
  - A threshold taken from a cluster's weakest member is set by one outlier.
- It does agree with cosine where hits share most of their dims, at scores of 28 and above.

## Next options

- Jaccard: divide the shared count by the size of the union.
  - It removes the set-size effect, runs from 0 to 1, and 1 minus it is a metric.
- Drop or down-weight dims present in nearly every hit, as IDF does for words.
  - This targets the shared core that lifts every pair.
- Weighted Jaccard (Ruzicka), on the values instead of 0/1 sets.
  - It needs no 90% cutoff.
- For thresholds, use a typical member, such as the median best match, instead of the weakest.
- Assign a hit that passes several clusters to the one it scores highest against.

## How it was calculated

### Hadamard products

- Run the checkpoint on the 221 hit images, found through the baseline report's `hit_table.npz` and `sample_ids`.
- Take conv2's output after its ReLU, flattened to 144 dims.
- Multiply it elementwise by row 9 of `head.1`'s weight, then L2-normalise each row.
- Check: the recomputed activations match the report's, and the recomputed cluster means match each `mean.pt` at cosine 1.0000.

### The 90% mass sets

- For the positive set, keep only the positive values and sort them, largest first.
- Walk down the sorted list, adding each dim while the sum of the dims before it is under 90% of the positive total.
  - So the set is the fewest dims whose sum reaches 90%.
- For the negative set, do the same on the magnitudes of the negative values.
- A row with no values of a sign gets an empty set for it.
- L2 normalisation does not change the sets, since it scales every value equally.

### Score and distance

- Score of two hits: the size of the intersection of their positive sets, plus the size of the intersection of their negative sets.
- Distance for clustering: the highest score between two different hits, minus the pair's score.
  - The diagonal is set to 0, since a hit's score with itself is its own set size.

### Reference points

- Chance: for each pair, |P_a| x |P_b| / 57 + |N_a| x |N_b| / 87.
  - This is the expected overlap of random sets of the same sizes.
  - Positive sets are drawn from the 57 positive-weight dims and negative sets from the 87 negative-weight dims.
- Pair kinds: the median over every pair of a kind, using the image labels and the report's cluster labels.
- 1 px shift: roll each hit image by one pixel up, down, left and right.
  - Recompute its Hadamard product and sets.
  - Score each shifted copy against the original, giving 4 scores per hit.

### Per-cluster thresholds

- A member's best match: its highest score with any other member of its cluster.
- A cluster's threshold: the lowest best match among its members.
- A noise hit's score against a cluster: its highest score with any member.
  - It passes the cluster if that score reaches the threshold.
- Clusters are the 4 cosine clusters of the baseline report, and noise is its 178 noise hits.

## Files

- Pages beside the baseline report, in `reports/mnist_cnn_small_all/linear/neuron_9`:
  - `rare_classes_hadamard.html` and `rare_classes_spatial.html`: digits with fewer than 20 hits.
  - `digit_4_hadamard.html`: the 42 fours, ordered by similarity.
  - `overlap_score_hist.png`: the score histogram.
