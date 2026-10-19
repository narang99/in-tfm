# MNIST 2/4 logit 9: distances and decompositions for Hadamard products

Run locally on 2026-10-19 and 2026-10-20, on a Mac.

- Most numbers here came from throwaway scripts, which are not kept.
  - "How it was calculated" below says what each one did.
- The first half is about a shared-dimension score.
- The second half, from "One hit's sorted values", tries decompositions, PCA and weighted Jaccard.
- The library's distance option lives on the local branch `feat/overlap-distance`, and is not part of any PR.
  - Commit `e36bc7cef` added `--clustering.distance mass_overlap`, the shared-dimension score.
  - Commit `ed6df3b13` replaced it with `--clustering.distance weighted_jaccard`.

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

## One hit's sorted values

- Hit: test image 678, the 9 with the median logit 9 among the 99 nines, at 4.00.
- Its 144 raw Hadamard values, signed and not normalised, sorted largest first.
  - They sum to 3.92, and the bias of 0.07 brings that to the logit.
- Plot: `reports/mnist_cnn_small_all/linear/neuron_9/one_nine_sorted_values.png`.
- The positive side has a sharp elbow.
  - The top 8 values run from 1.35 down to 0.83, then the next drops to 0.50.
  - Those 8 carry about 64% of the positive sum.
- 42 values are exactly 0, where conv2 output nothing.
- The logit is a small difference of two large sums.
  - 42 positive values sum to +12.76, and 60 negative values sum to -8.84.

| Share of the positive sum | Values needed | Smallest kept value |
|---|---|---|
| 70% | top 10 | 0.48 |
| 80% | top 13 | 0.37 |
| 90% | top 19 | 0.15 |

- So the 90% cut used for the score keeps about 11 small values past the elbow.
- Averaging sorted |values| over all 99 nines hid the elbow.
  - Each nine's elbow sits at a different rank and height.
  - Taking magnitudes also mixed the positive and negative sides.

## Decompositions of the 99 nines

- Input: the 99 nines' L2-normalised Hadamard products.
- Pages beside the baseline report: `digit_9_dictionary.html`, `digit_9_nmf.html`, `digit_9_pruned_2x.html`, `digit_9_pruned_3x.html`.

### Dictionary learning acts like clustering

- 10 latents, sparsity α = 0.05 on the codes, codes of 0 or more.
- It explains 75% of the variance around the mean nine, with a median of 4 latents per nine.
- Each latent looks like a whole nine.
  - The penalty is on the codes, so the cheapest fit gives each latent the shared core plus one variant.
  - A nine then picks its nearest latent and adds small corrections.

### NMF gives parts

- NMF with 16 latents on the absolute Hadamard products, with each latent given back its weights' signs.
  - Inputs come out of a ReLU, so each value already has its weight's sign, and nothing is lost.
- It explains 87% of the variance, with a reconstruction cosine of 0.989.
- A nine uses about 7 latents, and its heaviest carries only 24% of its weight.
- Most latents are one spot on the 6x6 grid, often the same spot in channels 1 and 2.
  - 11 of 16 latents have 65% or more of their squared mass in 3 cells.
  - One latent puts 94% of its mass on channel 3 at (3,2), the red cell where a 1's stroke lands.
- An L1 penalty on the latents did not help.
  - sklearn's scaling of it removed most latents at once.

### Forcing comparable weights

- Goal: a hit is a few latents of comparable weight, plus an error term.
- Cheap preview: prune each nine's NMF codes, then refit.
  - Drop weights under 1/band of the largest, refit the rest by NNLS, and repeat until all are within the band.
  - Whatever the dropped latents carried goes into the error.

| Band | Latents per nine (median) | Error median | Nines within 10% error |
|---|---|---|---|
| none (NMF) | 11 | 2.2% | 99% |
| 2x | 3 | 14.5% | 35% |
| 3x | 5 | 4.8% | 87% |
| 4x | 6 | 3.5% | 97% |
| 10x | 8 | 2.3% | 99% |

- Error is the share of each nine's squared norm.
- The small NMF weights were not noise: at 2x the error has clear structure.
- A full solver that refits the latents under a 2x band drifted to 1 latent per nine, which is clustering again.

## PCA, then HDBSCAN

- PCA on all 221 hits' L2-normalised Hadamard products.
- 13 components, 83% of the variance, kept at the elbow of the per-component variance curve.
- HDBSCAN on those components, with the report's settings.
- Report: `reports/mnist_cnn_small_all/linear_pca/neuron_9`, with the scree plot as `pca_scree.png`.

| Cluster | Hits | Digits | Median activation | Mean cosine |
|---|---|---|---|---|
| 0 | 6 | 6 ones | 0.07 | 0.922 |
| 1 | 8 | 8 sevens | 1.45 | 0.900 |
| 2 | 5 | 4 nines, 1 four | 4.67 | 0.920 |
| 3 | 20 | 20 sevens | 1.62 | 0.909 |
| 4 | 13 | 10 nines, 2 fours, 1 seven | 4.71 | 0.913 |
| 5 | 10 | 9 nines, 1 seven | 3.83 | 0.941 |

- The 4 cosine clusters carry over, and two new single-digit clusters appear: six 1s and eight 7s.
- Noise drops from 178 to 159, almost all from 1s and 7s.
- 39 of the 42 fours are still noise.

## Weighted Jaccard

- Each hit becomes its share of contribution per dim.
  - Positive and negative parts sit side by side, so +a never matches -a.
  - The shares sum to 1, which makes this an L1 normalisation.
- Distance: 1 - sum(min) / sum(max), also called the Ruzicka distance.
  - It is plain Jaccard with each dim partly in a set, so a dim counts in proportion to its size.
  - It has no cutoff, no set-size bias, and it is a metric.

### On contribution shares

- Report: `reports/mnist_cnn_small_all/linear_weighted_jaccard/neuron_9`.
- Any two hits have a median similarity of 0.453, and the top 10% reach 0.596.

| Cluster | Hits | Digits | Median activation | Mean similarity |
|---|---|---|---|---|
| 0 | 5 | 5 ones | 0.10 | 0.706 |
| 1 | 23 | 20 nines, 2 fours, 1 seven | 4.70 | 0.659 |
| 2 | 6 | 6 sevens | 1.59 | 0.714 |
| 3 | 5 | 5 sevens | 3.06 | 0.761 |

- Every cluster is above the 90th percentile of all pairs, so it is tight by its own measure.
- It mostly agrees with cosine and PCA.
  - It merges two of the cosine nine clusters, and splits the 7s by activation.
- Noise is 182.

### On raw values

- No normalisation at all, so a hit's size counts.
  - A hit's L1 size ranges from 9.3 to 31.7, a 3.4x spread.
- Only 2 clusters form, and 194 hits are noise.
  - 17 nines and 2 fours, at a median activation of 4.76.
  - 8 sevens, at a median activation of 1.78.
- The 1s cluster is gone, likely because size splits small groups below 5 hits.

## Noise across methods

| Clustering | Clusters | Noise of 221 |
|---|---|---|
| Euclidean on L2-normalised vectors (baseline) | 4 | 178 |
| Euclidean on 13 PCA components | 6 | 159 |
| Shared-dimension score | 22 | 45 |
| Weighted Jaccard on shares | 4 | 182 |
| Weighted Jaccard on raw values | 2 | 194 |

- The shared-dimension score clusters the most hits, but 6 of its clusters are loose by its own measure.
- Every other method leaves most 9s, 7s and 4s as noise.
- Between two nines with close activations, the cosine on the top PCA components has a median of 0.08.
  - So within a digit, activation level does not predict the pattern.

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

### Decompositions

- Dictionary learning: sklearn `DictionaryLearning`, coordinate descent, non-negative codes.
- NMF: sklearn `NMF` with `nndsvda` start on the absolute values.
  - Each latent is rescaled to unit norm, with its codes scaled up to match.
  - Each latent is multiplied by the signs of logit 9's weights to show it in colour.
- Band pruning, per nine:
  - Drop every weight under 1/band of the largest.
  - Refit the kept weights by non-negative least squares.
  - Repeat until no refit weight falls under the band.
  - Error is the absolute Hadamard product minus the reconstruction.

### PCA clustering

- PCA with all components on the 221 hits.
- The elbow is the component farthest from the straight line between the first and last points of the per-component variance curve.
  - It is `elbow_index` from the k-means grouping strategy.
- HDBSCAN on the kept components only, swapped into the normal pipeline, so DeepLift and rendering are the pipeline's own.

### Weighted Jaccard

- For each hit, concatenate its positive part and the magnitudes of its negative part, giving 288 values.
- For the share version, divide by their sum.
- For two hits, sum the elementwise minimum, sum the elementwise maximum, and take 1 minus their ratio.
- HDBSCAN takes the distance matrix through sklearn, with `min_samples` raised by 1.
  - sklearn counts a point as its own neighbour and the `hdbscan` package does not.

## Files

- Pages beside the baseline report, in `reports/mnist_cnn_small_all/linear/neuron_9`:
  - `rare_classes_hadamard.html` and `rare_classes_spatial.html`: digits with fewer than 20 hits.
  - `digit_4_hadamard.html`: the 42 fours, ordered by similarity.
  - `digit_9_hadamard.html`: the 99 nines sorted by activation, with 90% mass cells boxed and a toggle.
  - `digit_9_dictionary.html`, `digit_9_nmf.html`, `digit_9_pruned_2x.html`, `digit_9_pruned_3x.html`: the decompositions.
  - `overlap_score_hist.png`: the score histogram.
  - `digit_9_sorted_values.png` and `one_nine_sorted_values.png`: sorted values.
- Other reports for logit 9, in `reports/mnist_cnn_small_all`:
  - `linear_mass_overlap/neuron_9`: the shared-dimension score.
  - `linear_pca/neuron_9`: PCA, then HDBSCAN.
  - `linear_weighted_jaccard/neuron_9`: weighted Jaccard on shares.

## TODO

- Review every Hadamard product hit of logit 9 by eye.
- Try more distance measures, and compare them on the same hits.
- Can logit 9's output be discretised?
  - The question: can we say "this activation range comes from these clusters"?
  - What the data so far says:
    - The 1s cluster sits near 0.1, the 7s clusters at about 1.4 to 3.7, and the nine clusters at about 2.7 to 5.6.
    - So the clusters do occupy different ranges, but the ranges overlap.
    - Within one digit, activation level does not predict the pattern.
    - Most hits are noise, so a range would describe only the clustered few.
