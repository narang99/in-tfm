# Notes: clustering attention head blocks

Status on 2026-10-31. This file records what we think, and why. It is not a plan.

## 1. What the project does

- The project studies single neurons in a neural network.
- For one neuron, it asks: which inputs make this neuron fire?
  - It does not look at one example. It groups many examples, so a finding is about a group of inputs.
- The method:
  - Take the input of the layer at a position.
  - Multiply it by the neuron's weight row, element by element. This is the Hadamard product.
    - The sum of this product is the neuron's value. So the product is an exact split of the value.
  - Cluster these products over many positions.
    - The clusters group positions by which input dimensions made the value, not by how large the value is.
- The current clustering is a density-peak forest.
  - Each point links to its nearest denser neighbour. The links make trees, and each tree is a cluster.
  - Each point gets a membership entropy over the trees.
    - The entropy comes from the distance to each tree's dense peak, measured along the data.
    - A point under 0.5 bits is "confident": one tree clearly owns it.

## 2. The test model

- We trained our own small model, because we need the exact training data.
  - The model is a 2-layer, attention-only transformer, trained on TinyStories.
  - It has 8 heads of 32 dims per layer, and d_model 256.
  - Layer 0 head 2 is a previous-token head. Layer 1 head 0 is an induction head.
- We study layer 0 o_proj.
  - A neuron of o_proj is one dim of the residual stream.
  - The o_proj input holds the 8 head outputs side by side. So the Hadamard row splits into 8 head blocks.
  - We cluster one head block of one neuron at a time.

## 3. The problem: attention mixes the values

- In layer 0, a head's output at position t is a weighted sum of value vectors.
  - z_t = Σ_s A[t, s] · v_s. The attention weights A[t, s] add up to 1.
  - Each v_s depends only on the token at position s.
- So a head block row is a mix of a few fixed token rows.
  - A head that attends to one token gives a row close to that token's row. Such rows make dense peaks.
  - A head that spreads its attention gives a mix of several rows. Each mix is different, so the points spread out.
- The neuron cannot see the separate value vectors.
  - It reads only the mixed result, z_t times its weights.
  - W_O is linear. It cannot keep a special output for a frequent mix.
  - So the neuron can only encode the full combination. Splitting it into value parts is our view, not the neuron's.

## 4. Two ways to think about the mixing

- **View A: the mixing is arbitrary.**
  - Then most combinations are rare.
  - The points are sparse, and no clustering finds a dense core.
  - More data does not help, because the number of possible mixes grows much faster than the data.
- **View B: the network encodes the statistics of its training data.**
  - Then the mixes that matter are the ones that repeat in the training data.
  - Clustering on the full training set should find those repeated mixes as dense regions.
  - What looks like arbitrary mixing in a small sample should become repeatable patterns at full scale.
  - Small networks are known to learn bigrams, trigrams, copying and induction heads. These are repeated patterns.
- **Our working assumption is View B.**
  - We expect heads that encode bigram or trigram statistics to give large clusters on their W_O blocks.
- **Why we went to the full dataset:**
  - At 2,500 positions, only the previous-token head clustered well.
  - So we scaled the forest to millions of points, to test View B.

## 5. What we found so far

- Scale (neuron 0, confident share by number of points):

  | Head | Kind | 256k | 4M | 32M |
  |---|---|---|---|---|
  | h2 | previous token | 97% | 98% | 97.6% |
  | h3 | mixed | 32.5% | 41.4% | 46.0% |
  | h0 | diffuse | 11.0% | 13.6% | 16.7% |
  | h7 | diffuse | 8.1% | 6.8% | not run |

  - The previous-token head clusters almost fully.
  - Mixed heads improve with more data, but slowly.
  - Diffuse heads improve little or not at all, up to 32M points.
  - The number of trees keeps increasing with n. It does not settle.
- Across 10 neurons, at 4M points:
  - The confident share is set by the head, not the neuron. Each head stays within about 6 points across neurons.
  - The confident trees are almost the same in every neuron (ARI 0.82-0.96 on confident points).
  - The unconfident points are split differently by each neuron.
- What the trees mean:
  - In h2, a tree is a previous token, or a group of similar previous tokens.
  - In the mixed heads, a tree is keyed on the token the head attends to, often 1-3 tokens back.
    - Example: tree c1 of neuron 0 head 3 is "was" a few tokens back. "was" holds 33-52% of the row.
  - The tree is not keyed on the token with the largest part of the sum.
    - A token can have a small attention weight but a large value score, and the reverse.
- Confident points come from sharp attention.
  - In head 3, when the top attention weight is over 0.9, most points are confident.
  - Head 3 mixes about 4 tokens at a typical position. Only about 5% of positions put over 0.9 on one token.
- Confident points sit at the extremes of the block value.
  - Points with a large head 3 value, positive or negative, are mostly confident.
  - Points near 0 are mostly unconfident. Their parts cancel.

## 6. The open problem: are the clusters good?

- We now have many clusters: about 1,000 trees per head block at 4M points.
- We do not have a way to say whether these clusters are correct.
- Things that make this hard:
  - The distance scale is coarse.
    - In head 3, the median nearest-neighbour distance is 0.15 (weighted Jaccard on raw rows).
    - The median distance between any two points is 0.85.
    - So the distance does not separate points finely.
  - A tree is a chain of links, not a ball.
    - Two points in one tree can be far apart. Example: two confident c0 points are 0.52 apart.
  - Low entropy says "only one tree is near". It does not say "close to the centre of the tree".
  - The forest uses the raw rows. So the size of a row counts in the distance, not only its shape.
- Ideas to test the clusters:
  - **Do the trees explain the neuron?**
    - Replace each row with its tree's mean, and measure how much of the block value is kept.
    - Compare with simple groupings: by previous token, by most-attended token, by largest contributor, and random groups of the same sizes.
  - **Does it find structure we plant?**
    - Layer 0 lets us build rows from known mixes. Run the forest on them and measure what it recovers.
  - **Are the trees real or noise splits?**
    - Compare each tree's peak density with the density where it joins its neighbour.
    - Check that trees come back with half the data or another seed.
  - **Is the distance right for this?**
    - Compare weighted Jaccard on raw rows, on normalised rows, cosine and euclidean, with the tests above.

## 7. Tools built for reading the clusters

- Value split: each token's part of the block value, A[t, s] · (v_s · w). It is exact.
  - `value_split` in `src/in_tfm/models/attn_only_2l/model.py`.
- Shape share: each token's Shapley share of the row's shape.
  - A set of tokens scores the weighted Jaccard similarity of its partial row to the full row. The shares add up to 1.
  - `src/in_tfm/attribution/shape_shapley.py`.
- The cluster report can colour tokens by either one.
  - `scripts/head_block_report.py --colour-by sum` or `--colour-by shape`.
  - `--example-buckets` shows only confident examples, a share from each entropy bucket.

## 8. Next steps we discussed

- Points between 2, 3, 4 and 5 trees, with a hover card for each contributing tree.
  - This needs each point's probability per tree. The runs saved only the entropy and the top tree.
  - So the entropy step must save the top trees per point, and one pair must be rerun on Colab.
- Save the forest itself (density, neighbours, roots, peaks), so a rerun is not needed for each new question.
- Run a forest on the raw head output, without the neuron's weights, to separate the head from the neuron.
