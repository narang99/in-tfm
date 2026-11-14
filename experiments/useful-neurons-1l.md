# Useful neurons from every logit, 1L attn-only

Built 2026-11-11.

## Question

- Which neurons does one sequence use, across every logit at every position?

## Setup

- Model: `checkpoints/tiny_attn_only_1l_d128` (d 128, 4 heads of 32, no biases, no norm).
- Sequence: the first N tokens of `val_tokens.npy`.
- Every neuron splits into terms that add up to its value.
  - Each neuron keeps the largest terms making up 90% of its positive sum.
  - It does the same for its negative sum, on its own.
  - The neurons behind each kept term are picked, and the walk repeats on them.
- The cut is local.
  - A neuron's picks do not depend on which logit reached it.
  - So each neuron is expanded once and shared by all logits.
- How each kind splits:

| Neuron | Terms | Picks |
|---|---|---|
| logit | `W_U[t] * resid` | final resid dims |
| resid | carried resid, each head's write | resid below, o_proj |
| o_proj | `W_O[d] * z` | head_out |
| head_out | `pattern[src] * v[src]` | v and score, per source |
| v | `W_V[j] * resid` | resid |
| score | `q * k * scale` | q and k, per head dim |
| q, k | `W[j] * resid` and `W[j] * pos` | resid or pos |

## Results

| kind | 64 tokens, 0.9 | 64 tokens, 0.5 |
|---|---|---|
| resid | 94.2% | 76.0% |
| pos | 93.2% | 72.8% |
| o_proj | 82.5% | 38.8% |
| head_out | 94.3% | 75.6% |
| v | 92.8% | 63.7% |
| q | 82.8% | 46.1% |
| k | 93.0% | 63.1% |
| score | 68.2% | 43.9% |

- Shares are picked neurons over all neurons of that kind in the sequence.
- At 0.9 each logit keeps 66 of 128 final resid dims on average. At 0.5 it keeps 19.
- At 0.9 nearly every neuron is picked.
- The walk takes 30 s at 64 tokens on the Mac.

## Command

```
python scripts/find_useful_neurons.py checkpoints/tiny_attn_only_1l_d128 --seq-len 64 --share 0.9
```

## Ablation: zero every unpicked neuron

- One forward pass keeps only the picked neurons and zeroes the rest.
  - A dropped score zeroes its attention weight, with no renormalising.
    - That removes exactly the term the walk judged.
  - Biases and logits are untouched.
- The model has 98,432 neurons at 64 tokens.

| share | neurons kept | mean KL (nats) | max KL | top-1 unchanged | median next-token change |
|---|---|---|---|---|---|
| 0.9 | 85,503 | 0.021 | 0.113 | 93.8% | -0.4% |
| 0.5 | 55,131 | 1.263 | 4.879 | 50.0% | -58.6% |

- KL is KL(original || ablated) at each position, summed over the vocab.
- Not done yet: a control that drops the same number of neurons of each kind at random.

```
python scripts/ablate_unpicked_neurons.py checkpoints/tiny_attn_only_1l_d128 --seq-len 64 --share 0.9 --out ablation.html
```

## TinyStories-1M (GPT-Neo)

- Model: `roneneldan/TinyStories-1M`. 8 layers, d 64, 16 heads of 4, MLP 256, vocab 50,257.
- Text: the same val text as above, re-tokenized with its GPT-2 tokenizer, first 64 tokens.
- LayerNorm is made linear by freezing each token's mean and std.
  - A dim's term through a norm is `W[j, d] * (weight[d] / std) * x[d]`.
  - The ablated pass reuses the original pass's stats.
- Biases and the norms' constants are left out of the sums, as before.

| | 1L d128 | TinyStories-1M |
|---|---|---|
| neurons kept | 87% | about 71% |
| mean KL (nats) | 0.021 | 5.554 |
| top-1 unchanged | 93.8% | 9.4% |
| median next-token change | -0.4% | -97.0% |

- The walk and ablation take 144 s locally, with 1.6 GB peak memory.

### Why it breaks

- The residual mean is small (|mean| / std at most 0.35), so the frozen norm is not the cause.
- Zeroing one kind at a time, with everything else kept:

| kind zeroed | mean KL |
|---|---|
| mlp_down | 6.97 |
| head_writes (o_proj) | 4.49 |
| k | 1.50 |
| q | 0.18 |
| everything else | 0.11 or less |
| resid | 0.003 |

- Unpicked writes split by whether the walk reached the stream dim they write to:

| | parent reached | parent not reached |
|---|---|---|
| mlp_down | 0.20 | 6.95 |
| head_writes | 4.01 | 2.15 |

- The unreached-parent row is a diagnostic artifact, not the cause.
  - An MLP that writes -X onto a dim carrying +X leaves the dim near zero, so the walk never reaches it.
  - That test keeps every resid dim, so zeroing only the eraser brings the +X back.
  - The full ablation zeroes the unreached resid dim too, which removes the +X with it.
  - If the dim is reached, +X and -X have opposite signs, so the 90% rule keeps both.
- The reached-parent row is real: o_proj terms cut on near-cancelling sums.
  - A mid-stream dim sums the carried stream and 16 head writes.
  - The 90% rule bounds what is dropped by 10% of each sign's mass, not by 10% of the value.
  - When large positive and negative terms nearly cancel, the dropped part can exceed the net value.
- No one kind explains the full run. Restoring any single kind still leaves it broken.
  - Restoring head_writes helps most: 5.55 to 3.30.
  - Restoring mlp_hidden gives 4.25, and restoring k gives 4.66.
- The 1L model has 4 heads, not 16, and only one layer for the error to compound through.

```
python scripts/ablate_unpicked_neurons.py checkpoints/tiny_attn_only_1l_d128 --model tinystories-1m --out ablation.html
```

## Random baseline

- Each mask (one per kind per layer) drops the same number of neurons as the walk did, at random.
  - Attention weights are only reshuffled among sources a token can attend to.
- Five seeds per setting, 64 tokens.

| model, share | walk: mean KL | random: mean KL | walk: top-1 unchanged | random: top-1 unchanged |
|---|---|---|---|---|
| 1L d128, 0.9 | 0.021 | 2.65 to 3.32 | 93.8% | 12.5% to 23.4% |
| TinyStories-1M, 0.9 | 5.55 | 34.5 to 64.5 | 9.4% | 0% |
| TinyStories-1M, 0.95 | 2.57 | 40.1 to 53.5 | 29.7% | 0% to 3.1% |

- The walk's picks carry real signal in both models.
- On TinyStories-1M, random drops wreck the model completely, so the walk's KL of 2.6 to 5.6 is far from chance.

## Block mode: keep every writer, trim inside blocks

- A block is everything between the residual-stream readers and a writer.
  - Attention: q, k, v, attention weights and z. MLP: the hidden neurons.
- Every writer (o_proj per head, mlp_down) is kept, with the residual stream and embeddings.
- The walk starts from all writers and stops at the residual stream. Logits play no part.
- Share 0.9, 64 tokens, random baseline as above.

| model | walk: mean KL | random: mean KL | walk: top-1 unchanged | random: top-1 unchanged |
|---|---|---|---|---|
| 1L d128 | 0.012 | 1.89 to 2.55 | 95.3% | 23.4% to 34.4% |
| TinyStories-1M | 4.20 | 47.9 to 61.0 | 20.3% | 0% to 3.1% |

- Kept inside blocks on TinyStories-1M: q 95.5%, k 95.7%, v 88.6%, attention weights 69.0%, z 100%, MLP hidden 90.9%.
- A kept writer still computes its value from trimmed neurons, so its write is off by up to about 10%.
  - Later blocks read those changed writes, so the error still compounds across layers.
  - Block mode removes the cuts on the residual stream itself, not the compounding through blocks.

```
python scripts/ablate_unpicked_neurons.py checkpoints/tiny_attn_only_1l_d128 --model tinystories-1m --mode blocks --out blocks.html
```

## Recomputed LayerNorm stats in the ablated pass

- The walk still reads frozen stats, so its picks are unchanged.
- Only the ablated pass recomputes each token's mean and std after trimming.

| TinyStories-1M, share 0.9 | walk: mean KL | random: mean KL | walk: top-1 unchanged | random: top-1 unchanged |
|---|---|---|---|---|
| logits, frozen norms | 5.55 | 34.5 to 64.5 | 9.4% | 0% |
| logits, recomputed norms | 5.47 | 6.47 to 7.06 | 9.4% | 4.7% to 7.8% |
| blocks, frozen norms | 4.20 | 47.9 to 61.0 | 20.3% | 0% to 3.1% |
| blocks, recomputed norms | 2.27 | 5.63 to 6.32 | 29.7% | 4.7% to 15.6% |

- Frozen norms inflated the random baseline about sevenfold. They barely changed the logit walk.
  - Random drops shrink the stream, and a frozen norm cannot scale it back up.
- With recomputed norms, the logit walk is only slightly better than random.
- Block mode with recomputed norms is the best setting so far, at about 2.5 times better than random.

```
python scripts/ablate_unpicked_neurons.py checkpoints/tiny_attn_only_1l_d128 --model tinystories-1m --mode blocks --recompute-norms --out blocks.html
```

## Writer mode (code not merged)

- At each token, each head's write vector and each MLP's write vector keep their top-share dims, per sign.
- From the kept writer dims, the walk goes back inside the block and stops at the stream.
- Embeddings are untouched and no stream neuron is zeroed. Dropped writer dims are zeroed.
- Code: commit [`7612d88`](https://github.com/narang99/in-tfm/commit/7612d88c60df1fa4a1faa20103e2a3aeae6d63ae), closed PR #190.
- Share 0.9, 64 tokens.

| setting | walk: mean KL | random: mean KL | walk: top-1 unchanged | random: top-1 unchanged |
|---|---|---|---|---|
| 1L d128 | 0.065 | 2.50 to 3.04 | 85.9% | 15.6% to 21.9% |
| TinyStories-1M, frozen norms | 21.48 | 115.3 to 162.6 | 4.7% | 0% |
| TinyStories-1M, recomputed norms | 4.47 | 8.36 to 9.06 | 20.3% | 3.1% to 9.4% |

- Writers kept: 65.0% of head writes on 1L. 66.6% of head writes and 66.8% of MLP writes on TinyStories-1M.
- Inside the blocks, the kept shares are almost the same as in block mode.
  - TinyStories-1M: q 95.5%, k 95.7%, v 88.6%, attention weights 69.0%, z 100%, MLP hidden 90.3%.
  - A neuron inside a block is kept if any kept writer reaches it, and a third fewer writers barely changes that.

```
python scripts/ablate_unpicked_neurons.py checkpoints/tiny_attn_only_1l_d128 --model tinystories-1m --mode writers --recompute-norms --out writers.html
```

## Mean fill (code not merged)

- Dropped neurons took each neuron's mean over 99,968 val tokens instead of 0. Attention weights still became 0.
- Code: commit [`9836885`](https://github.com/narang99/in-tfm/commit/9836885bac64b7ee9aaaec0fe482357f9c61577b), closed PR #191.
- Share 0.9, 64 tokens, old attention split.

| model, mode, norms | zero: KL | mean: KL | mean: random KL |
|---|---|---|---|
| 1L, logits | 0.021 | 0.320 | 1.82 to 2.21 |
| 1L, blocks | 0.012 | 0.099 | 1.17 to 1.54 |
| 1L, writers | 0.065 | 0.290 | 1.87 to 2.08 |
| TS-1M, logits, recomputed | 5.47 | 3.71 | 6.16 to 6.65 |
| TS-1M, blocks, recomputed | 2.27 | 1.46 | 5.01 to 5.52 |
| TS-1M, writers, recomputed | 4.47 | 5.73 | 8.95 to 9.55 |

- Means were worse than 0 on 1L in every mode.
  - The walk drops terms that are small at this token, so the real value there is usually nearer 0 than the corpus mean.

## MLP neurons only (code not merged)

- TinyStories-1M, blocks, share 0.9, recomputed norms, zero fill. Every attention neuron kept.
- Code: commit [`7d63c4a`](https://github.com/narang99/in-tfm/commit/7d63c4ad3424a89782e5387a7c0ebf67a1d0fc14), closed PR #192.
- KL 0.0505 (random 4.14 to 4.83), top-1 unchanged at 89.1%. 90.9% of MLP hidden neurons kept.
- With attention trimmed too, the same setting gave 2.27. Most of the damage came from attention.

## Scores walk on their own

- Every result above used the old attention rule, where head_out kept a source's value and score together.
  - Unpicked scores then had their attention weight zeroed after the softmax.
- The rule now:
  - head_out splits a z dim by `pattern[src] * v[src, j]` and keeps value neurons only.
  - Every score with attention weight above 0 starts its own walk, and splits by `q[j] * k[j]`.
  - No attention weight is zeroed. Scores change only through dropped q and k dims.
- Blocks, share 0.9, zero fill. TinyStories-1M recomputes its norms.

| model | old rule: KL | new rule: KL | new rule: random KL | new rule: top-1 unchanged |
|---|---|---|---|---|
| 1L d128 | 0.012 | 0.0117 | 1.24 to 1.75 | 96.9% |
| TinyStories-1M | 2.27 | 0.488 | 5.76 to 6.21 | 64.1% |

- Kept on TinyStories-1M: q 95.8%, k 96.7%, v 88.6%, MLP hidden 90.9%.

```
python scripts/ablate_unpicked_neurons.py checkpoints/tiny_attn_only_1l_d128 --model tinystories-1m --mode blocks --recompute-norms --out blocks.html
```
