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
