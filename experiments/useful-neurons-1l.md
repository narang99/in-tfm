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
