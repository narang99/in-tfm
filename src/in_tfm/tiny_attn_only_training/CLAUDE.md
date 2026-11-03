# Tiny attn-only training

## Induction heads do not form from this code alone

- This package trains a plain next-token model on TinyStories.
  - On TinyStories alone, a 2-layer attn-only model never formed induction heads in 500M tokens.
  - Names are what repeats in TinyStories, and a single head that attends to earlier names can already copy them.
  - So the corpus never pays for the two-layer induction circuit.
- This branch, `feat/induction-heads`, holds the code that made induction heads form. It is off `feat/tiny-attn-only`.
  - It mixes in training windows that repeat a span, either of story text or of random tokens.
    - The second copy of a span is predictable only by induction.
  - It adds an induction check at each eval.
    - It compares the loss on the first and second copy of a repeated random sequence.
    - It also logs the best previous-token attention in layer 0 and the best induction attention in layer 1.
  - The `tiny_attn_only_2l_v4` checkpoint was trained with that branch.
- It is kept out of the main code on purpose, and is not meant to merge.

## The init is part of the same story

- `init_std` defaults to 0.1 with `lr` 3e-3, not GPT-2's 0.02.
  - Without normalization the residual stream is as small as the init, so attention scores start near 0.
  - At 0.02, neither TinyStories nor repeated random tokens formed a previous-token head.
- This stays in the main code, since it is needed to reproduce the trained checkpoints.
