# Induction heads of the attn-only 2L model

Run on 2026-10-07, on CPU, in a notebook.

## Question

- Is `in_tfm.models.attn_only_2l` the model the practice notebooks use?
- Which of its heads form the induction circuit?

## Findings

- Loss on ordinary English sits well below the uniform `log(50278) = 10.83`.
- Loss on the second copy of a repeated random sequence collapses, which is induction.
- Heads, scored on the repeated sequence:
  - **1.10** and **1.4** are induction heads, at about `0.87` and `0.65`.
    - Every other head scores under `0.04`.
  - **0.7** is a previous-token head, at about `0.997`.
- So it is the textbook circuit.
  - A previous-token head in layer 0 feeds induction heads in layer 1.
- Head `1.10` owns `q_proj` neurons `640` to `703` of layer 1.
  - A neuron index is `head * head_dim + dim`.

## Reproduce

- Removed from `main`. The last commit on `main` with it is [`0c79252`](https://github.com/narang99/in-tfm/tree/0c792524ada4568bd24eac7bfea2795820e6a864).
  - Notebook: [`notebooks/attn_only_2l_heads.ipynb`](https://github.com/narang99/in-tfm/blob/0c792524ada4568bd24eac7bfea2795820e6a864/notebooks/attn_only_2l_heads.ipynb)
  - It used `attention_patterns` from [`src/in_tfm/models/attn_only_2l/model.py`](https://github.com/narang99/in-tfm/blob/0c792524ada4568bd24eac7bfea2795820e6a864/src/in_tfm/models/attn_only_2l/model.py), removed with it.
- `git checkout 0c792524a`, `uv sync --extra viz`, then run the notebook.
