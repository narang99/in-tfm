# in-tfm

## What this is

- A library for studying individual neurons in a neural network
  - originally built for a vision encoder (RadDino / DINOv2 on chest X-rays)
  - now retargeted at decoder language models (`google/gemma-3-270m`)
- For one neuron, it answers: what kind of input makes this neuron fire, and does that
  generalise across many examples
  - not just "what activates it once" - it clusters many high-activation hits, so a
    finding is a claim about a group of inputs, not a single anecdote

## The pipeline

- Stream a corpus through the model, capturing one layer
  - vision: an MLP down-projection (`fc2` / `down_proj`)
  - language: `q_proj`, `k_proj`, or `down_proj`
- Pick a per-neuron activation threshold automatically, by elbow detection
  - replaces a hand-picked constant threshold
- Filter to positions above the threshold - the "hits"
- For each hit, take the layer's *input* and Hadamard-multiply it by the neuron's own weight
  row, then L2-normalise
  - this is an exact decomposition of the activation: `sum(input * weight[neuron])`
  - clustering groups hits by *which input dimensions drove the activation*, not by how
    strongly the neuron fired
- Cluster those vectors with HDBSCAN
- Track every vector back to its sample and position
  - this is what makes a human-readable report possible: firing tokens, attribution shading,
    per-cluster hit counts

## Two-pass capture

- A whole corpus's activations don't fit in memory for a real model
- So capture runs in two passes (`in_tfm.neuron_capture`)
  - **scan**: stream the whole corpus, keep only the studied neurons' output columns -
    enough to compute the threshold exactly
  - **gather**: re-run only the samples that had hits, keep the layer *input* at just those
    positions
- A consistency check re-derives each hit's activation in the gather pass and compares it
  against the scan, so a bug in the position bookkeeping surfaces immediately

## Reports

- One HTML report per neuron, showing its largest clusters
  - the firing tokens (language) or image crops (vision) for sampled hits per cluster
  - AttnLRP relevance shading, showing which part of the input the model itself weighted
  - each cluster's mean Hadamard vector (`mean.pt`), for comparing clusters across runs

## Experiments

Ablations that test one component of the pipeline against an alternative, on layer 11 (a
full-attention layer) of `google/gemma-3-270m`'s `q_proj` / `k_proj`, unless noted. Each has
its own README under `experiments/<name>/` with the exact setup, numbers and caveats.

- **`ablate-hadamard-vs-input`**: does the Hadamard weighting change what clusters, or does
  the raw input alone already group hits the same way?
  - layer 0: barely - clusters are near-identical, because the residual there is essentially
    the token embedding
  - layer 11: differs more, but clusters are still mostly single-token
- **`ablate-select-on-q-norm`**: `q_proj`'s raw output is hooked before Gemma's per-head
  `q_norm` and rope, so a large value can partly just mean a large head norm - does picking
  hits by the *post-norm* activation instead change which hits get selected, and does that
  fix the neurons whose raw-selected clusters look degenerate?
  - one neuron (300) went from one 4,313-hit blob to 28 well-formed clusters
  - another (90) lost its cleanest cluster (`of`/`in` after a place noun) once selection
    moved to the norm
  - inconclusive on its own: no ground truth for "the right hits" from cluster inspection
    alone
- **`occlusion-test`**: a functional test to settle the above - zero the neuron's coordinate
  (at the `q_norm` output, the value attention actually reads) at each selection's hits, and
  measure the resulting change in next-token loss
  - both selections' top hits matter about equally
  - the hits *raw* selection finds that *norm* selection misses matter more than the reverse
  - net finding: raw `q_proj` selection is at least as good functionally, and it stays the
    more principled choice for the Hadamard decomposition
- **`hdbscan-leaf`**: HDBSCAN's default cluster-picking rule (`eom`, excess of mass) can
  merge distinct sub-clusters into one large parent - does `cluster_selection_method=leaf`
  fix the degenerate neuron from the selection ablation, without needing to change hit
  selection?
  - yes: the same 4,313-hit blob splits into 23 clusters under `leaf`, and matching them
    against the norm-selection clusters (linear sum assignment on cosine similarity between
    `mean.pt` vectors) shows 12 of the top 20 pairs agree almost exactly
  - conclusion: `leaf` is now the library's default `cluster_selection_method`, and hit
    selection stays on raw `q_proj`

## Onboarding a model

- A model joins the pipeline as an adapter in `src/in_tfm/models/`, satisfying `ModelAdapter`
  - `get_model()` returns the raw module the layer getters index into, for example `hf_model.model`
  - `patch_for_attn_lrp()` patches the model for AttnLRP, and is called once, before NNsight wraps the model
    - NNsight stores each module's `forward` when it wraps, so a model wrapped first silently skips the patches
    - the adapters raise if the order is wrong
  - `make_source(samples)` and `make_presenter(source, clustered_label)` pick the modality
- Layer getters are not part of the adapter
  - they live in `layers.py`, and the script picks one per experiment
  - the Hadamard weight is `layer_getter(model).weight`, so there is no accessor for it
- The adapters so far
  - `gemma3.py`, and `rad_dino.py`
  - `attn_only_2l/`, a directory since it carries its own model code in `model.py`
- A decoder language model only needs `DecoderTextAdapter` and a patch
  - see `gemma3.py`
- `tests/test_model_adapters.py` runs the contract on tiny random models, with no downloads
  - add the new adapter to its fixture

## Repo layout

- `src/in_tfm/`: the library
  - `models/`: one adapter per model, see "Onboarding a model"
- `scripts/run_neuron_report.py`: vision (DICOM) reports
- `scripts/run_llm_neuron_report.py`: language model reports
- `experiments/`: ablation results (git-ignored - see each README for how to regenerate)
- `PLAN.md`, `COLAB_PLAN.md`: the vision-to-language port plan and the Colab setup notes
- `TODO.md`: open questions and known rough edges
