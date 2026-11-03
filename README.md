# in-tfm

## What this is

- A library for studying individual neurons in a neural network.
  - Language models: `google/gemma-3-270m`, and a 2-layer attention-only model.
  - Vision: InceptionV1 (lucent's port of the lucid model).
  - A two-conv MNIST network, small enough for quick end-to-end tests.
- For one neuron, it answers what kind of input makes the neuron fire, and whether that generalises.
  - It clusters many high-activation hits, so a finding is a claim about a group of inputs.
  - It is not a single anecdote of "what activates it once".

## The pipeline

- Stream a corpus through the model, capturing one layer.
  - The layer is a dotted submodule path, for example `layers.11.mlp.down_proj`.
- Pick the hits, the positions to cluster.
  - `elbow`: everything above the elbow of the sorted activations.
  - `bucketed`: equal-width activation buckets, with an equal share of hits from each.
  - `polarity` picks the positive tail, the negative tail, both apart, or both together (`signed`).
- For each hit, take the layer's *input* and multiply it element-wise by the neuron's weight row.
  - This is an exact decomposition of the activation: `sum(input * weight[neuron])`.
  - Clustering then groups hits by *which input dimensions drove the activation*.
  - It does not group them by how strongly the neuron fired.
- Cluster those vectors, with the strategy picked by `clustering.strategy.method`.
  - `hdbscan`, the default: dense regions split by sparse gaps, the rest is noise.
  - `density_peaks`: each hit links to its nearest denser hit, and every hit lands in a root's tree.
    - Each hit gets a probability per tree, so hits split between trees are shown apart.
- Track every vector back to its sample and position.
  - This is what makes a readable report possible.

## Two-pass capture

- A whole corpus's activations don't fit in memory for a real model.
- So capture runs in two passes, in `in_tfm.neuron_capture`.
  - **scan**: stream the whole corpus, keeping only the studied neurons' output columns.
    - That is enough to pick the hits exactly.
  - **gather**: re-run only the samples that had hits, keeping the layer *input* at just those positions.
- A consistency check re-derives each hit's activation in the gather pass and compares it with the scan.
  - So a bug in the position bookkeeping surfaces immediately.

## Reports

- One HTML report per neuron, `<out_dir>/neuron_<idx>/index.html`.
  - Plotly sections: the activation curve, each cluster's activation range, and for density peaks the forest.
  - Per cluster: the firing tokens (language) or image crops (vision) of a few sampled hits.
  - Attribution shading: AttnLRP for language models, DeepLift for the vision models.
  - Each cluster's mean Hadamard vector, `mean.pt`, for comparing clusters across runs.
- Density peaks adds two more pages.
  - `mixed_hits.html`: hits whose probability is split between clusters.
  - `borders/`: one page per border between clusters, with a few of its hits.

## Running it

- `scripts/run_neuron_report.py` writes neuron reports for any onboarded model.
  - Settings live in `src/in_tfm/run_config.py`.
  - `--config configs/<file>.yaml` sets defaults, and flags override them.
  - Nested sections take dotted flags, e.g. `--clustering.strategy.min-cluster-size 20`.
  - Repeat a list flag per value, e.g. `--neurons 5 --neurons 7`.
- InceptionV1 needs the `inception` extra and an image pool.
  - `uv run --extra inception python scripts/prepare_inception_images.py --out data/inception_pool`
- Larger runs go on a Colab GPU. See the `colab-experiments` skill.

## Choosing a layer

- `layer_name` is a path into the adapter's `get_model()`. `named_modules()` lists them.
- `q_proj` and `k_proj` are hooked *raw*, before `q_norm` / `k_norm` and rope.
  - The raw output is linear in its input, so `input * weight[neuron]` is an exact decomposition.
  - Rope has not mixed in the position yet, so the coordinate is position-free.
  - A flat neuron index is `head * head_dim + dim`, or `kv_head * head_dim + dim` for `k_proj`.
  - Gemma 3 uses grouped-query attention, so one key neuron is shared by several query heads.
- The attn-only model's `o_proj` is one `Linear` per head, see `PerHeadOutput`.
  - Its `.weight` is still the fused one, so `layers.0.self_attn.o_proj` works like any other layer.

## Onboarding a model

- A model joins the pipeline as an adapter in `src/in_tfm/models/`, satisfying `ModelAdapter`.
  - `get_model()` returns the raw module the layer paths index into, for example `hf_model.model`.
  - `patch_for_attn_lrp()` patches the model for AttnLRP, and is called once, before NNsight wraps the model.
    - NNsight stores each module's `forward` when it wraps, so a model wrapped first silently skips the patches.
    - The adapters raise if the order is wrong.
  - `make_source(samples)` and `make_presenter(source, clustered_label)` pick the modality.
- The Hadamard weight is the layer's own `.weight`, so there is no accessor for it.
- The adapters so far:
  - `gemma3.py`.
  - `attn_only_2l/`, a directory since it carries its own model code in `model.py`.
  - `inception_v1/`.
  - `mnist_cnn/`, with its training in `mnist_training.py`.
- A decoder language model only needs `DecoderTextAdapter` and a patch. See `gemma3.py`.
- `tests/test_model_adapters.py` runs the contract on tiny random models, with no downloads.
  - Add the new adapter to its fixture.

## Repo layout

- `src/in_tfm/`: the library.
  - `models/`: one adapter per model, see "Onboarding a model".
  - `neuron_run.py`: the scan, select, gather, cluster and report sequence.
  - `hit_selection/`: the choice of the positions to cluster.
  - `density_peaks/`, `weighted_jaccard/`: each has a README on its naive and vectorized mirrors.
  - `attribution/`: AttnLRP and DeepLift, one file each.
  - `report_page/`, `report_templates/`: the report's Plotly sections and its jinja templates.
  - `sources/`: where text samples come from. InceptionV1's source lives with its adapter.
  - `tiny_attn_only_training/`: trains our own 2-layer attn-only model on TinyStories.
- `scripts/`:
  - `run_neuron_report.py`: the reports.
  - `prepare_inception_images.py`: the InceptionV1 image pool.
  - `train_tiny_attn_only.py`: trains the tiny attn-only model.
  - `train_mnist.py`: trains the MNIST network.
  - `bench_forest.py`: times the torch forest against a saved one, the regression check for the GPU path.
  - `fake_text_report.py`: renders a text report from fake data, to iterate on the html with no model.
- `configs/`: one YAML per experiment.
- `experiments/`: one markdown log per experiment, with the branch, files and commands to reproduce it.
  - Experiment outputs sit beside the logs, and are git-ignored.
- `TODO.md`: open questions and known rough edges.
