# Port plan: vision interpretability toolkit -> language models

Status: proposed, nothing implemented yet.

This codebase was copied from a repo that analysed a RadDino/DINOv2 encoder on DICOM chest
X-rays. The goal is to retarget it at language models, and to reshape it into a library that
handles the routine parts of a neuron study while staying out of the way.

## The pipeline being generalised

1. Stream a corpus through the model, capturing one MLP down-projection layer.
2. Pick a per-neuron activation threshold by elbow detection.
3. Filter to the positions that exceed it - the "hits".
4. For each hit, take the layer's *input* and Hadamard-multiply it by the neuron's weight row,
   then L2-normalise.
5. Cluster those vectors with HDBSCAN.
6. Track every vector back to the sample and position it came from - this is what makes
   reports possible.

Steps 2 and 4 are already modality-agnostic and carry over untouched.

## The constraint that shapes the architecture

`activations.py` currently `torch.cat`s every input and every output for the whole corpus.

- That is survivable on DINOv2: 257 tokens, 768 output dims, 3072 intermediate dims.
- It is not survivable on an 8B language model. 2000 chunks x 512 tokens x 14336 intermediate
  dims x 4 bytes is roughly 58 GB for the inputs alone.

The shapes are asymmetric in a way that solves this:

- **Outputs** are what the elbow needs, and only for the neurons under study. Keeping
  `outputs[..., neurons]` gives `(n_samples, seq, k)`, which is single-digit MB for a handful
  of neurons.
- **Inputs** are what the Hadamard product needs, and only at hit positions - a tiny fraction
  of tokens by construction.

So the pipeline is two passes. This is the honest shape of the problem, not a compromise.

| | pass 1 (sweep) | pass 2 (gather) |
|---|---|---|
| runs over | whole corpus, streamed | only samples that had hits |
| captures | `output[..., neurons]` | `input` at hit positions |
| produces | threshold + `HitSet` | `(n_hits, d_intermediate)` |

Pass 1 is cacheable to disk, so iterating on threshold and clustering never re-runs the model.

### Consequence: neurons are named up front

- Pass 1 is parameterised by which neurons you care about.
- Keeping *all* neurons' outputs would cost ~17 GB on an 8B model, so it is not the default.
- Studying a new neuron later means re-running pass 1, which the disk cache makes cheap enough.

## Layer addressing

The current `fc2_getter` hardcodes `model.encoder.layer[idx].mlp.fc2`. A dotted path string
replaces it, and no per-architecture registry is needed.

Findings, verified against the installed libraries:

- `Envoy` is not an `nn.Module` (it is `class Envoy(Batchable)`).
- `envoy.get_submodule(path)` therefore falls through `Envoy.__getattr__` and returns the raw
  `Linear`, **not** an Envoy. It does not raise. Tracing then fails later at `.input`/`.output`
  with an unrelated-looking `AttributeError`.
- `nnsight.util.fetch_attr(obj, "a.b.c")` is a plain `getattr` loop. Because `_add_envoy`
  registers each child Envoy as a real instance attribute, every hop returns a child *Envoy*.
- Numeric atoms work, since `ModuleList` children are named `"0"`, `"1"`, and so on.
- The same string is accepted by `nn.Module.get_submodule` on the raw model.

So one literal such as `"model.layers.14.mlp.down_proj"` addresses both the tracer and the raw
model, preserving the dual-use contract `layers.py` documents today.

### What the path string does not solve

- **lxt patch maps key on classes, not paths.** `monkey_patch` needs `LlamaMLP`,
  `LlamaRMSNorm` and the `modeling_llama` module object. This stays a small lookup on
  `model.config.model_type`.
- **Weight orientation.** `hadamard.py` does `weight[neuron_idx]`, which assumes `nn.Linear`'s
  `(out, in)` layout. GPT-2 uses `Conv1D`, whose weight is `(in, out)` - verified as
  `(3072, 768)` versus `nn.Linear`'s `(768, 3072)`. On GPT-2 that indexes an input row instead
  of an output row, which may broadcast into silently wrong Hadamard products rather than
  raising. Needs an explicit orientation check at the call site.

## AttnLRP

`lxt.efficient` ships patch maps for llama, qwen2, qwen3, gemma3, gpt2 and bert, so the
hand-rolled DINOv2 map in `attribution.py` is no longer needed.

Those shipped maps are nonetheless broken on the pinned `transformers` 5.17:

- `lxt.efficient.patches.patch_attention` ends with
  `module.ALL_ATTENTION_FUNCTIONS = NEW_ATTENTION_FUNCTIONS`.
- In 5.17 that attribute is an `AttentionInterface` object, not a dict.

This is the same breakage `patch_dinov2_attention` already works around. That function is
therefore not a DINOv2 quirk but the general 5.x fix: patch `eager_attention_forward` only,
and force `attn_implementation="eager"`. It generalises to llama/qwen for free.

## Module layout

```
in_tfm/
  addressing.py   fetch_attr-based path resolution + weight-orientation handling
  corpus.py       Sample / Corpus protocol, chunking, batching, masks
  sweep.py        pass 1 -> NeuronSweep (activations + provenance)
  threshold.py    unchanged
  hits.py         HitSet: the provenance type
  hadamard.py     pass 2 + the existing math, unchanged
  cluster.py      HDBSCAN + cluster stats (lifted from neuron_report.py)
  attribution.py  AttnLRP, arch patch map
  render.py       modality-specific display (Renderer protocol)
  report.py       markdown/HTML emission
  study.py        the one-liner that chains it all
```

## The types that carry provenance

```
HitSet          sample_ids[n], positions[n]
NeuronSweep     activations, HitSet factory, corpus handle, layer address
ClusteredHits   HitSet + hdmds[n, d] + labels[n]
```

- `HitSet` replaces the parallel `batch_idx` / `token_idx` arrays currently hand-threaded
  through `NeuronClusterHits`'s 14 fields. One object means they cannot desync.
- `sample_ids` indexes the corpus rather than a `dcm_paths` list, so re-fetching a sample for
  attribution becomes a corpus lookup.
- Arrays stay as arrays inside these types, so vectorised filtering and masking still work.

## The two seams that keep it generic

### `Corpus`

Everything the pipeline needs from a modality:

- `batches(bs)` yields `(sample_ids, model_inputs, valid_mask)` for streaming pass 1.
- `gather(sample_ids)` returns model inputs for specific samples. This is what makes pass 2
  possible.

`valid_mask` is new and non-optional for text. Without it, padding positions pollute the elbow
computation and surface as spurious hits.

### `Renderer`

Turns `(sample, position, attribution)` into something displayable.

- For text: a context window around the hit token, background-shaded by relevance.
- `colormaps.py` survives almost intact - the same diverging red/black/green map turns
  relevance into a hex colour, emitted as an HTML `<span>` rather than a pixel.
- No matplotlib, no PIL, no image files on this path.

## API shape

Convenience on top:

```python
study = NeuronStudy.run(
    model, corpus, layer="model.layers.14.mlp.down_proj", neuron=2045
)
study.write_report("out/")
```

Pure functions underneath, each independently callable:

```python
sweep  = sweep_neurons(model, corpus, layer, neurons=[2045])
thresh = find_threshold(sweep, 2045)
hits   = sweep.above(2045, thresh)
feats  = hadamard_features(model, corpus, hits, layer, 2045)
labels = cluster(feats)
```

`NeuronStudy.run` is exactly that sequence with defaults. Nothing is hidden behind it.

## What survives, changes, and is new

**Untouched**

- `threshold.py` in full.
- `hadamard.py`'s math. On gated SwiGLU MLPs `down_proj.input` is
  `act_fn(gate_proj(x)) * up_proj(x)`, the exact semantic analog of `fc2.input`.
- `compat.py`.
- The cluster statistics helpers, with `unique_images` renamed to `unique_samples`.
- `save_elbow_plot` and the `mean.pt` cluster prototype logic.

**Simplified**

- `layers.py` becomes `addressing.py`; the per-architecture registry disappears.
- `attribution.py` keeps the `eager_attention_forward` workaround and drops the hand-rolled
  DINOv2 patch map.

**Replaced**

- `dicom.py` becomes `corpus.py`.
- `viz.py`'s image-compositing half becomes `render.py`'s HTML spans.
- `neuron_report.py` splits into `cluster.py`, `report.py` and `study.py`.

**New, with no vision analog**

- Chunking long documents into windows.
- Attention masks threaded through the whole pipeline.
- Disk caching of pass 1.
- A per-cluster "top hit tokens" summary. This is the standard way neuron dashboards summarise
  a feature, and reads far better at a glance than an image grid did.

**Also outstanding**

- `__init__.py` is still the uv template printing "Hello from in-tfm!", and
  `[project.scripts]` points at it.
- `transformers`, `pillow` and `tqdm` are imported directly but not declared in
  `pyproject.toml`; they currently resolve only transitively. `transformers` in particular is
  pinned tightly enough that `compat.py` exists to paper over a version break.
- `captum` and `pydicom` drop out of the dependency list if IG and DICOM support go.

## Build order

1. `addressing.py`, `corpus.py`, `sweep.py` - get real activations out of a real LM. Proves the
   spine.
2. `hits.py`, wire up the existing `threshold.py` and `hadamard.py`, add `cluster.py` -
   end-to-end to labels, no rendering.
3. Port `attribution.py`. Verify AttnLRP against the `AttentionInterface` breakage before
   building anything on top of it.
4. `render.py` and `report.py`.
5. `study.py` last, once the stage signatures have settled.

Steps 1 and 2 carry the design risk and need neither attribution nor rendering to validate.

## Open questions

- **Keep Integrated Gradients?** `neuron_ig` differentiates with respect to `pixel_values`.
  Token ids are not differentiable, so IG would have to move to the embedding layer
  (`LayerIntegratedGradients`, pad or BOS baseline). AttnLRP is one backward pass and is
  already the default `attr_fn` at every call site.
- **Tolerate `Conv1D` architectures like GPT-2?** Targeting Llama/Qwen-style models only keeps
  `weight[neuron_idx]` correct as written. Supporting GPT-2 means one deliberate normalisation
  in `hadamard.py`.
- **Corpus source format** - HF `datasets`, or local jsonl/txt, or both behind the `Corpus`
  protocol.

Neither of the first two blocks steps 1 and 2.
