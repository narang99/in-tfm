# Current state

The stack has landed its bottom layer, so only one PR is open now.

| Branch | PR | State | What it adds |
| --- | --- | --- | --- |
| `feat/attn-only-2l-shim` | #20 | **merged 2026-09-28** | The attn-only 2L shortformer model as a pure-torch shim, plus the circuitsvis notebook |
| `feat/cluster-inference-threshold` | #21 | open, targets `main` | Per-cluster cosine-similarity inference threshold, pulled from #19 |

- #20 is what makes #21 testable. The whole report pipeline plus the inference pass runs on CPU in
  about 28 seconds on that model, so the inference flow can be iterated on without Colab.
- #21 supersedes PR #19, which is still open against `main`. Close it when #21 lands.
- The `gh stack` registration was removed when #20 merged, since a one-PR stack means nothing. If
  another dependent layer is started, re-create it with `gh stack link <bottom> <top>`.

# Todos

## 1. The inference threshold is now too strict

The corpus-elbow rule was replaced by a cutoff fit from the cluster's own spread, which fixed the
over-admission (5% to 22% of the corpus, down to 0.06% to 0.70%). Sweeping the cutoff afterwards
showed it overshot: the concept usually survives well below the cluster's worst member.

Measured on layer 1 neuron 90, 32768 wikitext tokens, in
`experiments/cluster-inference-monosemanticity`:

| cluster | member min | edge where the concept breaks | tokens gained |
| --- | --- | --- | --- |
| 2 | 0.803 | 0.55 | 6.9x |
| 1 | 0.735 | 0.50 | 5.0x |
| 8 | 0.849 | 0.75 | 3.4x |
| 0 | 0.605 | 0.65 | 0.8x |

- The edge is cluster-dependent, spanning 0.50 to 0.80, so a single global constant will be wrong
  for some cluster. `MEMBER_RECALL` adapts to the cluster's geometry but is biased high by roughly
  0.1 to 0.25.
- Whatever replaces it has to keep the loose end: at 0.55 cluster 0 admits
  `received`/`released`/`produced`/`aired`/`featured`, which is one concept and five times the
  tokens. At 0.65 it has collapsed to just `received`, which scores better on every purity metric
  while being strictly less useful.

## 2. Monosemanticity needs a real measure, not a type count

- Type-count and purity heuristics both fail. Purity rewards a cluster that has collapsed onto a
  single token, and type counts punish genuine morphological families.
- The standard practice in the SAE literature is an LLM judge: explain the cluster from its top
  examples, then score whether a judge can predict membership on held-out text. Pick the cutoff
  that maximises tokens caught subject to judged precision.
- Sample across the whole similarity range when judging, not just the top. Top examples always look
  monosemantic.

## 3. Detect with a discriminative direction, not the centroid

- The cluster mean is a prototype, which is the wrong object for deciding membership. SAEs keep the
  encoder direction separate from the decoder direction for exactly this reason.
- An LDA axis or a small logistic probe separating cluster members from the corpus should widen the
  monosemantic band, and may matter more than any threshold rule.

## 4. One similarity scan per neuron, not per corpus

- `NeuronCapture.scan_cluster_similarities` takes `dict[neuron_idx, means]` and its docstring says
  scoring several neurons costs a single pass.
- `run_llm_neuron_report.report_corpus_admission` calls it one neuron at a time from inside
  `report_neuron`, so a 6-neuron two-polarity run makes 12 full corpus passes instead of 1.

## 5. The vision path never gets an inference threshold

- `scripts/run_neuron_report.py` still writes `mean.pt` but no threshold, so
  `ClusterMatcher.from_report` raises on any DICOM report.

## 6. Smaller things

- `inference.load_cluster_means` and `ClusterMatcher.from_report` load `mean.pt` with
  `torch.load(..., weights_only=False)`. These are plain tensors the pipeline wrote itself, so
  `weights_only=True` works and is strictly safer.
- `measure_corpus_admission` depends on `cluster_ids` being in the same order the scan received the
  means. Documented in both places but not enforced, and a silent mismatch would report each
  cluster another cluster's number.
- `find_elbow_index_in_sorted_data` is no longer used on cosine values, so the mixed-sign concern is
  now only theoretical. The activation path still filters to `> 0` before calling it.
- The over-common token investigation in `TODO.md` is the same question as todo 1. Several of the
  biggest clusters here are wikitext artifacts, `-` from `@-@` and `=` from heading markup, not
  language.
