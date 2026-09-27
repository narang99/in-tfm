# Current stack

Bottom targets `main`, each layer above targets the branch below it. Merge bottom-up.

| Layer | Branch | PR | What it adds |
| --- | --- | --- | --- |
| 2 (top) | `feat/cluster-inference-threshold` | #21 | Per-cluster cosine-similarity inference threshold, pulled from #19 |
| 1 (bottom) | `feat/attn-only-2l-shim` | #20 | The attn-only 2L shortformer model as a pure-torch shim, plus the circuitsvis notebook |

- Layer 1 is what makes layer 2 testable. The whole report pipeline plus the inference pass runs on
  CPU in about 12 seconds on that model, so the inference flow can be iterated on without Colab.
- Layer 2 supersedes PR #19, which is still open against `main`. Close it once this stack lands.

Useful commands:

```bash
gh stack view --short   # the chain and its PR links
gh stack sync           # cascading rebase after the bottom moves
gh stack merge          # merge the whole stack bottom-up
```

# Todos

## 1. The elbow finder breaks on mixed-sign values

Top priority.

- `find_elbow_index_in_sorted_data` is kneedle-style: it takes the max perpendicular distance from
  the chord joining the first and last points of the sorted curve.
- The activation path deliberately filters to `values > 0` first, in
  `threshold.positive_sorted_activations`. The cosine path in
  `inference.fit_and_patch_inference_thresholds` does not filter at all.
- A negative left tail moves the chord endpoint, which moves the knee. The two paths should agree
  on whether to filter, and the chord method needs a defined answer for sign-mixed input.
- This is currently **latent, not firing**. On layer 1 neuron 90 of the attn-only model the
  similarities run 0.0287 to 0.9489, so nothing is negative and both variants return 0.6149. It
  will bite wherever cosine actually goes negative, which nothing prevents.

## 2. The fitted inference threshold admits far too much

Discovered while running the flow end to end. Arguably the thing that blocks using it at all, so it
should probably be fixed alongside todo 1.

Measured on layer 1 neuron 90, cluster 4, over 8192 valid wikitext tokens:

| Measure | Value |
| --- | --- |
| Fitted threshold | 0.6149 |
| Tokens at or above it | 919 of 8192, 11.2% |
| Cluster's actual size | 39 hits, 0.48% |

- The cutoff is about 24 times more permissive than the cluster it is meant to represent.
- The curve is smooth and convex rather than flat-then-spiking, so the chord method has no sharp
  knee to find. Its p90 is 0.6274, almost exactly the fitted value, which is the tell: the elbow is
  tracking the bulk of the distribution instead of the tail.
- Worth comparing against a quantile cutoff, or fitting on the log of the tail, before keeping
  kneedle here.

## 3. One similarity scan per neuron, not per corpus

- `NeuronCapture.scan_cluster_similarities` takes `dict[neuron_idx, means]` and its docstring says
  scoring several neurons costs a single pass.
- `run_llm_neuron_report.fit_inference_thresholds` calls it with one neuron at a time, from inside
  `report_neuron`, so a 6-neuron two-polarity run makes 12 full corpus passes instead of 1.
- Hoist the scan out of `report_neuron` and fit every neuron's clusters from one pass.

## 4. The vision path never gets an inference threshold

- `scripts/run_neuron_report.py` was reverted to its pre-`all_hdmd` form, so DICOM reports write
  `mean.pt` and `activation_threshold` but never an inference threshold.
- `ClusterMatcher.from_report` therefore raises on any vision report.
- Wiring the same third pass into the vision script is the fix.

## 5. Smaller things

- `inference.load_cluster_means` and `ClusterMatcher.from_report` load `mean.pt` with
  `torch.load(..., weights_only=False)`. These are plain tensors the pipeline wrote itself, so
  `weights_only=True` works and is strictly safer.
- `fit_and_patch_inference_thresholds` depends on `cluster_ids` being in the same order the scan
  received the means. That coupling is documented in both places but not enforced, and a silent
  mismatch would assign each cluster another cluster's threshold.
- The over-common token investigation in `TODO.md` applies to the attn-only model too. Its cheap
  full scans plus the exact token and position split that shortformer allows make it a good place
  to settle that question.
