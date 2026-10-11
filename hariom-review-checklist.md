# Hariom's review checklist

Things to review or understand for the STL-10 mini-Inception work, in the order I would do them.

## 1. Understand DeepLIFT and our code (do this first)

- Read `src/in_tfm/deeplift.py` and `tests/test_deeplift.py` before anything else.
  - Every other attribution file depends on it.
- The idea to hold on to:
  - Each ReLU and MaxPool2d output is replaced by `y0 + m * (x - x0)`.
  - `y0` and `x0` come from a baseline pass, and `m` is a detached multiplier.
  - So the network becomes affine in the input, and attributions sum exactly to the activation change.
- ReLU uses the Rescale rule: the multiplier is `(y - y0) / (x - x0)`.
  - Check what happens when `x - x0` is near zero.
- Max pool splits the change between the input winner and the baseline winner.
  - The split is gathered by winner index so overlapping windows add up.
  - The Inception blocks use 3x3 stride 1 pools, which overlap, so this is the part to read slowly.
- Why not captum's DeepLift:
  - Its `max_unpool` rule drops contributions on overlapping windows, so completeness fails.
  - A scatter-add fix was still incomplete on a pool feeding a branch with another pool.
  - The exact mechanism was never found, see the memory note `captum-deeplift-maxpool-incomplete`.
- Questions to answer for yourself:
  - Why is the completeness test (`test_deeplift_attributions_sum_to_the_activation_change_from_the_baseline`) a good bug detector?
  - What does a zeros baseline mean here? It is the dataset-mean image in normalised space.
  - Which layers does `deeplift.py` not handle? BatchNorm is folded or in eval mode, check which.
- Then read `compute_deeplift_relevance` in `src/in_tfm/attribution.py`.

## 2. Attribution comparison script (#55)

- `scripts/compare_attribution.py` and `src/in_tfm/attribution_comparison.py`.
- Check that the comparison is fair:
  - Both methods use a zeros baseline and the same scalar: one neuron at each image's own peak position.
  - IG time includes `n_steps` forward and backward passes, DeepLift time includes its two passes.
  - Timings are synchronised on CUDA (`sync`), but not on MPS, so MPS timings are rough.
  - A warm-up call runs first, so compile and allocator cost is excluded.
- Check the metrics:
  - `pearson` is per image, then averaged.
  - `top_k_overlap` ranks pixels by absolute relevance summed over channels.
  - `completeness_gap` is relative to the activation change, with a clamp for tiny changes.
- `merge` weights quality metrics by batch size and sums times, check that this is what you want.
- Run it yourself once:
  - `python scripts/compare_attribution.py --n-images 64 --batch-size 16 --ig-steps '[8,32,128,512]'`
  - Results on Colab are in `docs/stl_deeplift_vs_integrated_gradients.md`.
- Tests: `tests/test_attribution_comparison.py`.
  - The "converged IG agrees with DeepLift" test uses a random network that is nearly linear, so it is a weak discriminator.

## 3. Possible change: use captum's `NeuronIntegratedGradients`

- I might replace the hand-written IG wrapper with captum's `NeuronIntegratedGradients`.
- What is hand-written today, in `src/in_tfm/integrated_gradients.py`:
  - `explained_activation` hooks the layer and picks `layer_output[:, neuron].flatten(1)[arange, positions]`.
  - `integrated_gradients` calls `IntegratedGradients` on that, with positions as `additional_forward_args`.
  - `pick_position` selects one position per image.
- What captum offers: `NeuronIntegratedGradients(model, layer).attribute(inputs, neuron_selector, baselines, n_steps, internal_batch_size)`.
  - `neuron_selector` can be a callable on the layer output, which is how a different position per image could be expressed.
  - Verify this against the installed captum version before relying on it.
- If switching:
  - Write a test that the new path matches the current `integrated_gradients` output on the same images and positions.
  - Check `internal_batch_size` still chunks the steps as it does today.
  - Check the hook does not break the DeepLift hooks if both run in one process.
  - Delete `explained_activation` and keep `pick_position` only if something else still uses it.
- Not a reason to touch DeepLift: captum has no neuron-level DeepLift that is complete on overlapping pools.

## 4. Report plumbing and the sampling fix (#56)

- `sampled_hit_indices` in `src/in_tfm/neuron_report.py`.
  - The bug: hit sampling used the global `np.random`, so two runs showed different hits in the same cluster.
  - The fix seeds per cluster with `[SAMPLE_SEED, cluster_id]`.
  - Check that this changes no existing report beyond which hits are shown.
- `attribution_fn` in `src/in_tfm/models/stl_inception/adapter.py` and `StlDataConfig` in `src/in_tfm/run_config.py`.
  - `--stl.attribution` and `--stl.ig-steps` are the only new flags.
  - The default stays DeepLift.
- `tests/test_hit_sampling.py`, `tests/test_stl_adapter.py`, `tests/test_run_config.py`.

## 5. Read the experiment write-up

- `docs/stl_deeplift_vs_integrated_gradients.md`, mainly "Results" and "Conclusion".
- Things to decide for yourself:
  - Is Pearson 0.93 to 0.94 close enough to prefer DeepLift?
  - The residual gap between the methods is not explained.
  - Faithfulness was not tested, only agreement. A deletion or insertion test would be needed.
- The top-edge and letterbox relevance seen in the overlays is unexplained and untested.

## 6. Already merged, only if time allows

- #46 to #53 (model, conv token layout, DeepLift, plumbing, adapter, hub loading, training).
- The open question from earlier: the image-normalization contract in `dicom.py` and `presenters/image.py`.
  - I passed an `ImageNormalization` Protocol and was asked whether the contract is transform and inverse-transform.
  - It was deferred and not answered.
