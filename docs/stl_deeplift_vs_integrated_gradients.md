# STL mini-Inception: DeepLift against Integrated Gradients

Run on 2026-10-11 on a Colab T4. Code is the tip of `feat/ig-reports` (PR #56), commit `e01f3ff`.

## Question

- Is our `in_tfm.deeplift` close to captum's Integrated Gradients (IG), and how much faster is it?
- Do neuron reports built with either method show the same thing?

## What was compared

- Both explain one conv neuron at one output position as a map over the input pixels.
  - The layer is `block_b.branch_3x3.1.0`, a 24x24 grid of positions.
  - Both use a zeros baseline, which is the dataset-mean image in normalised space.
- DeepLift is exact after two passes, and IG needs one forward and backward pass per step.
- Two experiments:
  - `scripts/compare_attribution.py` measures time and agreement at each IG step count.
  - `scripts/run_neuron_report.py` builds full reports once per method, over identical hits.

## Environment

- Colab runtime was a T4, Python 3.13.15, torch 2.11.0+cu130, numpy 2.1.3, scikit-learn 1.6.1, transformers 5.18.0.
- cuml 26.6.0 was preinstalled (`cuml-cu13`), so clustering ran on the GPU.
  - The log line `[cluster] 3000 points x 576 dims via cuml (gpu)` confirms it.
- Missing on Colab, so installed: captum, jaxtyping, lxt, nnsight, pydantic-settings, pydicom, rad-dino, pytest.
- Nothing needed a secret.
  - STL-10 comes from the public HF dataset `tanganke/stl10`, and the only warning was about unauthenticated HF requests.

## Steps

- Follow `CLAUDE.md` and `.claude/skills/colab-experiments` for the setup rules.
  - Install with `--no-deps -e`, assert `torch.cuda.is_available()` and `"+cu" in torch.__version__`, and print versions.

1. Setup, once per VM:
   ```
   pip install -q uv
   git clone https://github.com/narang99/in-tfm /content/in-tfm
   cd /content/in-tfm && git checkout feat/ig-reports
   uv pip install --system -q --no-deps -e .
   uv pip install --system -q captum jaxtyping lxt nnsight pydantic-settings pydicom rad-dino pytest
   ```
   - Torch stayed at `2.11.0+cu130` after the install.
   - `python -m pytest -q` passed on the VM at the time of the first run (95 tests, before this branch added more).
2. Train the network, in the background, about 12 seconds per epoch:
   ```
   cd /content/in-tfm && python scripts/train_stl.py
   ```
   - It writes `checkpoints/stl_inception/final.pt` and reached 68.3% test accuracy after 15 epochs, matching the laptop run.
3. Compare the methods on the trained network:
   ```
   python scripts/compare_attribution.py --n-images 64 --batch-size 16 \
     --ig-steps '[8,32,128,512]' --out-path /content/compare.json
   ```
4. Build the reports, once per method, with identical flags except for the attribution:
   ```
   python scripts/run_neuron_report.py --config configs/stl_inception_conv.yaml \
     --stl.n-images 300 --hits.max-hits 3000 --neurons 0 --neurons 1 --neurons 2 --neurons 3 \
     --out-dir /content/reports_deeplift
   # the same command, plus:
   #   --stl.attribution integrated-gradients --stl.ig-steps 128 --out-dir /content/reports_ig
   ```
   - The config points at `checkpoints/stl_inception/final.pt`, so step 2 must have run in the same checkout.
   - Run each as a subprocess, not in the kernel.
   - Running both at once on one T4 worked, and took 517 s and 553 s.
   - Run one after the other, they took 339 s and 393 s.
5. Verify before reading any number, see the checks below.
6. Release the VM with `from google.colab import runtime; runtime.unassign()`, and do not run another cell to check.

## Checks that mattered

- Each run exited 0 and printed its final `[total]` line, and the number of files was counted.
  - Each method wrote 362 files, of which 4 are `index.html`, 210 are JPEGs and 140 are `.pt` caches.
- The two runs must show the same hits, so hash the images:
  - `original.jpg` and `hadamard.jpg` must be byte-identical for every cluster, and only `overlay.jpg` may differ.
  - After the fix, 70 of 70 originals and 70 of 70 Hadamard tiles matched, and 0 of 70 overlays did.
- The `clusters` entry of each `meta.json` must be equal between methods.
  - Kept clusters for neurons 0 to 3 were 4, 27, 23 and 16, in both.

## Results

- Neuron 2, 64 test images, each at its own peak position, DeepLift took 0.11 s for all 64.

| IG steps | IG time | speedup | Pearson | top-100 overlap | IG completeness gap |
|---|---|---|---|---|---|
| 8 | 0.17 s | 1.5x | 0.923 | 0.834 | 0.068 |
| 32 | 0.58 s | 5.4x | 0.926 | 0.836 | 0.017 |
| 128 | 2.30 s | 21x | 0.927 | 0.837 | 0.004 |
| 512 | 9.52 s | 87x | 0.927 | 0.838 | 0.001 |

- DeepLift's completeness gap was below 1e-6 in every row.
- Correlation stops improving at 32 steps while IG's gap keeps shrinking, so the remaining difference is between the methods.
- Whole-report time was similar for both (491 s and 528 s), as clustering and scanning dominate.

## Problems hit, in order

- **Unseeded hit sampling.**
  - `sample_cluster` used the global `np.random.permutation`, so two runs displayed different hits per cluster.
  - Symptom: `meta.json` was identical, but only 3 of 70 `original.jpg` files matched.
  - Fixed in `sampled_hit_indices`, and both reports were re-run afterwards.
  - Reports made before that fix cannot be compared with each other.
- **Wrong branch on Colab.**
  - The first IG run failed with `unrecognized arguments: --stl.attribution`, since the VM was on `feat/ig-comparison`.
  - Always check out the branch that has the flag, and print `git log --oneline -1` in the same cell.
- **CLI parsing in tests.**
  - `ComparisonConfig` read pytest's `sys.argv` when it parsed the command line itself.
  - The script now passes `_cli_parse_args=True`, and the class does not parse by default.
- **Flag spelling.** The flags are kebab case, for example `--stl.n-images`, and an underscore gives `unrecognized arguments`.

## Getting the reports off Colab

- The reports are 179 MB and 206 MB, and 159 MB and 187 MB of that is `.pt` tensors.
  - A zip of only `*.html`, `*.jpg`, `*.png` and `*.json` is 31 MB and 445 files.
- `google.colab.files.download` saves to the laptop's Downloads through the open Chrome tab.
  - The first download worked.
  - The second was held by Chrome's "download multiple files" prompt, which needs a click on the laptop.
- An upload to an anonymous temporary file host was blocked by the permission classifier, twice.
  - It needs a permission rule in settings for `mcp__colab-mcp__run_code_cell`, a chat message is not enough.
- Nothing from this run is hosted anywhere, and the VM was released.

## Not done

- One layer, four neurons and ten displayed hits per cluster.
- No judgement of which method's overlays are more meaningful, only that they correlate at about 0.93.
- Clusters came from cuml, and CPU HDBSCAN may cluster differently.
- A page comparing the two sets of reports side by side is not built.
