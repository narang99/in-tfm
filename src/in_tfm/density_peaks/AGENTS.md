# density_peaks

The density-peak forest over a neuron's hits (Rodriguez and Laio, 2014).
`__init__.py` says what it is and why.

## Layout

- `forest.py` holds what has one implementation.
  - The `DensityPeakForest` and `ForestTrees` types.
  - `comparable_rows`, `merge_copies`, and the root rule: `elbow`, `root_cuts` and `root_mask`.
- `vectorized.py` is the implementation callers use.
  - `__init__.py` exposes it, so `from in_tfm.density_peaks import ...` gets the vectorized functions.
- `naive.py` mirrors it.
- `torch_forest.py` builds the same forest with torch, for a GPU.
  - The pipeline uses it when the run's device is CUDA. See `neuron_run.forest_device`.
  - It compares every pair in float32, so it matches `vectorized.py` up to float32 rounding.
    - Near ties can swap a few density ranks. On mixed4e:55 at 20,000 hits no parent, root or tree changed.
  - On CUDA the float32 distances come from the Triton kernel in `triton_distance.py`.
    - `torch.cdist` with p=1 rereads both rows for every pair, which took hours on 20,000 hits.
  - `tests/test_torch_forest.py` checks its trees and parents against `vectorized.py` on torch's CPU device.
    - The Triton kernel only runs on CUDA, so `scripts/bench_forest.py` compares it with a saved forest on Colab.
  - A change to the forest's rules changes it too, in the same commit.
- `ivf_forest.py` builds the forest from approximate neighbours, for more points than an all-pairs pass can take.
  - Each point is compared only with the points in a few IVF lists, and copies are not merged.
  - With every list probed it is exact. `tests/test_ivf_forest.py` checks that against `vectorized.py`.
  - It has no naive mirror. Its reference is the exact forest.
- `probabilities/` gives each point a probability per tree, from geodesic distance to each tree's peak.
  - It builds on the forest's stored neighbours.
  - It has its own naive and vectorized split. See its AGENTS.md.
- The distance itself comes from `in_tfm.weighted_jaccard`, which has its own naive and vectorized split.

## Naive and vectorized mirrors

- Every public function in `vectorized.py` has a mirror in `naive.py`.
  - The mirror has the same name, the same arguments and the same results.
  - It uses plain loops, one point and one pair at a time.
  - Arguments that only control speed, such as `block_elements`, exist only in the vectorized version.
- Read `naive.py` first to see what a function computes.
  - Then read `vectorized.py` to see how numpy computes the same thing quickly.
- Keep the two in step.
  - A change to a function in one file changes its mirror in the other, in the same commit.
  - A new public function gets a mirror too.
- Tests import both modules and check that they agree.
  - The naive version is fast enough on a few dozen points, so the check is cheap.
  - Run the vectorized side at a small `block_elements` too, so the blocked path is checked.
  - See `tests/test_density_peaks.py`.
