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
