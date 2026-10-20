# weighted_jaccard

Weighted Jaccard (Ruzicka) distance between signed vectors.
The math, line by line, is in https://github.com/narang99/in-tfm/pull/91.

## Naive and vectorized mirrors

- `vectorized.py` is the implementation callers use.
  - `__init__.py` exposes it, so `from in_tfm.weighted_jaccard import ...` gets the vectorized functions.
- `naive.py` mirrors it: the same function names, the same arguments, the same results.
  - It uses plain loops, one row and one pair at a time.
  - Arguments that only control speed, such as `chunk`, exist only in the vectorized version.
- Read `naive.py` first to see what a function computes.
  - Then read `vectorized.py` to see how numpy computes the same thing quickly.
- Keep the two in step.
  - A change to a function in one file changes its mirror in the other, in the same commit.
  - A new public function gets a mirror too.
- Tests import both modules and check that they agree.
  - The naive version is fast enough on a few dozen points, so the check is cheap.
  - Run the vectorized side at a small `chunk` too, so the chunked path is checked.
  - See `tests/test_weighted_jaccard.py`.
