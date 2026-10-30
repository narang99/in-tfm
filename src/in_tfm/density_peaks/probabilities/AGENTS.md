# density_peaks.probabilities

A probability per tree for each point, from geodesic distance to each tree's peak.
`__init__.py` says what it is and why.

## The steps

- `peak_cores`: each tree's points at least `peak_share` as dense as its root.
- `knn_graph`: each point joined to its k nearest, read both ways.
- `geodesic_distances`: the shortest path along that graph from each point to each tree's peak.
- `step_length`: τ, the median distance to the nearest other point.
- `softmax_over_trees`: a softmax of −distance / τ over the trees.
- `geodesic_probabilities` runs them in that order. `calculate_probs` is the name the pipeline calls.
- `geodesic_entropy` gives only each point's entropy, for forests too large for the points x trees table.
  - It leaves out trees more than a margin of steps past a point's nearest peak.
  - With a wide margin it equals the entropy of `geodesic_probabilities`.
- `geodesic_entropy` gives only each point's entropy, for forests too large for the points x trees table.
  - It leaves out trees more than a margin of steps past a point's nearest peak.
  - With a wide margin it equals the entropy of `geodesic_probabilities`.

## Naive and vectorized mirrors

- `vectorized.py` is the implementation callers use.
  - `__init__.py` exposes it, so `from in_tfm.density_peaks.probabilities import ...` gets the vectorized functions.
- `naive.py` mirrors it: the same function names, the same arguments, the same results.
  - It uses plain loops, one point and one edge at a time.
  - Its Dijkstra is written out with a heap, where the vectorized version calls scipy's.
- One result differs in form: the graph.
  - The vectorized `knn_graph` returns a scipy sparse matrix.
  - The naive one returns a list of dicts, each point's neighbours and edge lengths.
  - They hold the same edges. The test compares them as dense matrices.
- Read `naive.py` first to see what a function computes.
  - Then read `vectorized.py` to see how numpy and scipy compute the same thing quickly.
- Keep the two in step.
  - A change to a function in one file changes its mirror in the other, in the same commit.
  - A new public function gets a mirror too.
- Tests import both modules and check that they agree.
  - The naive version is fast enough on a few dozen points, so the check is cheap.
  - See `tests/test_density_peak_probabilities.py`.
