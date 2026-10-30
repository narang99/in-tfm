"""The fused IVF kernel against the torch path. It needs CUDA, so it runs on Colab and skips elsewhere."""

import numpy as np
import pytest
import torch

from in_tfm.density_peaks import ivf_forest

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the Triton kernel needs CUDA")


def rows_and_lists(seed: int, n: int, distance: str):
    rng = np.random.default_rng(seed)
    centres = rng.normal(scale=4.0, size=(12, 32))
    points = np.abs(centres[rng.integers(0, 12, size=n)] + rng.normal(size=(n, 32)))
    rows = ivf_forest.Rows(points, distance, False, "cuda")
    return rows, ivf_forest.build_lists(rows, 40, seed=0)


@pytest.mark.parametrize("distance", ["weighted_jaccard", "euclidean"])
def test_the_fused_search_finds_what_the_torch_search_finds(distance):
    rows, lists = rows_and_lists(0, 5000, distance)
    queries = torch.arange(len(rows), device="cuda")
    fused = ivf_forest.fused_search(rows, lists, queries, 6, 32, None)
    torch_path = ivf_forest.probed_search(rows, lists, queries, 6, 32)
    assert (fused[0] == torch_path[0]).float().mean() > 0.999
    assert torch.allclose(fused[1], torch_path[1], atol=1e-5)


def test_the_fused_search_keeps_only_denser_points():
    rows, lists = rows_and_lists(1, 4000, "weighted_jaccard")
    ranks = torch.randperm(len(rows), device="cuda")
    queries = torch.arange(0, len(rows), 5, device="cuda")
    fused = ivf_forest.fused_search(rows, lists, queries, 8, 1, ranks)
    denser_only = lambda q, m: ranks[m][None, :] > ranks[q][:, None]  # noqa: E731
    torch_path = ivf_forest.probed_search(rows, lists, queries, 8, 1, denser_only)
    assert torch.equal(fused[0], torch_path[0])
