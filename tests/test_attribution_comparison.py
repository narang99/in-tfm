import pytest
import torch

from in_tfm.attribution import compute_deeplift_relevance, compute_integrated_gradients_relevance
from in_tfm.attribution_comparison import (
    ComparisonConfig,
    completeness_gap,
    measure_batch,
    merge,
    pearson,
    top_k_overlap,
)
from in_tfm.layers import conv_getter
from in_tfm.models.stl_inception import StlInception
from in_tfm.sources import ModelBatch

LAYER = "block_b.branch_3x3.1.0"


@pytest.fixture
def model() -> StlInception:
    torch.manual_seed(0)
    return StlInception().eval()


@pytest.fixture
def images() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(2, 3, 96, 96)


def test_a_map_agrees_perfectly_with_itself(images):
    assert pearson(images, images) == pytest.approx(1.0, abs=1e-5)
    assert top_k_overlap(images, images, k=50) == 1.0


def test_unrelated_maps_share_few_top_pixels():
    a = torch.zeros(1, 3, 96, 96)
    b = torch.zeros(1, 3, 96, 96)
    a[0, 0, :5] = 1.0
    b[0, 0, -5:] = 1.0
    assert top_k_overlap(a, b, k=100) == 0.0


def test_completeness_gap_is_relative_to_the_change():
    attribution = torch.zeros(2, 3, 4, 4)
    attribution[:, 0, 0, 0] = torch.tensor([1.0, 2.0])
    assert completeness_gap(attribution, torch.tensor([2.0, 2.0])) == pytest.approx(0.25)


def test_converged_integrated_gradients_agrees_with_deeplift(model, images):
    config = ComparisonConfig(ig_steps=[4, 64], device="cpu", neuron=2)
    layer = conv_getter(LAYER)(model)
    _, many = merge([measure_batch(model, layer, images, config)], [len(images)])
    assert many.ig_completeness_gap < 0.01
    assert many.pearson > 0.9
    assert many.deeplift_completeness_gap < 1e-3


def test_ig_relevance_has_the_shape_and_the_sum_of_deeplift_relevance(model, images):
    batch = ModelBatch(kwargs={"x": images}, grad_leaf_key="x", valid_mask=torch.ones(2, 24 * 24, dtype=torch.bool))
    args = (model, batch, conv_getter(LAYER), 2, 100)
    deeplift = compute_deeplift_relevance(*args)
    ig = compute_integrated_gradients_relevance(*args, n_steps=128)
    assert ig.shape == deeplift.shape == (2, 3, 96, 96)
    assert ig.sum(axis=(1, 2, 3)) == pytest.approx(deeplift.sum(axis=(1, 2, 3)), rel=0.1, abs=1e-3)
