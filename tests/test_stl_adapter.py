import numpy as np
import pytest
import torch
from PIL import Image

from in_tfm.attribution import compute_deeplift_relevance
from in_tfm.layers import conv_getter
from in_tfm.models.stl_inception import StlInception
from in_tfm.models.stl_inception.adapter import StlInceptionAdapter, tokens_in_layer
from in_tfm.run_config import load_run_config

LAYER = "block_b.branch_3x3.1.0"


class FakeStl:
    def __init__(self, n: int) -> None:
        rng = np.random.default_rng(0)
        self.images = [Image.fromarray(rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)) for _ in range(n)]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        return self.images[index], 0


@pytest.fixture
def adapter():
    return StlInceptionAdapter(StlInception(), LAYER, FakeStl(3), "test")


def test_block_b_has_a_24_by_24_grid_of_positions():
    model = StlInception()
    assert tokens_in_layer(model, conv_getter(LAYER)(model)) == 24 * 24


def test_sample_ids_name_the_split_and_index(adapter):
    assert adapter.make_source([2, 0]).sample_ids() == ["test:2", "test:0"]


def test_the_mask_is_as_wide_as_the_layers_positions(adapter):
    source = adapter.make_source([0, 1])
    batch = source.to_model_batch(list(source.sample_ids()))
    assert batch.valid_mask.shape == (2, 24 * 24)
    assert batch.kwargs["x"].shape == (2, 3, 96, 96)


def test_deeplift_is_exactly_zero_far_from_the_neuron(adapter):
    source = adapter.make_source([0])
    batch = source.to_model_batch(list(source.sample_ids()))
    corner = 0
    relevance = compute_deeplift_relevance(adapter.get_model(), batch, conv_getter(LAYER), 0, corner)
    assert relevance.shape == (1, 3, 96, 96)
    assert relevance.any()
    assert not relevance[..., 48:, 48:].any()


def test_deeplift_attributions_sum_to_the_activation_change_from_the_baseline(adapter):
    """DeepLift's completeness axiom, which fails if a ReLU or pool is not being seen."""
    model = adapter.get_model().eval()
    source = adapter.make_source([0])
    batch = source.to_model_batch(list(source.sample_ids()))
    token, neuron = 100, 5
    seen: list[torch.Tensor] = []
    layer = conv_getter(LAYER)(model)
    handle = layer.register_forward_hook(lambda m, i, o: seen.append(o[:, neuron].flatten(1)[:, token]))
    with torch.no_grad():
        model(batch.kwargs["x"])
        model(torch.zeros_like(batch.kwargs["x"]))
    handle.remove()
    relevance = compute_deeplift_relevance(model, batch, conv_getter(LAYER), neuron, token)
    assert relevance.sum() == pytest.approx((seen[0] - seen[1]).item(), rel=1e-3, abs=1e-4)


def test_conv_target_needs_a_layer_name():
    with pytest.raises(ValueError, match="layer_name"):
        load_run_config(["--model", "self/stl-inception", "--target", "conv"])


def test_hub_images_are_rgb_pairs():
    from in_tfm.models.stl_inception.source import HubImages

    gray = Image.fromarray(np.zeros((96, 96), dtype=np.uint8))
    image, label = HubImages([{"image": gray, "label": 4}])[0]
    assert image.mode == "RGB"
    assert label == 4
