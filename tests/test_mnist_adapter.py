import numpy as np
import pytest
import torch
from PIL import Image

from in_tfm.attribution import compute_deeplift_relevance
from in_tfm.layers import conv_getter
from in_tfm.models.mnist_cnn import MnistCnn
from in_tfm.models.mnist_cnn.adapter import MnistCnnAdapter, MnistCnnSmallAdapter
from in_tfm.run_config import ModelName, load_run_config


def fake_rows(n: int) -> list[dict]:
    rng = np.random.default_rng(0)
    return [{"image": Image.fromarray(rng.integers(0, 255, (28, 28), dtype=np.uint8)), "label": 0} for _ in range(n)]


@pytest.fixture
def adapter():
    return MnistCnnAdapter(MnistCnn(MnistCnnAdapter.channels), "conv2", fake_rows(3), "test")


def batch_for(adapter, indices: list[int]):
    source = adapter.make_source(indices)
    return source.to_model_batch(list(source.sample_ids()))


def test_conv2_has_a_6_by_6_grid_of_positions(adapter):
    batch = batch_for(adapter, [2, 0])
    assert adapter.make_source([2, 0]).sample_ids() == ["test:2", "test:0"]
    assert batch.valid_mask.shape == (2, 6 * 6)
    assert batch.kwargs["x"].shape == (2, 1, 28, 28)


def test_pixels_are_raw_so_the_zero_baseline_is_a_blank_image(adapter):
    images = batch_for(adapter, [0]).kwargs["x"]
    assert images.min() >= 0 and images.max() <= 1


def test_deeplift_only_reaches_the_positions_receptive_field(adapter):
    """conv2's first position reads conv1's first 3 rows, which read pixels 0 to 5 through conv1's padding."""
    position = 0
    relevance = compute_deeplift_relevance(adapter.get_model(), batch_for(adapter, [0]), conv_getter("conv2"), 0, position)
    assert relevance.shape == (1, 1, 28, 28)
    assert not relevance[..., 6:, :].any() and not relevance[..., :, 6:].any()


def test_deeplift_attributions_sum_to_the_activation_change_from_a_blank_image(adapter):
    model = adapter.get_model().eval()
    batch = batch_for(adapter, [0])
    token, neuron = 20, 3
    seen: list[torch.Tensor] = []
    handle = model.conv2.register_forward_hook(lambda m, i, o: seen.append(o[:, neuron].flatten(1)[:, token]))
    with torch.no_grad():
        model(batch.kwargs["x"])
        model(torch.zeros_like(batch.kwargs["x"]))
    handle.remove()
    relevance = compute_deeplift_relevance(model, batch, conv_getter("conv2"), neuron, token)
    assert relevance.sum() == pytest.approx((seen[0] - seen[1]).item(), rel=1e-3, abs=1e-5)


def test_an_adapter_refuses_the_other_networks_weights():
    with pytest.raises(ValueError, match="network"):
        MnistCnnSmallAdapter(MnistCnn(MnistCnnAdapter.channels), "conv2", fake_rows(1), "test")


def test_both_networks_are_models_of_their_own():
    small = load_run_config(["--model", "self/mnist-cnn-small", "--target", "conv", "--layer-name", "conv1"])
    assert small.model is ModelName.MNIST_CNN_SMALL
    assert small.modality == "mnist"
    with pytest.raises(ValueError, match="mnist data"):
        load_run_config(["--model", "self/mnist-cnn", "--target", "conv", "--layer-name", "conv1", "--stl.n-images", "3"])


def test_first_per_class_takes_the_same_number_of_each_digit_in_order():
    from in_tfm.models.mnist_cnn.source import first_per_class

    assert first_per_class([7, 2, 7, 7, 2, 1], 2) == [0, 1, 2, 4, 5]


def test_the_linear_head_has_one_position_per_image():
    adapter = MnistCnnAdapter(MnistCnn(MnistCnnAdapter.channels), "head.1", fake_rows(2), "test")
    assert batch_for(adapter, [0, 1]).valid_mask.shape == (2, 1)


def test_deeplift_on_a_logit_sums_to_its_change_from_a_blank_image():
    adapter = MnistCnnAdapter(MnistCnn(MnistCnnAdapter.channels), "head.1", fake_rows(1), "test")
    model = adapter.get_model().eval()
    batch = batch_for(adapter, [0])
    with torch.no_grad():
        change = (model(batch.kwargs["x"]) - model(torch.zeros_like(batch.kwargs["x"])))[0, 7]
    relevance = compute_deeplift_relevance(model, batch, conv_getter("head.1"), 7, 0)
    assert relevance.sum() == pytest.approx(change.item(), rel=1e-3, abs=1e-5)
