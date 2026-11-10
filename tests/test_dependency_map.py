import pytest
import torch

from in_tfm.dependencies import (
    Dependency,
    FlattenMap,
    NeuronWriter,
    SupportsDependencies,
    Tap,
    WriterTerm,
    check_terms_rebuild_input,
)
from in_tfm.models.mnist_cnn.adapter import MnistCnnSmallAdapter
from in_tfm.models.mnist_cnn.model import IMAGE_SIZE, MnistCnn, MnistCnnChannels
from in_tfm.token_layout import ConvLayout

CHANNELS, CELLS = 2, 3


@pytest.fixture
def flatten_map() -> FlattenMap:
    return FlattenMap("conv", CHANNELS, CELLS, CHANNELS * CELLS, torch.relu)


@pytest.fixture
def conv_outputs() -> dict[Tap, torch.Tensor]:
    """Cell-major like `ConvLayout`: row = cell, column = channel."""
    return {Tap(layer="conv"): torch.tensor([[1.0, -4.0], [-2.0, 5.0], [3.0, 6.0]])}


def test_a_flattened_dim_is_one_channel_at_one_cell(flatten_map, conv_outputs):
    dependency = flatten_map.dependency(4, 0, conv_outputs)
    (term,) = dependency.terms
    assert term.writer == NeuronWriter(layer="conv", neuron=1, token=1)
    assert term.value == 5.0


def test_the_term_is_the_value_after_the_elementwise_op(flatten_map, conv_outputs):
    (term,) = flatten_map.dependency(3, 0, conv_outputs).terms
    assert (term.value, term.term) == (-4.0, 0.0)


def test_a_map_whose_widths_do_not_match_the_target_is_rejected():
    with pytest.raises(ValueError, match="reads 7 dims"):
        FlattenMap("conv", CHANNELS, CELLS, 7, torch.relu)


def test_terms_that_do_not_sum_to_the_input_are_rejected():
    writer = NeuronWriter(layer="conv", neuron=0, token=0)
    dependency = Dependency(
        dim=1, terms=[WriterTerm(writer=writer, value=2.0, term=2.0)]
    )
    assert check_terms_rebuild_input([dependency], torch.tensor([0.0, 2.0])) == 0.0
    with pytest.raises(ValueError, match="map is wrong"):
        check_terms_rebuild_input([dependency], torch.tensor([2.0, 0.0]))


def captured_head_reads(
    model: MnistCnn, images: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """conv2's output in `ConvLayout` and the head's input, from one forward pass."""
    saved: dict[str, torch.Tensor] = {}
    hooks = [
        model.conv2.register_forward_hook(
            lambda m, i, o: saved.__setitem__("conv2", o)
        ),
        model.head[1].register_forward_hook(
            lambda m, i, o: saved.__setitem__("head", i[0])
        ),
    ]
    with torch.no_grad():
        model(images)
    for hook in hooks:
        hook.remove()
    return ConvLayout(model.conv2).outputs(saved["conv2"]), saved["head"]


def test_the_mnist_head_map_rebuilds_every_input_dim_of_a_real_forward_pass():
    torch.manual_seed(0)
    model = MnistCnn(MnistCnnChannels(conv1=2, conv2=4)).eval()
    adapter = MnistCnnSmallAdapter(model, "head.1", [], "test")
    assert isinstance(adapter, SupportsDependencies)
    dependency_map = adapter.dependency_map_for("head.1")

    conv2_out, head_in = captured_head_reads(
        model, torch.rand(3, 1, IMAGE_SIZE, IMAGE_SIZE)
    )
    for image in range(3):
        outputs = {Tap(layer="conv2"): conv2_out[image]}
        dependencies = [
            dependency_map.dependency(dim, 0, outputs)
            for dim in range(head_in.shape[1])
        ]
        assert check_terms_rebuild_input(dependencies, head_in[image]) < 1e-6


def test_a_conv_target_has_no_map():
    model = MnistCnn(MnistCnnChannels(conv1=2, conv2=4))
    adapter = MnistCnnSmallAdapter(model, "conv2", [], "test")
    with pytest.raises(ValueError, match="conv patches"):
        adapter.dependency_map_for("conv2")
