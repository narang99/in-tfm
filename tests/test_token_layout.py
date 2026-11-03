import pytest
import torch

from in_tfm.layers import submodule_getter
from in_tfm.token_layout import ConvLayout


@pytest.mark.parametrize("kernel_size,stride", [(1, 1), (3, 1), (5, 1), (3, 2)])
def test_patches_times_weight_reproduce_the_conv_output(kernel_size, stride):
    conv = torch.nn.Conv2d(4, 6, kernel_size, stride=stride, padding=kernel_size // 2)
    x = torch.randn(2, 4, 9, 9)
    layout = ConvLayout(conv)
    assert conv.bias is not None
    by_patches = layout.inputs(x) @ layout.weight(conv.weight).T + conv.bias
    assert torch.allclose(by_patches, layout.outputs(conv(x)), atol=1e-5)


def test_flat_token_index_is_row_major_over_the_output_grid():
    conv = torch.nn.Conv2d(1, 1, 1)
    out = conv(torch.randn(1, 1, 3, 4))
    tokens = ConvLayout(conv).outputs(out)
    y, x = divmod(6, out.shape[-1])
    assert tokens[0, 6, 0] == out[0, 0, y, x]


def test_unfold_pads_with_zeros_at_the_border():
    conv = torch.nn.Conv2d(1, 1, 3, padding=1)
    patch = ConvLayout(conv).inputs(torch.ones(1, 1, 4, 4))[0, 0]
    assert patch.tolist() == [0, 0, 0, 0, 1, 1, 0, 1, 1]


def test_grouped_convs_are_rejected():
    with pytest.raises(ValueError, match="grouped"):
        ConvLayout(torch.nn.Conv2d(4, 4, 3, groups=2))


def test_submodule_getter_follows_a_dotted_path_through_sequentials():
    model = torch.nn.Sequential(
        torch.nn.ReLU(), torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3))
    )
    conv = submodule_getter("1.0")(model)
    assert isinstance(conv, torch.nn.Conv2d)
    assert conv.kernel_size == (3, 3)
