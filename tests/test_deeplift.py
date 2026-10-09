import pytest
import torch
from torch import nn

from in_tfm.deeplift import deeplift_attribution
from in_tfm.layers import conv_getter
from in_tfm.models.stl_inception import StlInception

POSITION = 5


class PoolThenBranches(nn.Module):
    """A pool whose output feeds one branch that has another pool and one that has none.

    Captum's DeepLift was incomplete on exactly this shape.
    """

    def __init__(self) -> None:
        super().__init__()
        self.pre = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())
        self.stem = nn.MaxPool2d(3, 2, 1)
        self.inner = nn.MaxPool2d(3, 1, 1)
        self.a = nn.Sequential(nn.Conv2d(8, 4, 1), nn.ReLU())
        self.b = nn.Sequential(nn.Conv2d(8, 4, 1), nn.ReLU())
        self.out = nn.Conv2d(8, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(self.pre(x))
        return self.out(torch.cat([self.a(self.inner(h)), self.b(h)], dim=1))


def select_position(layer_output: torch.Tensor) -> torch.Tensor:
    return layer_output[:, 0].flatten(1)[:, POSITION]


def change_from_baseline(model, layer, x, baseline) -> torch.Tensor:
    seen: list[torch.Tensor] = []
    handle = layer.register_forward_hook(lambda m, i, o: seen.append(select_position(o)))
    with torch.no_grad():
        model(x)
        model(baseline)
    handle.remove()
    return seen[0] - seen[1]


@pytest.mark.parametrize("seed", range(10))
def test_attributions_sum_to_the_change_from_the_baseline_across_overlapping_pools(seed):
    torch.manual_seed(seed)
    model = PoolThenBranches().eval()
    x = torch.randn(2, 3, 16, 16)
    baseline = torch.zeros_like(x)
    attribution = deeplift_attribution(model, x, model.out, select_position, baseline)
    expected = change_from_baseline(model, model.out, x, baseline)
    assert torch.allclose(attribution.flatten(1).sum(1), expected, atol=1e-4)


def test_a_nonzero_baseline_is_used():
    torch.manual_seed(0)
    model = PoolThenBranches().eval()
    x, baseline = torch.randn(1, 3, 16, 16), torch.randn(1, 3, 16, 16)
    attribution = deeplift_attribution(model, x, model.out, select_position, baseline)
    assert attribution.sum().item() == pytest.approx(change_from_baseline(model, model.out, x, baseline).item(), abs=1e-4)


@pytest.mark.parametrize(
    "layer_name", ["stem.0", "block_a.branch_pool.1.0", "block_b.branch_1x1.0", "block_b.branch_3x3.1.0", "block_d.branch_5x5.1.0"]
)
def test_attributions_are_complete_across_the_real_model(layer_name):
    torch.manual_seed(0)
    model = StlInception().eval()
    x = torch.randn(1, 3, 96, 96)
    layer = conv_getter(layer_name)(model)
    attribution = deeplift_attribution(model, x, layer, select_position)
    expected = change_from_baseline(model, layer, x, torch.zeros_like(x))
    assert attribution.sum().item() == pytest.approx(expected.item(), abs=1e-3, rel=1e-3)
