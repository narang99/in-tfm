import torch

from in_tfm.models.stl_inception import StlInception


def test_forward_gives_one_logit_per_class():
    model = StlInception().eval()
    assert model(torch.zeros(2, 3, 96, 96)).shape == (2, 10)


def test_blocks_run_at_the_documented_resolutions():
    model = StlInception().eval()
    x = torch.zeros(1, 3, 96, 96)
    after_b = model.block_b(model.block_a(model.stem(x)))
    assert after_b.shape == (1, 224, 24, 24)
    assert model.block_c(model.downpool(after_b)).shape == (1, 288, 12, 12)
