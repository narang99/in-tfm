import numpy as np
import pytest
import torch
from PIL import Image

from in_tfm.attribution import compute_deeplift_relevance
from in_tfm.layers import submodule_getter
from in_tfm.models.inception_v1.adapter import InceptionV1Adapter
from in_tfm.models.inception_v1.model import IMAGE_SIZE, AttributableInceptionV1
from in_tfm.models.inception_v1.source import image_paths
from in_tfm.run_config import ModelName, RunConfig

pytest.importorskip("lucent")

LAYER = "mixed4e_1x1_pre_relu_conv"
NEURON = 55


@pytest.fixture(scope="module")
def model() -> AttributableInceptionV1:
    torch.manual_seed(0)
    return AttributableInceptionV1(pretrained=False).eval()


@pytest.fixture
def image_dir(tmp_path):
    rng = np.random.default_rng(0)
    for name in ("n02123045/b.jpg", "n02123045/a.jpg", "n03930630/c.jpg"):
        (tmp_path / name).parent.mkdir(exist_ok=True)
        Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)).save(tmp_path / name)
    return tmp_path


def test_the_forward_matches_lucents_own(model):
    from lucent.modelzoo import inceptionv1

    original = inceptionv1(pretrained=False, redirected_ReLU=False).eval()
    original.load_state_dict(model.state_dict())
    x = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE) * 60
    with torch.no_grad():
        torch.testing.assert_close(model(x), original(x), rtol=1e-4, atol=1e-6)


def test_deeplift_attributions_sum_to_the_activation_change_from_the_baseline(model, image_dir):
    """Fails if the stem's local response norm, a pool or a ReLU is not being seen."""
    adapter = InceptionV1Adapter(model, LAYER, image_dir)
    source = adapter.make_source(image_paths(image_dir)[:1])
    batch = source.to_model_batch(list(source.sample_ids()))
    token = 100
    seen: list[torch.Tensor] = []
    handle = submodule_getter(LAYER)(model).register_forward_hook(lambda m, i, o: seen.append(o[:, NEURON].flatten(1)[:, token]))
    with torch.no_grad():
        model(batch.kwargs["x"])
        model(torch.zeros_like(batch.kwargs["x"]))
    handle.remove()
    relevance = compute_deeplift_relevance(model, batch, submodule_getter(LAYER), NEURON, token)
    assert relevance.sum() == pytest.approx((seen[0] - seen[1]).item(), rel=1e-3, abs=1e-3)


def test_image_paths_are_relative_and_sorted(image_dir):
    assert image_paths(image_dir) == ["n02123045/a.jpg", "n02123045/b.jpg", "n03930630/c.jpg"]


def test_mixed4e_has_a_14_by_14_grid_of_positions(model, image_dir):
    source = InceptionV1Adapter(model, LAYER, image_dir).make_source(["n03930630/c.jpg"])
    batch = source.to_model_batch(["n03930630/c.jpg"])
    assert batch.valid_mask.shape == (1, 14 * 14)
    assert batch.kwargs["x"].shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)


def test_the_input_is_lucids_range(model, image_dir):
    source = InceptionV1Adapter(model, LAYER, image_dir).make_source(["n03930630/c.jpg"])
    x = source.to_model_batch(["n03930630/c.jpg"]).kwargs["x"]
    assert -117.01 <= x.min() and x.max() <= 138.01


def test_the_model_needs_the_conv_target_and_a_layer_name():
    config = RunConfig(model=ModelName.INCEPTION_V1, target="conv", layer_name=LAYER)
    assert config.modality == "inception"
    with pytest.raises(ValueError, match="conv"):
        RunConfig(model=ModelName.INCEPTION_V1, target="down_proj")


def test_another_models_data_section_is_rejected():
    with pytest.raises(ValueError, match="stl"):
        RunConfig(model=ModelName.INCEPTION_V1, target="conv", layer_name=LAYER, stl={"n_images": 3})
