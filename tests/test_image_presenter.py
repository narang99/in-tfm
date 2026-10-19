import numpy as np
import torch

from in_tfm.presenters import ClusterArtifacts, ClusterHit, ImagePresenter


class Unnormalized:
    image_mean = [0.0]
    image_std = [1.0]


def grayscale_hit() -> ClusterHit:
    rng = np.random.default_rng(0)
    return ClusterHit(
        sample_id="test:0",
        token_idx=0,
        relevance=rng.normal(size=(1, 1, 28, 28)),
        model_input=torch.rand(1, 1, 28, 28),
        hadamard=rng.normal(size=9),
    )


def test_single_channel_images_render(tmp_path):
    ImagePresenter(Unnormalized()).render([grayscale_hit()], ClusterArtifacts(cluster_dir=tmp_path), (3, 3))
    assert {p.name for p in tmp_path.iterdir()} == {"original.jpg", "overlay.jpg", "hadamard.jpg"}
