from pathlib import Path

import pytest
import torch

from in_tfm.dependencies.view import NoTrim
from in_tfm.dependency_run import check_same_samples, dependency_map_of
from in_tfm.models.mnist_cnn.source import MnistSource
from in_tfm.report_meta import ReportMeta
from in_tfm.run_config import load_run_config


def write_yaml(tmp_path: Path, text: str) -> str:
    path = tmp_path / "run.yaml"
    path.write_text(text)
    return str(path)


def test_dependency_defaults_are_the_raw_ones():
    config = load_run_config([]).dependencies
    assert (config.dim_mass, config.writer_mass) == (0.9, 0.9)
    assert (config.min_fired_ratio, config.threshold) == (0.1, 0.8)
    assert config.view.describe().endswith("shown until 90% of the total share")


def test_the_view_policy_is_set_from_the_file(tmp_path):
    path = write_yaml(
        tmp_path, "dependencies: {view: {trim: {method: none}}, dim_mass: 0.5}\n"
    )
    config = load_run_config(["--config", path]).dependencies
    assert config.view.trim == NoTrim()
    assert config.selection.dim_mass == 0.5


def test_an_adapter_without_a_map_cannot_make_dependency_reports():
    with pytest.raises(ValueError, match="no dependency map"):
        dependency_map_of(torch.nn.Linear(2, 2), "head")


def test_a_report_made_from_other_samples_is_rejected(tmp_path):
    meta = ReportMeta(
        neuron_idx=0,
        picks=[],
        n_hits=0,
        min_uniq_images_per_cluster=1,
        hadamard_shape=(1, 1),
        clusters={},
        sample_ids=["test:0", "test:1"],
    )
    same = MnistSource([], [0, 1], "test", n_tokens=1)
    check_same_samples(meta, same, tmp_path)
    with pytest.raises(ValueError, match="other samples"):
        check_same_samples(meta, MnistSource([], [1, 2], "test", n_tokens=1), tmp_path)
