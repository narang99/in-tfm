from pathlib import Path

import pytest
from pydantic import ValidationError

from in_tfm.run_config import DensityPeaksConfig, HdbscanConfig, ModelName, load_run_config


def write_yaml(tmp_path: Path, text: str) -> str:
    path = tmp_path / "run.yaml"
    path.write_text(text)
    return str(path)


def test_defaults_need_no_arguments():
    config = load_run_config([])
    assert config.model is ModelName.GEMMA3_270M
    assert config.neuron_idxs == [90]


def test_file_sets_defaults(tmp_path):
    path = write_yaml(tmp_path, "layer_name: layers.3.mlp.down_proj\nclustering: {strategy: {method: hdbscan, min_cluster_size: 20}}\n")
    config = load_run_config(["--config", path])
    assert config.layer_name == "layers.3.mlp.down_proj"
    assert config.clustering.strategy.min_cluster_size == 20


def test_flags_override_the_file(tmp_path):
    path = write_yaml(tmp_path, "layer_name: layers.3.mlp.down_proj\n")
    assert load_run_config(["--config", path, "--layer-name", "layers.5.mlp.down_proj"]).layer_name == "layers.5.mlp.down_proj"


def test_a_flag_overrides_one_key_of_a_section_and_keeps_the_rest(tmp_path):
    path = write_yaml(tmp_path, "clustering: {strategy: {method: hdbscan, min_cluster_size: 20, selection_method: eom}}\n")
    config = load_run_config(["--config", path, "--clustering.strategy.min-cluster-size", "30"])
    assert config.clustering.strategy == HdbscanConfig(min_cluster_size=30, selection_method="eom")


def test_normalising_is_on_unless_turned_off():
    assert load_run_config([]).clustering.normalise
    assert not load_run_config(["--clustering.normalise", "false"]).clustering.normalise


def test_hdbscan_is_the_default_strategy():
    assert load_run_config([]).clustering.strategy == HdbscanConfig()


def test_density_peaks_is_picked_by_its_method(tmp_path):
    path = write_yaml(tmp_path, "clustering: {distance: weighted_jaccard, strategy: {method: density_peaks}}\n")
    config = load_run_config(["--config", path, "--clustering.strategy.k", "7"])
    assert config.clustering.strategy == DensityPeaksConfig(k=7)
    assert config.clustering.distance == "weighted_jaccard"


def test_a_setting_of_the_other_strategy_is_an_error():
    with pytest.raises(ValidationError, match="Extra inputs"):
        load_run_config(["--clustering.strategy.method", "hdbscan", "--clustering.strategy.k", "7"])


def test_a_strategy_without_a_method_is_hdbscan():
    config = load_run_config(["--clustering.strategy.min-cluster-size", "20"])
    assert config.clustering.strategy == HdbscanConfig(min_cluster_size=20)
    with pytest.raises(ValidationError, match="Extra inputs"):
        load_run_config(["--clustering.strategy.k", "7"])


@pytest.mark.parametrize("path", sorted(Path("configs").glob("*.yaml")), ids=lambda path: path.name)
def test_every_checked_in_config_loads(path):
    load_run_config(["--config", str(path)])


def test_neurons_come_from_repeated_flags():
    assert load_run_config(["--neurons", "5", "--neurons", "7"]).neuron_idxs == [5, 7]


def test_neuron_range_is_inclusive():
    assert load_run_config(["--neuron-start", "2", "--neuron-end", "4"]).neuron_idxs == [2, 3, 4]


def test_model_is_given_by_its_value():
    assert load_run_config(["--model", "self/attn-only-2l"]).model is ModelName.ATTN_ONLY_2L


def test_a_typo_in_the_file_is_an_error(tmp_path):
    path = write_yaml(tmp_path, "clustering: {strategy: {method: hdbscan, min_cluster_sise: 20}}\n")
    with pytest.raises(ValidationError, match="min_cluster_sise"):
        load_run_config(["--config", path])


def test_environment_variables_do_not_reach_the_config(monkeypatch):
    monkeypatch.setenv("LAYER_NAME", "layers.0.self_attn.k_proj")
    assert load_run_config([]).layer_name == "layers.10.mlp.down_proj"
