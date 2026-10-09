from pathlib import Path

import pytest
from pydantic import ValidationError

from in_tfm.run_config import ModelName, Target, load_run_config


def write_yaml(tmp_path: Path, text: str) -> str:
    path = tmp_path / "run.yaml"
    path.write_text(text)
    return str(path)


def test_defaults_need_no_arguments():
    config = load_run_config([])
    assert config.model is ModelName.GEMMA3_270M
    assert config.neuron_idxs == [90]


def test_file_sets_defaults(tmp_path):
    path = write_yaml(tmp_path, "layer_idx: 3\nclustering: {min_cluster_size: 20}\n")
    config = load_run_config(["--config", path])
    assert config.layer_idx == 3
    assert config.clustering.min_cluster_size == 20


def test_flags_override_the_file(tmp_path):
    path = write_yaml(tmp_path, "layer_idx: 3\n")
    assert load_run_config(["--config", path, "--layer-idx", "5"]).layer_idx == 5


def test_a_flag_overrides_one_key_of_a_section_and_keeps_the_rest(tmp_path):
    path = write_yaml(tmp_path, "clustering: {min_cluster_size: 20, selection_method: eom}\n")
    config = load_run_config(["--config", path, "--clustering.min-cluster-size", "30"])
    assert config.clustering.min_cluster_size == 30
    assert config.clustering.selection_method == "eom"


def test_neurons_come_from_repeated_flags():
    assert load_run_config(["--neurons", "5", "--neurons", "7"]).neuron_idxs == [5, 7]


def test_neuron_range_is_inclusive():
    assert load_run_config(["--neuron-start", "2", "--neuron-end", "4"]).neuron_idxs == [2, 3, 4]


def test_model_is_given_by_its_value():
    assert load_run_config(["--model", "self/attn-only-2l", "--target", "q_proj"]).model is ModelName.ATTN_ONLY_2L


def test_a_target_the_model_lacks_is_rejected_up_front():
    with pytest.raises(ValidationError, match="has no down_proj"):
        load_run_config(["--model", "self/attn-only-2l"])


def test_a_typo_in_the_file_is_an_error(tmp_path):
    path = write_yaml(tmp_path, "clustering: {min_cluster_sise: 20}\n")
    with pytest.raises(ValidationError, match="min_cluster_sise"):
        load_run_config(["--config", path])


def test_environment_variables_do_not_reach_the_config(monkeypatch):
    monkeypatch.setenv("LAYER_IDX", "99")
    monkeypatch.setenv("TARGET", Target.K_PROJ.value)
    config = load_run_config([])
    assert (config.layer_idx, config.target) == (10, Target.DOWN_PROJ)


def test_rad_dino_takes_fc2_and_not_the_language_model_targets():
    config = load_run_config(["--model", "microsoft/rad-dino", "--target", "fc2"])
    assert config.modality == "image"
    with pytest.raises(ValidationError, match="has no down_proj"):
        load_run_config(["--model", "microsoft/rad-dino"])


def test_a_data_section_for_the_other_modality_is_rejected():
    with pytest.raises(ValidationError, match="reads image data, but the text section was set"):
        load_run_config(["--model", "microsoft/rad-dino", "--target", "fc2", "--text.n-samples", "5"])
    with pytest.raises(ValidationError, match="reads text data, but the image section was set"):
        load_run_config(["--image.n-dicoms", "5"])
