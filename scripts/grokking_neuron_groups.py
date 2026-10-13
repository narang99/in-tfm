#!/usr/bin/env python
"""Groups the neurons of grokking states by their spectra, checks each group by ablation, and follows the groups through training.

    uv run python scripts/grokking_neuron_groups.py --states 0 1000 4400 6000 40000 --out experiments_grokking/groups.json
"""

import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from in_tfm.grokking_groups import (
    AblationResult, LayerGroups, group_by_spectrum, key_frequency_set, layer_spectra, loss_with_mlp_neurons_set_to_mean, summarize_groups,
)
from in_tfm.models.grokking import load_model
from in_tfm.neuron_tracking import state_path


class GroupsConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", cli_kebab_case=True, cli_parse_args=True)

    states: list[int]
    states_dir: Path = Path("checkpoints/grokking/states")
    out: Path
    n_random: int = 5


class GroupAblation(BaseModel):
    dominant_freq: int
    size: int
    group: AblationResult
    random_same_size: AblationResult


class StateGroups(BaseModel):
    state: int
    key_frequencies: list[int]
    layers: list[LayerGroups]
    intact: AblationResult
    mlp_ablations: list[GroupAblation]


def mean_result(results: list[AblationResult]) -> AblationResult:
    return AblationResult(loss=float(np.mean([r.loss for r in results])), accuracy=float(np.mean([r.accuracy for r in results])))


def analyse_state(state: int, config: GroupsConfig) -> StateGroups:
    model = load_model(str(state_path(config.states_dir, state)))
    spectra = layer_spectra(model)
    labels = {name: group_by_spectrum(s) for name, s in spectra.items()}
    layers = [summarize_groups(name, spectra[name], labels[name]) for name in spectra]
    rng = np.random.default_rng(0)
    ablations = []
    for label, summary in zip(sorted(set(labels["up_proj"].tolist()) - {-1}), layers[0].groups):
        members = np.where(labels["up_proj"] == label)[0].tolist()
        random = [loss_with_mlp_neurons_set_to_mean(model, rng.choice(512, len(members), replace=False).tolist()) for _ in range(config.n_random)]
        ablations.append(GroupAblation(
            dominant_freq=summary.dominant_freq, size=len(members),
            group=loss_with_mlp_neurons_set_to_mean(model, members), random_same_size=mean_result(random),
        ))
    return StateGroups(
        state=state, key_frequencies=key_frequency_set(model), layers=layers,
        intact=loss_with_mlp_neurons_set_to_mean(model, []), mlp_ablations=ablations,
    )


def main() -> None:
    config = GroupsConfig()
    results = []
    for state in config.states:
        result = analyse_state(state, config)
        results.append(result)
        print(state, "keys", result.key_frequencies, [(g.target, g.n_groups, round(g.median_dominant_share, 2)) for g in result.layers])
        config.out.write_text(json.dumps([r.model_dump() for r in results], indent=1))
    print("DONE", config.out)


if __name__ == "__main__":
    main()
