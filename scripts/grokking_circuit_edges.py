#!/usr/bin/env python
"""Traces the circuit at one state: heads, MLP frequency groups and the edges between them.

    uv run python scripts/grokking_circuit_edges.py --state 40000 --out experiments_grokking/edges_final.json
"""

import json
from pathlib import Path

import numpy as np
from pydantic_settings import BaseSettings, SettingsConfigDict

from in_tfm.grokking_edges import attention_to_operands, head_contributions_to_mlp_neurons, variance_shares
from in_tfm.grokking_fourier import attention_frequency_spectra, mlp_frequency_spectra, mlp_pre_activation_grid
from in_tfm.grokking_groups import group_by_spectrum
from in_tfm.models.grokking import load_model
from in_tfm.neuron_tracking import state_path


class EdgesConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", cli_kebab_case=True, cli_parse_args=True)

    state: int = 40000
    states_dir: Path = Path("checkpoints/grokking/states")
    out: Path


def main() -> None:
    config = EdgesConfig()
    model = load_model(str(state_path(config.states_dir, config.state)))
    spectra = mlp_frequency_spectra(model)
    labels = group_by_spectrum(spectra)
    strongest = spectra.argmax(axis=1) + 1
    pre = mlp_pre_activation_grid(model).reshape(-1, 512)
    shares = variance_shares(head_contributions_to_mlp_neurons(model), pre)
    groups = {}
    for label in sorted(set(labels.tolist()) - {-1}):
        members = labels == label
        groups[label] = {
            "freq": int(np.bincount(strongest[members]).argmax()), "size": int(members.sum()),
            "head_share": [float(s) for s in shares[:, members].mean(axis=1)],
            "head_share_std": [float(s) for s in shares[:, members].std(axis=1)],
        }
    attention = attention_to_operands(model)
    head_freqs = {}
    for target in ("v_proj", "k_proj"):
        coord_freq = attention_frequency_spectra(model, target).argmax(axis=1) + 1
        head_freqs[target] = [np.bincount(coord_freq[h * 32:(h + 1) * 32], minlength=57).nonzero()[0].tolist() for h in range(4)]
        head_freqs[target + "_counts"] = [{int(f): int((coord_freq[h * 32:(h + 1) * 32] == f).sum()) for f in np.unique(coord_freq[h * 32:(h + 1) * 32])} for h in range(4)]
    result = {"groups": groups, "attention_from_equals": attention.tolist(), "head_coordinate_freqs": head_freqs}
    config.out.write_text(json.dumps(result, indent=1))
    for label, g in groups.items():
        print(f"group f{g['freq']} size {g['size']}: head shares", np.round(g["head_share"], 2))
    print("attention from '=' to (a, b, =) per head:\n", np.round(attention, 2))
    print(head_freqs["v_proj_counts"], head_freqs["k_proj_counts"])


if __name__ == "__main__":
    main()
