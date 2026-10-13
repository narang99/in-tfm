#!/usr/bin/env python
"""The last edge of the circuit: what each MLP frequency group writes to the logits.

    uv run python scripts/grokking_logit_edges.py --out experiments_grokking/logit_edges_final.json
"""

import json
from pathlib import Path

import numpy as np
from pydantic_settings import BaseSettings, SettingsConfigDict

from in_tfm.grokking_edges import group_logits, mlp_post_activations, neuron_logit_weights, share_at_sum_frequency
from in_tfm.grokking_fourier import frequency_energy, mlp_frequency_spectra
from in_tfm.grokking_groups import group_by_spectrum
from in_tfm.models.grokking import load_model
from in_tfm.models.grokking.data import all_pairs
from in_tfm.models.grokking.model import MODULUS
from in_tfm.neuron_tracking import state_path


class LogitEdgesConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", cli_kebab_case=True, cli_parse_args=True)

    state: int = 40000
    states_dir: Path = Path("checkpoints/grokking/states")
    out: Path


def correct_logit_margin(logits: np.ndarray, labels: np.ndarray) -> float:
    """Correct logit minus the mean logit, averaged over pairs: how much a path pushes towards the right answer."""
    correct = logits[np.arange(len(labels)), labels]
    return float((correct - logits.mean(axis=1)).mean())


def main() -> None:
    config = LogitEdgesConfig()
    model = load_model(str(state_path(config.states_dir, config.state)))
    spectra = mlp_frequency_spectra(model)
    labels = group_by_spectrum(spectra)
    strongest = spectra.argmax(axis=1) + 1
    post = mlp_post_activations(model)
    weights = neuron_logit_weights(model)
    _, answers = all_pairs()
    answers = answers.numpy()
    total = group_logits(post, weights, np.arange(512))
    results = {"all_mlp_margin": correct_logit_margin(total, answers), "groups": []}
    for label in sorted(set(labels.tolist()) - {-1}):
        members = np.where(labels == label)[0]
        freq = int(np.bincount(strongest[members]).argmax())
        out_freqs = frequency_energy(weights[members].T).sum(axis=1)
        logits = group_logits(post, weights, members)
        results["groups"].append({
            "freq": freq, "size": len(members),
            "output_direction_freq": int(out_freqs.argmax()) + 1,
            "output_direction_share": float(out_freqs.max() / out_freqs.sum()),
            "share_cos_w_a_plus_b_minus_c": share_at_sum_frequency(logits, freq, MODULUS),
            "correct_logit_margin": correct_logit_margin(logits, answers),
        })
        print(results["groups"][-1])
    print("all MLP neurons, correct logit margin", round(results["all_mlp_margin"], 2))
    config.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
