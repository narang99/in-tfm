#!/usr/bin/env python
"""AttnLRP against the exact per-token contribution, on the OV output of the heads' neurons.

    uv run python scripts/grokking_attnlrp_experiment.py --neurons '[3, 40, 70, 100]'

For each neuron (a coordinate of a head's attention-weighted values, read at `=`) and each state:
- the exact contribution of `a`, `b` and `=` is `attention * value`, see `in_tfm.grokking_attribution`.
- plain gradient * input is computed before the AttnLRP rules are patched in, and AttnLRP after.
- both are compared with the exact contribution, and the AttnLRP relevance at each state is compared with the final state's.
"""

import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from in_tfm.grokking_attribution import Agreement, PerToken, agreement, attnlrp_per_token, exact_contributions
from in_tfm.models.grokking import GrokkingAdapter, all_pairs, load_model
from in_tfm.neuron_tracking import state_path


class ExperimentConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", cli_kebab_case=True, cli_parse_args=True)

    tracking_dir: Path = Path("checkpoints/grokking")
    out_dir: Path = Path("reports_grokking")
    states: list[int]
    neurons: list[int]
    chunk: int = 2048


class StateResult(BaseModel):
    state: int
    neuron: int
    gradient_times_input: Agreement
    attnlrp: Agreement
    attnlrp_vs_final_pearson: float | None = None
    mean_abs_share: list[float]
    """AttnLRP's share of absolute relevance on `a`, `b` and `=`."""
    exact_mean_abs_share: list[float]


def relevance_in_chunks(model, tokens, neuron: int, chunk: int) -> PerToken:
    return np.concatenate([attnlrp_per_token(model, tokens[i : i + chunk], neuron) for i in range(0, len(tokens), chunk)])


def shares(values: PerToken) -> list[float]:
    absolute = np.abs(values).mean(0)
    return (absolute / absolute.sum()).tolist()


def save_scatter(relevance: PerToken, contributions: PerToken, path: Path, title: str) -> None:
    fig = Figure(figsize=(5, 5))
    ax = fig.subplots()
    for position, name in enumerate("ab="):
        ax.scatter(contributions[::7, position], relevance[::7, position], s=3, alpha=0.4, label=name)
    ax.set_xlabel("exact contribution (attention x value)")
    ax.set_ylabel("AttnLRP relevance")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)


def main() -> None:
    config = ExperimentConfig()
    tokens, _ = all_pairs()
    models = {s: load_model(str(state_path(config.tracking_dir / "states", s))) for s in config.states}
    exact = {(s, n): exact_contributions(models[s], tokens, n) for s in config.states for n in config.neurons}

    gradient = {
        (s, n): np.concatenate(
            [attnlrp_per_token(models[s], tokens[i : i + config.chunk], n) for i in range(0, len(tokens), config.chunk)]
        )
        for s in config.states
        for n in config.neurons
    }
    adapter = GrokkingAdapter(models[config.states[0]])
    adapter.patch_for_attn_lrp()
    lrp = {(s, n): relevance_in_chunks(models[s], tokens, n, config.chunk) for s in config.states for n in config.neurons}

    final = max(config.states)
    results: list[StateResult] = []
    for s in config.states:
        for n in config.neurons:
            contributions, value = exact[(s, n)]
            results.append(
                StateResult(
                    state=s,
                    neuron=n,
                    gradient_times_input=agreement(gradient[(s, n)], contributions, value, n),
                    attnlrp=agreement(lrp[(s, n)], contributions, value, n),
                    attnlrp_vs_final_pearson=float(np.corrcoef(lrp[(s, n)].ravel(), lrp[(final, n)].ravel())[0, 1]),
                    mean_abs_share=shares(lrp[(s, n)]),
                    exact_mean_abs_share=shares(contributions),
                )
            )
    (config.out_dir / "attnlrp_experiment.json").write_text(json.dumps([r.model_dump() for r in results], indent=1))
    for n in config.neurons:
        save_scatter(lrp[(final, n)], exact[(final, n)][0], config.out_dir / f"attnlrp_vs_exact_neuron_{n}.png", f"state {final}, neuron {n}")
    print("state neuron | pearson(gxi) pearson(lrp) | conserved(gxi) conserved(lrp) | lrp vs final | lrp share a b =")
    for r in results:
        print(
            f"{r.state:5d} {r.neuron:5d} | {r.gradient_times_input.pearson:+.3f} {r.attnlrp.pearson:+.3f} | "
            f"{r.gradient_times_input.conserved:.3f} {r.attnlrp.conserved:.3f} | {r.attnlrp_vs_final_pearson:+.3f} | "
            f"{' '.join(f'{x:.2f}' for x in r.mean_abs_share)}"
        )


if __name__ == "__main__":
    main()
