#!/usr/bin/env python
"""Cluster reports of value neurons at the drops in test loss of a grokking run, and before and after training.

    uv run python scripts/report_grokking_steps.py --count 5

Needs the output of scripts/train_grokking.py.
The neurons are the most token-selective value neuron of each head in the final network, unless `--neurons` is given.
Each state is a normal neuron report, so `state_NNNNN/neuron_K/index.html` reads like any other.
A state whose reports already exist is skipped, so an interrupted run can be restarted.
"""

import json
import time
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from in_tfm.grokking_tracking import (
    LOSS_PLOT_NAME,
    drop_between,
    largest_drops,
    loss_drops,
    most_token_selective_neurons,
    saved_epochs,
    save_loss_and_distance_plot,
    states_to_report,
    tracking_log_from_states,
    write_index,
)
from in_tfm.grokking_training import METRICS_NAME, EpochMetrics
from in_tfm.models.grokking import GrokkingAdapter, load_model
from in_tfm.models.grokking.data import pairs_for
from in_tfm.neuron_run import run_neuron_reports
from in_tfm.neuron_tracking import state_path
from in_tfm.run_config import load_run_config
from in_tfm.tracking_reports import load_cluster_means, match_clusters


class ReportDropsConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", cli_kebab_case=True, cli_parse_args=True)

    tracking_dir: Path = Path("checkpoints/grokking")
    report_config: Path = Path("configs/grokking_v_proj.yaml")
    stride: int = Field(10, description="epochs between saved states")
    count: int = 5
    min_gap: int = 1
    windows: list[tuple[int, int]] = Field(
        [], description="drops reported as given, as (state before, state after), on top of the largest ones"
    )
    neurons: list[int] | None = None
    out_dir: Path = Path("reports_grokking")
    lr: float = 1e-3


def state_dir(out_dir: Path, state: int) -> Path:
    return out_dir / f"state_{state:05d}"


def report_state(config: ReportDropsConfig, neurons: list[int], state: int) -> None:
    if all((state_dir(config.out_dir, state) / f"neuron_{n}" / "index.html").exists() for n in neurons):
        print(f"state {state}: reports exist, skipped")
        return
    argv = ["--config", str(config.report_config), "--out-dir", str(state_dir(config.out_dir, state))]
    for n in neurons:
        argv += ["--neurons", str(n)]
    argv += ["--grok.checkpoint", str(state_path(config.tracking_dir / "states", state))]
    run = load_run_config(argv)
    adapter = GrokkingAdapter.from_pretrained(str(run.grok.checkpoint))
    run_neuron_reports(run, adapter, pairs_for(run.grok.pairs, run.grok.train_fraction, run.grok.data_seed))


def main() -> None:
    config = ReportDropsConfig()
    metrics = [EpochMetrics.model_validate(m) for m in json.loads((config.tracking_dir / METRICS_NAME).read_text())]
    epochs = saved_epochs(metrics, config.stride)
    final_state = epochs[-1]
    chosen = largest_drops(loss_drops(metrics, config.stride), config.count, config.min_gap)
    chosen = sorted(chosen + [drop_between(metrics, before, after) for before, after in config.windows], key=lambda d: d.before)
    states = states_to_report(chosen, final_state)
    final_model = load_model(str(state_path(config.tracking_dir / "states", final_state)))
    neurons = config.neurons or most_token_selective_neurons(final_model)
    print("drops:", [(d.before, d.after, f"{d.drop:.3f}") for d in chosen])
    print("states:", [(s.state, s.labels) for s in states])
    print("neurons:", neurons)

    config.out_dir.mkdir(parents=True, exist_ok=True)
    for report in states:
        start = time.perf_counter()
        report_state(config, neurons, report.state)
        print(f"state {report.state} ({', '.join(report.labels)}): {time.perf_counter() - start:.0f}s")

    logs = [tracking_log_from_states(config.tracking_dir / "states", epochs, n, config.lr) for n in neurons]
    save_loss_and_distance_plot(metrics, logs, chosen, config.out_dir / LOSS_PLOT_NAME)
    pairs = {
        n: {
            f"{d.before} to {d.after}": match_clusters(
                load_cluster_means(state_dir(config.out_dir, d.before) / f"neuron_{n}"),
                load_cluster_means(state_dir(config.out_dir, d.after) / f"neuron_{n}"),
            )
            for d in chosen
        }
        for n in neurons
    }
    print("index:", write_index(config.out_dir, neurons, chosen, states, pairs))


if __name__ == "__main__":
    main()
