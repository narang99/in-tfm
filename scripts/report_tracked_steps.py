#!/usr/bin/env python
"""Cluster reports for the large steps of a tracked run, and for the network before training and after it.

    uv run python scripts/report_tracked_steps.py --count 5

Needs the output of scripts/track_neuron_training.py.
Each state is a normal neuron report, so `state_NNNN/neuron_K/index.html` reads like any other.
A state whose report already exists is skipped, so an interrupted run can be restarted.
"""

import time
from pathlib import Path

from datasets import load_dataset
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from in_tfm.cluster_units import load_clusters
from in_tfm.models.stl_inception.adapter import StlInceptionAdapter, load_model
from in_tfm.models.stl_inception.source import HubImages, spread_indices
from in_tfm.neuron_run import run_neuron_reports
from in_tfm.neuron_tracking import LOG_NAME, TrackingLog, largest_steps, state_path
from in_tfm.run_config import RunConfig, load_run_config
from in_tfm.tracking_reports import (
    DISTANCE_PLOT_NAME,
    match_clusters,
    save_distance_plot,
    state_report_dir,
    states_to_report,
    write_index,
)


class ReportStepsConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", cli_kebab_case=True, cli_parse_args=True)

    tracking_dir: Path = Path("checkpoints/stl_tracking")
    report_config: Path = Path("configs/stl_neuron_tracking.yaml")
    count: int = 5
    min_gap: int = 1
    always_steps: list[int] = Field([], description="steps reported even when they are not among the largest")
    out_dir: Path | None = Field(None, description="replaces the report config's out_dir")


def report_state(
    config: ReportStepsConfig, run: RunConfig, dataset: HubImages, state: int, reports_dir: Path
) -> Path:
    neuron_dir = state_report_dir(reports_dir, state) / f"neuron_{run.neuron_idxs[0]}"
    if (neuron_dir / "index.html").exists():
        print(f"state {state}: report exists, skipped")
        return neuron_dir
    model = load_model(str(state_path(config.tracking_dir / "states", state)))
    adapter = StlInceptionAdapter(model, run.layer_name, dataset, run.stl.split, run.stl.attribution, run.stl.ig_steps)
    state_config = load_run_config(["--config", str(config.report_config), "--out-dir", str(state_report_dir(reports_dir, state))])
    run_neuron_reports(state_config, adapter, spread_indices(len(dataset), run.stl.n_images))
    return neuron_dir


def main() -> None:
    config = ReportStepsConfig()
    run = load_run_config(["--config", str(config.report_config)])
    log = TrackingLog.model_validate_json((config.tracking_dir / LOG_NAME).read_text())
    chosen = largest_steps(log.steps, config.count, config.min_gap)
    chosen += [r for r in log.steps if r.step in config.always_steps and r not in chosen]
    chosen.sort(key=lambda r: r.step)
    states = states_to_report(chosen, n_states=len(log.steps) + 1)
    print("large steps:", [(r.step, f"{r.cosine_distance:.3e}") for r in chosen])
    print("states:", [(s.state, s.labels) for s in states])

    reports_dir = config.out_dir or run.out_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    dataset = HubImages(load_dataset(run.stl.dataset, split=run.stl.split))
    neuron_dirs = {}
    for report in states:
        start = time.perf_counter()
        neuron_dirs[report.state] = report_state(config, run, dataset, report.state, reports_dir)
        print(f"state {report.state} ({', '.join(report.labels)}): {time.perf_counter() - start:.0f}s")

    clusters = {state: load_clusters(path) for state, path in neuron_dirs.items()}
    pairs = {"initial to final": match_clusters(clusters[0], clusters[len(log.steps)])}
    for r in chosen:
        pairs[f"step {r.step}, before to after"] = match_clusters(clusters[r.step - 1], clusters[r.step])

    save_distance_plot(log, chosen, reports_dir / DISTANCE_PLOT_NAME)
    print("index:", write_index(reports_dir, log, chosen, states, pairs))


if __name__ == "__main__":
    main()
