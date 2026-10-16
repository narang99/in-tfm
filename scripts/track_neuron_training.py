#!/usr/bin/env python
"""Train the STL-10 mini-Inception while following one neuron, saving the network after every step.

    uv run python scripts/track_neuron_training.py

The neuron is the one named in `--report-config` (configs/stl_neuron_tracking.yaml).
Writes `<out-dir>/steps.json` (per-step distances) and `<out-dir>/states/state_NNNN.pt`, state 0 being the initial network.
Pick the steps and write their reports with scripts/report_tracked_steps.py.
"""

from pathlib import Path

from in_tfm.neuron_run import layer_getter_for
from in_tfm.neuron_tracking import LOG_NAME, NeuronTracker
from in_tfm.run_config import load_run_config
from in_tfm.stl_training import TrainConfig, load_hub_splits, train


class TrackedTrainConfig(TrainConfig):
    out_dir: Path = Path("checkpoints/stl_tracking")
    report_config: Path = Path("configs/stl_neuron_tracking.yaml")


def main() -> None:
    config = TrackedTrainConfig()
    print(config)
    run = load_run_config(["--config", str(config.report_config)])
    train_images, test_images = load_hub_splits(config.dataset)
    tracker = NeuronTracker(
        layer_getter_for(run),
        run.layer_name,
        run.neuron_idxs[0],
        config.out_dir / "states",
        len(train_images) // config.batch_size,
    )
    train(config, train_images, test_images, observer=tracker)
    (config.out_dir / LOG_NAME).write_text(tracker.log().model_dump_json(indent=1))
    print(f"tracked {len(tracker.records)} steps of {tracker.layer_name} neuron {tracker.neuron}")


if __name__ == "__main__":
    main()
