# STL mini-Inception: following one neuron through training

Run in 2026-10.

## Question

- How does one conv neuron's cluster structure change while the network trains?

## What it did

- `scripts/track_neuron_training.py` trained the STL-10 mini-Inception and saved the network after every step.
  - It logged a per-step distance for the followed neuron to `steps.json`.
- `scripts/report_tracked_steps.py` picked the steps with the largest change.
  - It wrote a normal neuron report per picked state, plus the states before and after training.
  - It matched clusters between states by their mean Hadamard vectors.
  - It wrote one index page with a distance plot.

## Reproduce

- Removed from `main`. The last commit on `main` with it is [`0c79252`](https://github.com/narang99/in-tfm/tree/0c792524ada4568bd24eac7bfea2795820e6a864).
  - [`src/in_tfm/neuron_tracking.py`](https://github.com/narang99/in-tfm/blob/0c792524ada4568bd24eac7bfea2795820e6a864/src/in_tfm/neuron_tracking.py), [`src/in_tfm/tracking_reports.py`](https://github.com/narang99/in-tfm/blob/0c792524ada4568bd24eac7bfea2795820e6a864/src/in_tfm/tracking_reports.py)
  - [`scripts/track_neuron_training.py`](https://github.com/narang99/in-tfm/blob/0c792524ada4568bd24eac7bfea2795820e6a864/scripts/track_neuron_training.py), [`scripts/report_tracked_steps.py`](https://github.com/narang99/in-tfm/blob/0c792524ada4568bd24eac7bfea2795820e6a864/scripts/report_tracked_steps.py)
  - Config: [`configs/stl_neuron_tracking.yaml`](https://github.com/narang99/in-tfm/blob/0c792524ada4568bd24eac7bfea2795820e6a864/configs/stl_neuron_tracking.yaml)
- `git checkout 0c792524a`, then:
  - `uv run python scripts/track_neuron_training.py`
  - `uv run python scripts/report_tracked_steps.py --count 5`
