#!/usr/bin/env python
"""Every report of a neuron from one saved hit space: the cluster report and the bucket report, linked to each other.

    python scripts/space_reports.py --config configs/inception_v1_mixed4e_peaks.yaml \
        --inception.image-dir data/inception_pool --out-dir reports/space_mixed4e

- Each neuron gets `<out_dir>/neuron_<idx>/` with `space/`, `clusters/`, `buckets/`, `index.html` and `manifest.json`.
- A space already collected with the same settings is reused, so a re-run only rebuilds the reports.
- `<out_dir>/index.html` lists the neurons of the run.
See in_tfm.space_pipeline for the stages.
"""

import sys
import time

from run_neuron_report import load_adapter, load_samples

from in_tfm.neuron_run import prepare_run, timed
from in_tfm.run_config import load_run_config
from in_tfm.space_pipeline import run_space_reports


def main() -> None:
    start = time.perf_counter()
    config = load_run_config(sys.argv[1:])
    with timed("load samples"):
        samples = load_samples(config)
    with timed("load model"):
        run = prepare_run(config, load_adapter(config), samples)
    print(f"run index -> {run_space_reports(config, run)}")
    print(f"[total] {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
