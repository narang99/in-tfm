#!/usr/bin/env python
"""Dependency reports for the clusters of an existing neuron report.

Give it the config the neuron report ran with, so it finds the reports and loads the same samples:

    python scripts/run_neuron_report.py --config configs/mnist_cnn_small_logit9.yaml
    python scripts/run_dependency_report.py --config configs/mnist_cnn_small_logit9.yaml

Only models whose adapter has a dependency map for the layer can do this, see `in_tfm.dependencies`.
"""

import sys
import time

from in_tfm.dependency_run import run_dependency_reports
from in_tfm.neuron_run import timed
from in_tfm.run_config import load_run_config
from in_tfm.run_inputs import load_adapter, load_samples


def main() -> None:
    start = time.perf_counter()
    config = load_run_config(sys.argv[1:])
    with timed("load samples"):
        samples = load_samples(config)
    with timed("load model"):
        adapter = load_adapter(config)
    run_dependency_reports(config, adapter, samples)
    print(f"[total] {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
