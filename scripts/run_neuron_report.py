#!/usr/bin/env python
"""Neuron reports for any onboarded model: a language model over a text corpus, MNIST, or InceptionV1 over an image folder.

The model name decides the modality, and everything after loading the samples is
`in_tfm.neuron_run.run_neuron_reports`. Settings are in `in_tfm.run_config.RunConfig`, with
defaults from `--config <yaml>` (see `configs/`) and flags on top:

    python scripts/run_neuron_report.py --config configs/gemma3_l11_down_proj_peaks.yaml --neurons 90
    python scripts/run_neuron_report.py --config configs/inception_v1_mixed4e_peaks.yaml

Clustering picks a GPU backend when one is available, see in_tfm.clustering. On CPU, keep the
corpus small: hit count grows linearly with it while clustering cost grows with the square of
the hit count.
"""

import sys
import time

from in_tfm.neuron_run import run_neuron_reports, timed
from in_tfm.run_config import load_run_config
from in_tfm.run_inputs import load_adapter, load_samples


def main() -> None:
    pipeline_start = time.perf_counter()
    config = load_run_config(sys.argv[1:])

    with timed("load samples"):
        samples = load_samples(config)
    with timed("load model"):
        adapter = load_adapter(config)

    run_neuron_reports(config, adapter, samples)
    print(f"[total] {time.perf_counter() - pipeline_start:.2f}s")


if __name__ == "__main__":
    main()
