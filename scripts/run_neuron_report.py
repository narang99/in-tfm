#!/usr/bin/env python
"""Neuron reports for a vision model (rad-dino) over a directory of DICOMs.

Everything after loading the samples is `in_tfm.neuron_run.run_neuron_reports`, shared with
run_llm_neuron_report.py. Settings are in `in_tfm.run_config.RunConfig`:

    python scripts/run_neuron_report.py --config configs/rad_dino_fc2.yaml --image.n-dicoms 8

Clustering picks a GPU backend when one is available, see in_tfm.clustering. On CPU, keep
`--image.n-dicoms` small: hit count grows linearly with it while clustering cost grows with the
square of the hit count, so 44 dicoms is ~30x the clustering work of 8, not ~5x.

CLI wrapper around the same steps embs-v2.ipynb walks through interactively. See that notebook
for the exploratory version (elbow plot, spot-checking clusters) this distills.
"""

import sys
import time

from in_tfm.models.rad_dino import RadDinoAdapter
from in_tfm.neuron_run import run_neuron_reports, timed
from in_tfm.run_config import ModelName, load_run_config
from in_tfm.sources import dicom_paths


def main() -> None:
    pipeline_start = time.perf_counter()
    config = load_run_config(sys.argv[1:])
    if config.model is not ModelName.RAD_DINO:
        sys.exit(f"{config.model.value} reads text, use run_llm_neuron_report.py")

    paths = dicom_paths(config.image.dcm_dir, config.image.n_dicoms)
    print(f"found {len(paths)} dicoms in {config.image.dcm_dir} (requested up to {config.image.n_dicoms})")

    with timed("load model"):
        adapter = RadDinoAdapter.from_pretrained()

    run_neuron_reports(config, adapter, paths)
    print(f"[total] {time.perf_counter() - pipeline_start:.2f}s")


if __name__ == "__main__":
    main()
