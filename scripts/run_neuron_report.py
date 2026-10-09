#!/usr/bin/env python
"""Neuron reports for any onboarded model: a language model over wikitext, or rad-dino over DICOMs.

The model name decides the modality, and everything after loading the samples is
`in_tfm.neuron_run.run_neuron_reports`. Settings are in `in_tfm.run_config.RunConfig`, with
defaults from `--config <yaml>` (see `configs/`) and flags on top:

    python scripts/run_neuron_report.py --config configs/rad_dino_fc2.yaml --image.n-dicoms 8
    python scripts/run_neuron_report.py --config configs/gemma3_down_proj.yaml --neurons 90

Clusters group positions by *which input dimensions drove the activation*, so a cluster is a
claim about what kind of input the neuron responds to. The report shows the firing token or
image region for each cluster so that claim is readable at a glance. See in_tfm.sources for
why text hands the model embeddings rather than token ids.

Clustering picks a GPU backend when one is available, see in_tfm.clustering. On CPU, keep
`--image.n-dicoms` small: hit count grows linearly with it while clustering cost grows with the
square of the hit count, so 44 dicoms is ~30x the clustering work of 8, not ~5x.

CLI wrapper around the same steps embs-v2.ipynb walks through interactively. See that notebook
for the exploratory version (elbow plot, spot-checking clusters) this distills.
"""

import sys
import time
from pathlib import Path
from typing import Any

from in_tfm.models import ModelAdapter
from in_tfm.models.attn_only_2l import AttnOnly2LAdapter
from in_tfm.models.gemma3 import Gemma3Adapter
from in_tfm.models.rad_dino import RadDinoAdapter
from in_tfm.models.stl_inception.adapter import StlInceptionAdapter
from in_tfm.neuron_run import run_neuron_reports, timed
from in_tfm.run_config import ModelName, RunConfig, load_run_config
from in_tfm.sources import dicom_paths
from in_tfm.text_corpus import load_texts


def load_adapter(config: RunConfig) -> ModelAdapter[Any]:
    match config.model:
        case ModelName.GEMMA3_270M:
            return Gemma3Adapter.from_pretrained(config.model.value, config.text.max_length)
        case ModelName.ATTN_ONLY_2L:
            return AttnOnly2LAdapter.from_pretrained(config.text.max_length)
        case ModelName.RAD_DINO:
            return RadDinoAdapter.from_pretrained()
        case ModelName.STL_INCEPTION:
            return StlInceptionAdapter.from_pretrained(
                config.layer_name, str(config.stl.root), config.stl.split, config.stl.checkpoint
            )


def load_samples(config: RunConfig) -> list[str] | list[Path] | list[int]:
    if config.modality == "stl":
        return list(range(config.stl.n_images))
    if config.modality == "text":
        texts = load_texts(config.text, config.seed)
        print(f"loaded {len(texts)} {config.text.unit}s")
        return texts
    paths = dicom_paths(config.image.dcm_dir, config.image.n_dicoms)
    print(f"found {len(paths)} dicoms in {config.image.dcm_dir} (requested up to {config.image.n_dicoms})")
    return paths


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
