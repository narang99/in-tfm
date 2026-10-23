#!/usr/bin/env python
"""The bucket report, from the arrays `collect_hit_space.py` saved.

    python scripts/bucket_report.py --config configs/inception_v1_mixed4e_peaks.yaml \
        --inception.image-dir data/inception_pool --out-dir runs/mixed4e_55

It reads `<out_dir>/space/` and writes `<out_dir>/bucket_report/`: an `index.html` and a page per bucket.
Only the model pass for the cards runs here, so clustering is not repeated.
"""

import sys
from pathlib import Path

import numpy as np
from cluster_connectivity import entropy_bits
from pair_mixed_report import saved_hits
from run_neuron_report import load_adapter, load_samples

from in_tfm.bucket_graph import peak_hits, sample_hits, sample_map
from in_tfm.bucket_report import write_bucket_report
from in_tfm.hit_buckets import all_buckets
from in_tfm.neuron_run import max_entropy_of, prepare_run, timed
from in_tfm.run_config import load_run_config

MIN_HITS = 10
SAMPLED_PER_BUCKET = 8


def main() -> None:
    config = load_run_config(sys.argv[1:])
    space = Path(config.out_dir) / "space"
    forest, trees = np.load(space / "forest.npz"), np.load(space / "trees.npz")
    probabilities = np.load(space / "probabilities.npy")[forest["distinct_of_hit"]]
    confident = entropy_bits(probabilities) < max_entropy_of(config.clustering)

    buckets = all_buckets(probabilities)
    peaks = peak_hits(forest["distinct_of_hit"], trees["roots"])
    sample = sample_hits(buckets, probabilities, peaks, MIN_HITS, SAMPLED_PER_BUCKET)
    with timed("layout"):
        mapped = sample_map(np.load(space / "distances.npy", mmap_mode="r"), probabilities, sample, peaks)
    print(f"{len(sample)} sampled hits, {mapped.neighbours} neighbours each")

    run = prepare_run(config, load_adapter(config), load_samples(config))
    hits = saved_hits(config, run, space, probabilities)
    with timed("bucket pages"):
        index = write_bucket_report(
            hits, Path(config.out_dir) / "bucket_report", buckets, mapped, probabilities, confident, MIN_HITS, run.attr_fn,
        )
    print(f"bucket index -> {index}")


if __name__ == "__main__":
    main()
