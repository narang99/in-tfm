#!/usr/bin/env python
"""The pair-by-pair mixed membership report, from a space `space_reports.py` saved.

    python scripts/pair_mixed_report.py --config configs/inception_v1_mixed4e_peaks.yaml \
        --inception.image-dir data/inception_pool --out-dir reports/space_mixed4e --neurons 55

It reads `<out_dir>/neuron_<idx>/space/` and writes `<out_dir>/neuron_<idx>/pair_report/`: an `index.html` and a page per pair.
Only the model pass for the cards runs here, so clustering is not repeated.
"""

import sys
from pathlib import Path

from run_neuron_report import load_adapter, load_samples

from in_tfm.hit_space import confident_hits, entropy_bits, load_space
from in_tfm.neuron_run import polarities_for, prepare_run, report_name, timed
from in_tfm.pair_mixed_report import write_pair_mixed_pages
from in_tfm.run_config import load_run_config
from in_tfm.space_pipeline import density_peaks_of, space_cluster_hits


def main() -> None:
    config = load_run_config(sys.argv[1:])
    neuron_idx = config.neuron_idxs[0]
    neuron_dir = Path(config.out_dir) / report_name(neuron_idx, polarities_for(config)[0])
    space = load_space(neuron_dir / "space")
    max_entropy = density_peaks_of(config.clustering).max_entropy

    run = prepare_run(config, load_adapter(config), load_samples(config))
    hits = space_cluster_hits(space, run, config, neuron_idx, max_entropy)
    with timed("pair pages"):
        index = write_pair_mixed_pages(
            hits, neuron_dir / "pair_report", space.probabilities, entropy_bits(space.probabilities),
            confident_hits(space.probabilities, max_entropy), run.attr_fn,
        )
    print(f"pair index -> {index}")


if __name__ == "__main__":
    main()
