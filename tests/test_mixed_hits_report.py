import numpy as np

from in_tfm.mixed_hits_report import ENTROPY_BINS, HITS_PER_BIN, sample_by_entropy


def test_the_sample_covers_every_entropy_bin_least_mixed_first():
    entropy = np.concatenate([np.zeros(50), np.linspace(0.5, 2.0, 400)])
    picked = sample_by_entropy(entropy, min_entropy=0.5)
    assert len(picked) == ENTROPY_BINS * HITS_PER_BIN
    assert (entropy[picked] >= 0.5).all()
    assert np.all(np.diff(entropy[picked]) >= 0)
    edges = np.linspace(0.5, 2.0, ENTROPY_BINS + 1)
    assert len(np.unique(np.clip(np.searchsorted(edges, entropy[picked], side="right") - 1, 0, ENTROPY_BINS - 1))) == ENTROPY_BINS


def test_no_mixed_hits_give_an_empty_sample():
    assert len(sample_by_entropy(np.zeros(10), min_entropy=0.5)) == 0
