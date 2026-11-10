import numpy as np

from in_tfm.neuron_capture import NeuronCapture


def capture_with_batch_size(batch_size: int) -> NeuronCapture:
    capture = NeuronCapture.__new__(NeuronCapture)
    capture.batch_size = batch_size
    return capture


def test_gather_reruns_whole_scan_batches_holding_a_hit():
    capture = capture_with_batch_size(4)
    hits = np.array([1, 2, 9, 9, 10])
    batches = capture._scan_batches_holding(hits, n_samples=11)
    assert [b.tolist() for b in batches] == [[0, 1, 2, 3], [8, 9, 10]]
